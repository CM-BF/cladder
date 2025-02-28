import re
import warnings
from collections import defaultdict

from causalllm.definitions import ROOT_PATH, missing_step


def partial_replace(template, replacements):
    return re.sub(r'\{(\w+)\}', lambda m: replacements.get(m.group(1), m.group(0)), template)

class FormatDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


class TextInterfaceForLLMs:
    truth2norm = {
        '1': 1,
        '0': 0,

        'yes': 1,
        'entailment': 1,
        'neutral': 0.5,
        'unknown': 0.5,
        'contradiction': 0,
        'not-counterfactual': 0,
        'counterfactual': 1,
        'no': 0,
    }

    prefix2norm = {
        'Yes': 1,
        'No': 0,
    }
    query_list_file = f'{ROOT_PATH}/config/meta_queries.json'
    from efficiency.log import fread
    query_str2id = {i['query_type_str']: i['query_id'] for i in fread(query_list_file, verbose=False)}
    query_type2id = {i['query_type']: i['query_id'] for i in fread(query_list_file, verbose=False)}

    q_type2step_prefix = {
        "graph": "Extract the causal graph",
        "query_type": "Determine the query type",
        "formal_form": "Formalize the query",
        "given_info": "Gather all relevant data",
        "estimand": "Deduce the estimand using causal inference",
        "estimate": "Calculate the estimate",
    }
    #     q_type2prompt_suffix['cot'] = f'''
    # Hint: You can answer the question by following the subquestions below:
    #
    # Step 1) Extract the causal graph: {q_type2prompt_suffix["graph"]}
    #
    # Step 2) Identify the query type: {q_type2prompt_suffix["query_type"]}
    #
    # Step 3) Translate the query to an estimand: {q_type2prompt_suffix["step1"]}
    #
    # Step 4) Collect all the available data: {q_type2prompt_suffix["given_info"]}
    #
    # Step 5) Solve for the estimand: {q_type2prompt_suffix["estimand"]}
    #     '''.strip()

    refusal_to_answer_prefices = [
        'As a',
        "I'm sorry ",
        "neither ",
        "none ",
    ]

    long_query_types = {
        'correlation': 'conditional probability',
        'marginal': 'marginal probability',
        'exp_away': 'explaining away effect',
        'ate': 'average treatment effect',
        'backadj': 'backdoor adjustment set',
        'collider_bias': 'collider bias',
        'ett': 'average treatment effect on treated',
        'nde': 'natural direct effect',
        'nie': 'natural indirect effect',
        'det-counterfactual': 'normal counterfactual question'
    }


    def init_prompt(self):
        r'''
        Compose COT prompts from the steps
        '''
        if missing_step:
            del (self.q_type2step_prefix[missing_step])
        if self.ask_about in self.q_type2step_prefix.keys():
            ask_step = list(self.q_type2step_prefix.keys()).index(self.ask_about)
        else:
            ask_step = len(self.q_type2step_prefix)
        cot_steps = [
            f'Step {i + 1}) {step}: {self.q_type2prompt_suffix[q_type]}'
            for i, (q_type, step) in enumerate(self.q_type2step_prefix.items()) if i <= ask_step
        ]

        cot_steps = '\n\n'.join(["Guidance: Address the question by following the steps below:"] + cot_steps)
        self.q_type2prompt_suffix['cot'] = cot_steps

        for key in ['query_type', 'graph', 'formal_form', 'given_info', ]:
            self.q_type2prompt_suffix[key] += ' Answer concisely.'

    def __init__(self, save_path, ask_about=None, enable_fewshot=False, enable_cot=False, given_cot_until_step=None):
        if ask_about == 'query_type':
            self.prefix2norm = self.query_str2id
            self.truth2norm = self.query_type2id

        from efficiency.log import verbalize_list_of_options
        self.q_type2prompt_suffix = {
            'answer': f'Start your answer with {verbalize_list_of_options(self.prefix2norm)}, followed by additional reasoning or evidence'
                      f' to support your explanation.',
            'graph': 'Identify the causal graph that depicts the relationships in the scenario. {var_notions} '
                     'The diagram should simply consist of edges denoted in "var1 -> var2" format, separated by commas.',
            'query_type': f'Identify the type of query implied by the main question. Choices include'
                          f' {verbalize_list_of_options(self.query_str2id)}. '
                          f'Your answer should only be a term from the list above, enclosed in quotation marks.',
            'formal_form':
                'Translate the query into its formal mathematical expression based on its type, utilizing the "do(·)" notation or counterfactual notations as needed.',
            'given_info': 'Extract all the available data. Your answer should contain '
                          'nothing but marginal probabilities and conditional probabilities in the form "P(...)=..." or "P(...|...)=...", each probability being separated by a semicolon. Stick to the previously mentioned denotations for the variables.',
            'estimand': 'Given all the information above, deduce the estimand using skills such as do-calculus, counterfactual prediction, and the basics of probabilities. Answer step by step.',
            'estimate': 'Insert the relevant data into the estimand, perform basic arithmetic calculations, and derive the '
                          'final answer. Answer step by step.',
            'cot_final': 'Based on all the reasoning above, output one word to answer the initial question {question} with just ' + verbalize_list_of_options(self.prefix2norm),
        }
        self.prefix2norm.update({i: -1 for i in self.refusal_to_answer_prefices})
        self.prefix2norm = dict(sorted(self.prefix2norm.items(), key=lambda i: len(i[0]), reverse=True))

        self.ask_about = ask_about
        self.given_cot_until_step = given_cot_until_step
        self.init_prompt() # Compose COT prompts from the steps
        self.save_path = save_path
        self.enable_fewshot = enable_fewshot
        self.enable_cot = enable_cot


    def prepare_prompt(self, list_of_dicts):
        if list_of_dicts is not None:
            self._prepare_fewshot(list_of_dicts)
            self.data_in = self.prompt_composer(list_of_dicts, self.ask_about)

    def _datum2var_notions(self, datum, keep_var_values=False):
        var_symb_suff = 'name'
        from efficiency.function import rstrip_word
        var_symb2text = {}
        for k, v in datum['old']['variable_mapping'].items():
            if k.endswith(var_symb_suff):
                k = rstrip_word(k, var_symb_suff)
            elif keep_var_values:
                k = k[:-1] + '=' + k[-1:]
            else:
                continue
            var_symb2text[k] = v

        var_notions = [f'Use "{s}" to represent "{t}".' for s, t in var_symb2text.items()]
        var_notions = ' '.join(var_notions)
        return var_notions

    def _prepare_fewshot(self, data):
        r'''
        Compose few-shot examples for each query type
        '''
        self.fewshot_examples = {'correlation': [20946, 22532, 14193],
                                 'ate': [2023, 28681, 27304],
                                 'marginal': [26874, 25572, 16429],
                                 'backadj': [21850, 25098, 5150],
                                 'ett': [19688, 15380, 16853],
                                 'det-counterfactual': [29884, 30531, 30821],
                                 'nde': [7775, 13272, 16351],
                                 'nie': [24877, 17209, 28444],
                                 'exp_away': [17817, 11920, 1935],
                                 'collider_bias': [6720, 7458, 2681]}
        from efficiency.function import flatten_list
        example_ids = flatten_list(self.fewshot_examples.values())
        id2datum = {i['question_id']: i for i in data if i['question_id'] in example_ids}
        self.fewshot_id2datum = id2datum
        ask_about_step = QAComposer.known_steps.index(self.ask_about) if self.ask_about != 'answer' else len(QAComposer.known_steps)
        self.id2fewshotprompt = {id: QAComposer(self).compose_qa_pair(datum, self.enable_cot, guide_cot_until_step=ask_about_step) for id, datum in id2datum.items()}

    def _compose_fewshot_prefix(self, datum):
        exclude_id = datum['question_id']
        examples = {k: [i for i in v if i != exclude_id][0] for k, v in self.fewshot_examples.items()}
        few_shot_prompt = [self.id2fewshotprompt[i] for i in examples.values()]
        few_shot_prompt = '\n----------\n'.join(few_shot_prompt) + '\n----------\n'
        return few_shot_prompt

    def prompt_composer(self, data, ask_about):
        def convert_truth_to_norm(value):
            return self.truth2norm.get(value.lower() if isinstance(value, str) else value, value)

        from copy import deepcopy

        for datum in data:
            truth_norm = convert_truth_to_norm(datum['truth'])
            # -- Complete all thoughts: Change {var_notions} to e.g. Use "X" to represent "eating citrus". Use "V2" to represent "vitmain C". Use "Y" to represent "curly hair" --
            q2prompt = deepcopy(self.q_type2prompt_suffix)
            q2prompt = {k: v.format(var_notions=self._datum2var_notions(datum, keep_var_values=k == 'given_info'),
                                    question=datum['old']['question'])
                        for k, v in q2prompt.items()
                        }
            default_query_suffix = q2prompt[ask_about]
            key = 'raw_prompt_without_q' if ask_about in {'graph', 'given_info'} else 'raw_prompt'
            # TODO: add few-shot to prompt here.
            prompt = f"{datum[key]}\n\n{default_query_suffix}" # Simpliest prompt: Background + Answer_requirement
            if self.enable_cot:
                prompt = f"{datum[key]}\n\n{q2prompt['cot']}" # Concatenate the COT guidance AFTER the prompt
            if self.enable_fewshot:
                prompt = f"{self._compose_fewshot_prefix(datum)}{prompt}" # Concat few-shot examples BEFORE the prompt
            if self.given_cot_until_step:
                ask_about_step = QAComposer.known_steps.index(self.ask_about) if self.ask_about != 'answer' else len(QAComposer.known_steps)
                if ask_about_step <= self.given_cot_until_step:
                    self.given_cot_until_step = ask_about_step
                    warnings.warn(f'ask_about_step is {ask_about_step} <= given_cot_until_step {self.given_cot_until_step}.')
                cot_guide = QAComposer(self).compose_qa_pair(datum, self.enable_cot, guide_cot_until_step=self.given_cot_until_step)
                prompt = f"{prompt}{cot_guide}" # Concat given COT guidance AFTER the prompt

            # del datum['raw_prompt'], datum['raw_prompt_without_q'], datum['old']
            datum.update({
                'prompt': prompt,
                'truth_norm': truth_norm,
            })

        return data

    def convert_to_norm(self, value):
        invalid = -1
        value = str(value).lower().strip().strip('"')

        for prefix, norm in self.prefix2norm.items():
            if value.startswith(prefix.lower()):
                return norm
        return invalid

    def response_processor(self, **kwargs):

        from efficiency.log import fread
        data = fread(self.save_path)
        if data:
            self.data_out = data

            for datum in self.data_out:
                datum['pred_norm'] = self.convert_to_norm(datum['pred'])
                datum.update(kwargs)
            self.save()

    def save(self):
        from efficiency.log import write_dict_to_csv
        write_dict_to_csv(self.data_out, self.save_path, verbose=True)
        # import pandas as pd
        # df = pd.DataFrame(self.data)
        # df.to_csv(self.save(), index=False)
        # print('')


class QAComposer():
    from typing import Dict, Any, Iterable

    known_steps = ['variables', 'graph', 'query_type', 'formal_form', 'given_info',  # parsing
                   'estimand', 'estimate', 'interpretation'  # reasoning
                   ]

    def __init__(self, text_interface):
        self.text_interface = text_interface


    def compose_qa_pair(self, datum, enable_cot, guide_cot_until_step=None):
        if not enable_cot:
            qa_pair = f"{datum['raw_prompt']}\n\n{datum['truth'].capitalize()}"
        else:
            if guide_cot_until_step is None:
                include_steps = self.known_steps
            else:
                include_steps = self.known_steps[:1 + guide_cot_until_step]
            step2answer = self.sub_question_prompts(datum, include_steps=include_steps)
            cot_truth = '\n\n'.join(step2answer)
            qa_pair = f"{datum['raw_prompt']}\n\n{cot_truth}"
        return qa_pair

    def sub_question_prompts(self, data: Dict[str, Any], include_steps: Iterable = ('variables', 'graph')):
        r'''
        Compose question : answer pairs for each sub step

        when given an unordered collection of known steps returns a dict of solved sub-questions and their answers
        when given a list or tuple of known steps returns the corresponding list of solved sub-questions and their answers

        Note: adjustement set questions are not supported
        '''
        self.q_type2step_prefix = self.text_interface.q_type2step_prefix
        known_steps = self.known_steps
        data = data['old']
        # data['reasoning'] = data['old']['reasoning']
        # import pdb;pdb.set_trace()

        assert all([step in known_steps for step in include_steps]), f'include_steps must be a subset of {known_steps}'

        terms = {}

        if 'variables' in include_steps:
            try:
                terms['variables'] = data['reasoning']['step0']
            except:
                # it is the "backadj" case: =========== No reasoning provided ===========
                return [data['answer'].capitalize()]

        if 'graph' in include_steps:
            step1 = data['reasoning']['step1'].replace(',', ', ')
            terms['graph'] = f'Step 1) {self.q_type2step_prefix["graph"]}: {step1}.'

        qtype = data['meta']['query_type']
        query_title = self.text_interface.long_query_types[qtype].lower()
        if 'query_type' in include_steps:
            terms['query_type'] = f'Step 2) {self.q_type2step_prefix["query_type"]}: "{query_title}".'

        formal_form = data['meta']['formal_form']  # should be equivalent to data['reasoning']['step2']
        if 'formal_form' in include_steps:
            terms['formal_form'] = f'Step 3) {self.q_type2step_prefix["formal_form"]}: {formal_form}.'

        if 'given_info' in include_steps:
            step4 = data['reasoning']['step4'].replace('\n', '; ')
            if len(step4) == 0:
                # terms['given_info'] = ''
                step4 = 'No relevant data is provided'
            terms['given_info'] = f'Step 4) {self.q_type2step_prefix["given_info"]}: {step4}.'

        estimand = data['meta'].get('estimand')  # should be equivalent to data['reasoning']['step3']
        if estimand is None:  # for rung 1 questions, estimand is the same as formal_form
            estimand = formal_form
            if qtype == 'collider_bias':
                estimand = '0'
        if 'estimand' in include_steps:
            terms['estimand'] = f'Step 5) {self.q_type2step_prefix["estimand"]}: ' \
                                f'We use causal inference to derive the estimand ' \
                                f'implied by the causal graph for the query type "{query_title}":\n' \
                                f'{formal_form}\n' \
                                f'= {estimand}'
        if 'estimate' in include_steps:
            estimate = data['reasoning']['step5']
            terms['estimate'] = f'Step 6) {self.q_type2step_prefix["estimate"]}:\n' \
                                f'{estimand}\n' \
                                f'= {estimate}'

        end = data['reasoning']['end']
        answer = data['answer']
        if 'interpretation' in include_steps:
            if qtype == 'collider_bias':
                reasoning = data['reasoning']['step3'].replace('.', '')
                terms['interpretation'] = f'Since {reasoning}, ' \
                                          f'the overall answer to the question is {answer}.'
            else:
                terms['interpretation'] = f'Since the estimate for the estimand is {end}, ' \
                                          f'the overall answer to the question is {answer}.'

        if isinstance(include_steps, (list, tuple)):
            return [terms[step] for step in include_steps]
        return terms


if __name__ == '__main__':
    template = "Hello, {name}! You have {count} new messages."
    replacements = {"name": "Alice"}
    print(partial_replace(template, replacements))

    # defaultdict().__missing__()
    # dict().__missing__()
    fdict = FormatDict(replacements)
    # fdict.update(replacements)
    print(template.format_map(fdict))