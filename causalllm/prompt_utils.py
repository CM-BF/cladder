import json
import random
import re
import warnings
from collections import defaultdict

from causalbenchmark.eval.data_stats_old import enable_cot
from causalllm.definitions import ROOT_PATH, missing_step

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph, START
from langchain_core.messages import AnyMessage
from langchain_core.runnables.config import RunnableConfig
from pathlib import Path
from tqdm import tqdm

from causalllm.structured_data_template import Anonymizer
from copy import deepcopy

models = {'gpt-4o-mini': init_chat_model("gpt-4o-mini", model_provider="openai"),
          'gpt-4o': init_chat_model("gpt-4o", model_provider="openai"),
          'o1': init_chat_model("o1", model_provider="openai"),
          'o1-mini': init_chat_model("o1-mini", model_provider="openai"),
          'o3-mini': init_chat_model("o3-mini", model_provider="openai"),
          'gpt-4': init_chat_model("gpt-4", model_provider="openai"),
          'gpt-3.5-turbo': init_chat_model("gpt-3.5-turbo", model_provider="openai")}


def partial_replace(template, replacements):
    return re.sub(r'\{(\w+)\}', lambda m: replacements.get(m.group(1), m.group(0)), template)

class FormatDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"

class TextInterface:
    truth2norm = {
        '1': 1,
        '0': 0,

        'yes': 1,
        'true': 1,
        'entailment': 1,
        'neutral': 0.5,
        'unknown': 0.5,
        'contradiction': 0,
        'not-counterfactual': 0,
        'counterfactual': 1,
        'no': 0,
        'false': 0,
    }

    prefix2norm = {
        'Yes': 1,
        'No': 0,
    }

    def __init__(self, save_path):
        self.save_path = save_path

    def convert_to_norm(self, value):
        invalid = -1
        value = str(value).lower().strip().strip('"')

        for prefix, norm in self.prefix2norm.items():
            if value.startswith(prefix.lower()):
                return norm
        return invalid

    def convert_truth_to_norm(self, value):
        return self.truth2norm.get(value.lower() if isinstance(value, str) else value, value)

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


class ProntoQATextInterface(TextInterface):
    system_prompt = "You are an expert in logic reasoning. Your task is to answer a logic question based on the given facts."

    prefix2norm = {
        'True': 1,
        'False': 0,
    }
    def __init__(self, save_path, ask_about=None, enable_fewshot=False, enable_cot=False, given_cot_until_step=None):
        super().__init__(save_path)

        from efficiency.log import verbalize_list_of_options
        self.q_type2prompt_suffix = {
            'answer': f'Start your answer with {verbalize_list_of_options(self.prefix2norm)}, followed by additional reasoning or evidence'
                      f' to support your explanation.',
            'direct_answer': f'Answer the question using only {verbalize_list_of_options(self.prefix2norm)}.',
            'thinking_answer': f'Show your work in <think> </think> tags. And return the final answer {verbalize_list_of_options(self.prefix2norm)} in <answer> </answer> tags, for example <answer> True </answer>',
            'cot': "Guidance: Address the question by following the steps below:\n\n1. List the given claim and the facts.\n\n2. Deduce claims using the facts.\n\n3. Answer the question based on the reasoning.\n\n4. The answer is either True or False.",
            'cot_final': 'Based on all the reasoning above, output one word to answer the initial question with just ' + verbalize_list_of_options(
                self.prefix2norm),
        }

    def _prepare_fewshot(self, data):
        r'''
        Compose few-shot examples for each query type
        '''

        data_ids = random.Random(123).sample(list(range(len(data))), 30)
        # from efficiency.function import flatten_list
        # example_ids = flatten_list(self.fewshot_examples.values())
        id2datum = {i: data[i] for i in data_ids}
        self.fewshot_id2datum = id2datum
        self.id2fewshotprompt = {id: [self.compose_raw_query(datum) + '\n\n' + self.compose_response(datum, reasoning=False),
                                      self.compose_raw_query(datum) + '\n\n' + self.compose_response(datum, reasoning=True)] for id, datum in id2datum.items()}

    def _compose_fewshot_prefix(self, datum, cot=False):
        sample_ids = random.sample(list(self.fewshot_id2datum.keys()), 3)
        cot = int(cot)
        few_shot_prompts = [self.id2fewshotprompt[i][cot] for i in sample_ids]
        few_shot_prompt = '\n----------\n'.join(few_shot_prompts) + '\n----------\n'
        return few_shot_prompt

    def get_cot_prompt(self, sampled_data):
        formatted_examples = ""
        for i, entry in enumerate(sampled_data, 1):
            formatted_examples += f"Q: {entry['Facts']} {entry['claims'][0]} {entry['Query']}\n"
            formatted_examples += f"A: {entry['claims'][0]} "
            for j, (claim, next_step) in enumerate(zip(entry['claims'][1:], entry['next_steps'][:-1]), 1):
                formatted_examples += f"{next_step} So {claim} "
            tf = not (("not" in entry['claims'][-1]) ^ ("not" in entry['Query']))
            formatted_examples += f"The answer is {'true' if tf else 'false'}.\n\n"
        return formatted_examples

    def prepare_prompt_sft(self, list_of_dicts, reasoning=False):
        if list_of_dicts is not None:
            for datum in list_of_dicts:
                datum['truth'] = str(not (("not" in datum['claims'][-1]) ^ ("not" in datum['Query'])))
            self._prepare_fewshot(list_of_dicts)
            self.data_in = self.prompt_composer_sft(list_of_dicts, reasoning=reasoning)

    def prompt_composer_sft(self, data, ask_about='answer', reasoning=False):

        for datum in data:
            datum['truth'] = str(not (("not" in datum['claims'][-1]) ^ ("not" in datum['Query'])))
            truth_norm = self.convert_truth_to_norm(datum['truth'])
            # -- Complete all thoughts: Change {var_notions} to e.g. Use "X" to represent "eating citrus". Use "V2" to represent "vitmain C". Use "Y" to represent "curly hair" --
            q2prompt = deepcopy(self.q_type2prompt_suffix)
            # q2prompt = {k: v.format(var_notions=self._datum2var_notions(datum, keep_var_values=k == 'given_info'),
            #                         question=datum['old']['question'])
            #             for k, v in q2prompt.items()
            #             }
            datum['raw_prompt'] = self.compose_raw_query(datum)

            direct_response = self.compose_response(datum, reasoning=False)
            reasoning_response = self.compose_response(datum, reasoning=True)

            # del datum['raw_prompt'], datum['raw_prompt_without_q'], datum['old']
            datum.update({
                'cot': q2prompt['cot'],
                'fewshot': self._compose_fewshot_prefix(datum, cot=False),
                'cot_fewshot': self._compose_fewshot_prefix(datum, cot=True),
                'answer_suffix': self.q_type2prompt_suffix['answer'],
                'direct_answer_suffix': self.q_type2prompt_suffix['direct_answer'],
                'thinking_answer_suffix': self.q_type2prompt_suffix['thinking_answer'],
                'direct_response': direct_response,
                'reasoning_response': reasoning_response,
                'truth_norm': truth_norm,
            })

        return data

    def compose_raw_query(self, datum):
        return f"Given facts: {datum['Facts']}\n\nGiven {datum['claims'][0].strip('.')}, answer the question: {datum['Query']}"

    def compose_response(self, datum, reasoning):
        if not reasoning:
            response = f"{datum['truth'].capitalize()}"
        else:
            thinking = f"Let's think about it step by step. First, we have {datum['claims'][0]}\n\n"
            for j, (claim, next_step) in enumerate(zip(datum['claims'][1:], datum['next_steps'][:-1]), 1):
                thinking += f"{next_step} So {claim}\n\n"
            thinking += f"Therefore, the answer is {datum['truth']}."
            response = f"<think> {thinking} </think>\n<answer> {datum['truth'].capitalize()} </answer>"
        return response




class CladderTextInterface (TextInterface):
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

    system_prompt = "You are an expert in causal inference. The following question is not a typical commonsense query, but rather a meticulously designed question created by a professor specializing in causal inference, intended to assess the students' mastery of the course content."


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
        super().__init__(save_path)
        if ask_about == 'query_type':
            self.prefix2norm = self.query_str2id
            self.truth2norm = self.query_type2id

        from efficiency.log import verbalize_list_of_options
        self.q_type2prompt_suffix = {
            'answer': f'Start your answer with {verbalize_list_of_options(self.prefix2norm)}, followed by additional reasoning or evidence'
                      f' to support your explanation.',
            'direct_answer': f'Answer the question using only {verbalize_list_of_options(self.prefix2norm)}.',
            'thinking_answer': f'Show your work in <think> </think> tags. And return the final answer {verbalize_list_of_options(self.prefix2norm)} in <answer> </answer> tags, for example <answer> Yes </answer>',
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
            'cot_final': 'Based on all the reasoning above, output one word to answer the initial question with just ' + verbalize_list_of_options(self.prefix2norm),
        }
        self.prefix2norm.update({i: -1 for i in self.refusal_to_answer_prefices})
        self.prefix2norm = dict(sorted(self.prefix2norm.items(), key=lambda i: len(i[0]), reverse=True))

        self.ask_about = ask_about
        self.given_cot_until_step = given_cot_until_step
        self.init_prompt() # Compose COT prompts from the steps
        self.enable_fewshot = enable_fewshot
        self.enable_cot = enable_cot


    def prepare_prompt(self, list_of_dicts):
        if list_of_dicts is not None:
            self._prepare_fewshot(list_of_dicts)
            self.data_in = self.prompt_composer(list_of_dicts, self.ask_about)

    def prepare_prompt_sft(self, list_of_dicts, reasoning=False):
        if list_of_dicts is not None:
            self._prepare_fewshot(list_of_dicts)
            self.data_in = self.prompt_composer_sft(list_of_dicts, self.ask_about, reasoning=reasoning)

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
        # self.fewshot_examples = {'correlation': [20946, 22532, 14193],
        #                          'ate': [2023, 28681, 27304],
        #                          'marginal': [26874, 25572, 16429],
        #                          'backadj': [21850, 25098, 5150],
        #                          'ett': [19688, 15380, 16853],
        #                          'det-counterfactual': [29884, 30531, 30821],
        #                          'nde': [7775, 13272, 16351],
        #                          'nie': [24877, 17209, 28444],
        #                          'exp_away': [17817, 11920, 1935],
        #                          'collider_bias': [6720, 7458, 2681]}
        few_shot_class = ['correlation', 'ate', 'marginal', 'backadj', 'ett', 'det-counterfactual', 'nde', 'nie', 'exp_away', 'collider_bias']
        self.fewshot_examples = {k: [] for k in few_shot_class}
        left_class = set(few_shot_class)
        import random
        data_indices = list(range(len(data)))
        random.Random(123).shuffle(data_indices)
        for i in data_indices:
            datum = data[i]
            if datum['query_type'] in left_class:
                self.fewshot_examples[datum['query_type']].append(datum['question_id'])
                if len(self.fewshot_examples[datum['query_type']]) >= 3:
                    left_class.remove(datum['query_type'])
            if not left_class:
                break
        from efficiency.function import flatten_list
        example_ids = flatten_list(self.fewshot_examples.values())
        id2datum = {i['question_id']: i for i in data if i['question_id'] in example_ids}
        self.fewshot_id2datum = id2datum
        ask_about_step = QAComposer.known_steps.index(self.ask_about) if self.ask_about != 'answer' else len(QAComposer.known_steps)
        self.id2fewshotprompt = {id: [QAComposer(self).compose_qa_pair(datum, enable_cot=False, guide_cot_until_step=ask_about_step),
                                      QAComposer(self).compose_qa_pair(datum, enable_cot=True, guide_cot_until_step=ask_about_step)] for id, datum in id2datum.items()}

    def _compose_fewshot_prefix(self, datum, cot=False):
        exclude_id = datum['question_id']
        examples = {k: [i for i in v if i != exclude_id][0] for k, v in self.fewshot_examples.items()}
        cot = int(cot)
        few_shot_prompt = [self.id2fewshotprompt[i][cot] for i in examples.values()]
        few_shot_prompt = '\n----------\n'.join(few_shot_prompt) + '\n----------\n'
        return few_shot_prompt

    def prompt_composer(self, data, ask_about):

        for datum in data:
            truth_norm = self.convert_truth_to_norm(datum['truth'])
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

    def prompt_composer_sft(self, data, ask_about, reasoning=False):

        for datum in data:
            truth_norm = self.convert_truth_to_norm(datum['truth'])
            # -- Complete all thoughts: Change {var_notions} to e.g. Use "X" to represent "eating citrus". Use "V2" to represent "vitmain C". Use "Y" to represent "curly hair" --
            q2prompt = deepcopy(self.q_type2prompt_suffix)
            q2prompt = {k: v.format(var_notions=self._datum2var_notions(datum, keep_var_values=k == 'given_info'),
                                    question=datum['old']['question'])
                        for k, v in q2prompt.items()
                        }
            default_query_suffix = q2prompt[ask_about]
            # key = 'raw_prompt_without_q' if ask_about in {'graph', 'given_info'} else 'raw_prompt'
            # prompt = f"{datum[key]}\n\n{default_query_suffix}" # Simpliest prompt: Background + Answer_requirement

            # response = QAComposer(self).compose_response(datum, reasoning=reasoning)
            direct_response = QAComposer(self).compose_response(datum, reasoning=False)
            reasoning_response = QAComposer(self).compose_response(datum, reasoning=True)

            # del datum['raw_prompt'], datum['raw_prompt_without_q'], datum['old']
            datum.update({
                'cot': q2prompt['cot'],
                'fewshot': self._compose_fewshot_prefix(datum, cot=False),
                'cot_fewshot': self._compose_fewshot_prefix(datum, cot=True),
                'answer_suffix': self.q_type2prompt_suffix['answer'],
                'direct_answer_suffix': self.q_type2prompt_suffix['direct_answer'],
                'thinking_answer_suffix': self.q_type2prompt_suffix['thinking_answer'],
                'direct_response': direct_response,
                'reasoning_response': reasoning_response,
                'truth_norm': truth_norm,
            })

        return data



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

    def compose_response(self, datum, reasoning):
        if not reasoning:
            response = f"{datum['truth'].capitalize()}"
        else:
            step2answer = self.sub_question_prompts(datum, include_steps=self.known_steps)
            thinking = '\n\n'.join(step2answer)
            response = f"<think> {thinking} </think>\n<answer> {datum['truth'].capitalize()} </answer>"
        return response

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