from pathlib import Path
from typing import Literal, Any, List
from tqdm import tqdm
import pandas as pd
from itertools import product
import networkx as nx

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph, START

from causalllm.data import DataFileList
from causalllm.evaluate import Scorer
from causalllm.prompt_utils import partial_replace, TextInterfaceForLLMs
from causalllm.definitions import ROOT_PATH, missing_step, paraphrase_i
from causalllm.structured_data_template import MultiAgentData, SCGString
from causalllm.graph_parser import parse_graph_from_text, draw_graph
from causalllm.langgraph_builder import AgentState, ConfigSchema, executor
from graph_parser import visualize_langgraph



def scg_design(just_scoring, model_version):
    if not just_scoring:

        model = init_chat_model(model_version, model_provider="openai")
        messages = [
            SystemMessage(Path('agents/SCG_system.txt').read_text()),
            HumanMessage(Path('agents/SCG_user.txt').read_text()),
        ]

        pred = model.invoke(messages).content

        SCG_extractor_template = ChatPromptTemplate.from_messages([
            ("system", Path('agents/SCG_extractor_system.txt').read_text()),
            ("human", "{text}")
        ])

        SCG_extractor = SCG_extractor_template | model.with_structured_output(schema=SCGString)

        SCG = SCG_extractor.invoke({'text': pred})
        scg_string = SCG.scg_string
        # short_scg_string = SCG.short_scg_string

        G, nodes = parse_graph_from_text(scg_string)
        draw_graph(G, nodes)
        nodes_str = '\n'.join([f'{n}: {l}' for n, l in nodes.items()])
        merged_edge_list = [', '.join(list(edge[0] for edge in G.in_edges(node))) + ' -> ' + node for node in
                            nodes.keys() if len(G.in_edges(node)) > 0]
        merged_edge_str = '\n'.join(merged_edge_list)

        prompter_messages = [
            SystemMessage(Path('agents/prompter_system.txt').read_text()),
            HumanMessage(partial_replace(Path('agents/prompter_user.txt').read_text(),
                                         {'nodes': nodes_str, 'edges': merged_edge_str})),
        ]  # Can use template but need to use {{}} to excaped the curly braces replacement
        pred_prompts = model.invoke(prompter_messages).content

        extractor_template = ChatPromptTemplate.from_messages([
            ("system", Path('agents/extractor_system.txt').read_text()),
            ("human", "{text}")
        ])

        extractor = extractor_template | model.with_structured_output(schema=MultiAgentData)
        extracted_multi_agent_data = extractor.invoke({'text': pred_prompts})
        multi_agent_data = {st.subtask_id: st for st in extracted_multi_agent_data.subtasks}

        execution_order = list(nx.bfs_layers(G, "X"))
        linear_order = [node for layer in execution_order for node in layer]

        builder = StateGraph(AgentState, ConfigSchema)
        for node in G.nodes:
            builder.add_node(node, executor)
        builder.add_edge(START, "X")
        for s, t in G.edges:
            builder.add_edge(s, t)
        builder.add_edge("Y", END)

        graph = builder.compile()

        visualize_langgraph(graph, ROOT_PATH / 'outputs')
        return graph, multi_agent_data
    else:
        return None, None


class Tester:

    def __init__(self):
        from efficiency.function import set_seed
        set_seed()

    # def cot(self, query: Any, chat: Any, max_tokens: int, cot_final: str):
    #     datum = {}
    #     queries = [query, cot_final.format(query['question'])]
    #     for query_i, query in enumerate(queries):
    #         response = chat.ask(
    #             query,
    #             max_tokens=max_tokens if query_i else 1024,
    #         )
    #
    #         datum[f'query{query_i}'] = query
    #         datum[f'pred{query_i}'] = response
    #     return response

    def run_default_test(self, just_scoring: bool=False, enable_cot: bool=False,
                         enable_fewshot: bool=False, model_versions: List[str]=[],
                         given_cot_until_step: int = None,
                         ask_about: Literal['answer', 'graph', 'query_type', 'formal_form', 'given_info', 'estimand', 'estimate']='answer',
                         majority_vote: bool=False):
        """
        Args:
            just_scoring: if True, the function will only score the existing responses in the files
            enable_cot: if True, the function will guide the model to answer the question step by step
            enable_fewshot: if True, the function will provide the model with a few-shot example before asking the question
            model_versions: a list of model versions to be tested
            ask_about: the type of question to be asked: 'answer', 'graph', 'query_type', 'formal_form', 'given_info', 'estimand', 'estimate'
            given_cot_until_step: = [None, 1, 2, 3, 4]
            majority_vote: if True, the function will use majority vote to determine the final answer
        Returns:
        """
        assert given_cot_until_step in [None, 1, 2, 3, 4, 5, 6]

        system_prompt = '''
You are an expert in causal inference. The following question is not a typical commonsense query, but rather a meticulously designed question created by a professor specializing in causal inference, intended to assess the students' mastery of the course content.
        '''.strip()
        max_tokens = 200
        if ask_about == 'answer':
            max_tokens = 1
        elif ask_about == 'query_type':
            max_tokens = 20


        ask_about_suffix = f'_{ask_about.replace("_", "-")}' if ask_about != 'answer' else ''
        missing_step_suffix = f'_no{missing_step.replace("_", "-")}' if missing_step else ''

        write_out_files = []

        if just_scoring:
            combs = list(product(model_versions, [False], range(0, 5))) \
                    + list(product(['gpt4'], [True], range(0, 5)))
            combs = list(product(model_versions, [enable_cot], [paraphrase_i]))

        else:
            combs = list(product(model_versions, [enable_cot], [paraphrase_i]))

        print(combs)
        # if not just_scoring: import pdb;pdb.set_trace()
        for model_version, enable_cot, para_i in combs:
            if 'gpt' not in model_version:
                max_tokens += 2

            # == make file name ==
            cot_suffix = 'cot' if enable_cot else ''
            if cot_suffix:
                cot_suffix += str(given_cot_until_step) if given_cot_until_step is not None else ''
            fewshot_suffix = '_10shot' if enable_fewshot else ''
            para_suffix = f'_pa{para_i}' if para_i else ''
            majority_vote_suffix = '_majority' if majority_vote else ''
            assert majority_vote and ask_about == 'query_type' or not majority_vote

            write_out_file = \
                f'{ROOT_PATH}/outputs/My{model_version}{cot_suffix}' \
                f'{fewshot_suffix}{ask_about_suffix}{missing_step_suffix}{para_suffix}{majority_vote_suffix}.csv'
            write_out_files.append(write_out_file)
            # == make file name end ==

            # --- Create LLM Text interface taking charge of template/prompt composer/response processor ---
            text_interface = TextInterfaceForLLMs(write_out_file, ask_about=ask_about, enable_fewshot=enable_fewshot,
                                                  enable_cot=enable_cot, given_cot_until_step=given_cot_until_step)

            graph, multi_agent_data = scg_design(just_scoring, 'gpt-4o')

            df_list = DataFileList(ask_about=ask_about)
            for data_file_obj in df_list.data_objs:
                if not just_scoring:
                    text_interface.prepare_prompt(data_file_obj.data)
                    data = text_interface.data_in

                    tqdm_desc = f'Model={model_version}, Data={write_out_file}'

                    print(tqdm_desc)
                    for datum_i, datum in tqdm(enumerate(data), desc=tqdm_desc):
                        # query = datum['prompt']
                        variables = {'X': datum['raw_prompt']}
                        agent_response = dict()
                        states = graph.invoke({'variables': variables, 'agent_response': agent_response},
                                              {'configurable': {"model": "gpt-4o-mini", 'multi_agent_data': multi_agent_data}})
                        state_dict = dict(states)

                        datum['pred'] = states['variables']['Y']

                        df = pd.DataFrame(data[:datum_i + 1])
                        df.to_csv(write_out_file, index=False)
                        if datum_i > 100:
                            break

                text_interface.response_processor(model_version=f"{model_version}{cot_suffix}")

        scorer = Scorer(write_out_files, ask_about)


from jsonargparse import CLI

# def main():
#
#     tester = Tester()
#     tester.run_default_test()


if __name__ == '__main__':
    CLI(Tester, parser_mode='omegaconf')
