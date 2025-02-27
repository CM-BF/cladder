import operator
from pathlib import Path
from typing import Literal, Any, List, TypedDict, Annotated, Dict, Tuple

import asyncio

from langchain.chains.question_answering.map_reduce_prompt import messages
from pydantic import BaseModel
from sympy.assumptions.satask import satask
from tqdm import tqdm
import pandas as pd
from itertools import product
import networkx as nx

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph, START
from langchain_core.messages import AnyMessage
from langchain_core.runnables.config import RunnableConfig

from causalllm.data import DataFileList
from causalllm.evaluate import Scorer
from causalllm.prompt_utils import partial_replace, TextInterfaceForLLMs
from causalllm.definitions import ROOT_PATH, missing_step, paraphrase_i
from causalllm.structured_data_template import MultiAgentData, SCGString, Verifier, Reflector
from causalllm.graph_parser import parse_graph_from_text, draw_graph
from causalllm.langgraph_builder import AgentState, ConfigSchema, executor
from graph_parser import visualize_langgraph
from langgraph_builder import models





class SCGState(TypedDict):
    scg_messages: List[AnyMessage]
    scg_window: int
    scg_design_results: Annotated[List[any], operator.add]
    prompter_messages: List[AnyMessage]
    prompter_window: int
    prompter_results: Annotated[List[any], operator.add]
    execution_results: Annotated[List[any], operator.add]
    last_node: Annotated[List[str], operator.add]

def scg_design(states: SCGState, config: RunnableConfig):
    model = models[config["configurable"].get("model", 'gpt-4o')]
    scg_window = states['scg_window']
    if len(states['last_node']) == 0:
        messages = [
            SystemMessage(Path('agents/SCG/SCG_system.txt').read_text()),
            HumanMessage(partial_replace(Path('agents/SCG/SCG_user.txt').read_text(), {'samples': Path(
                'agents/samples.txt').read_text()})),
        ]
    elif states['last_node'][-1] == 'scg_design':
        # self-loop feedback
        messages = states['scg_messages']
        messages.append(HumanMessage(states['scg_design_results'][-1]['feedback']))
    elif states['last_node'][-1] == 'execution_trials':
        # execution feedback
        messages = states['scg_messages']
        messages.append(HumanMessage(states['execution_results'][-1]['feedback']))
        messages = messages[:scg_window] + messages[-2:]
        scg_window += 2

    # Message filtering
    if len(messages) > scg_window + 6:
        messages = messages[:scg_window] + messages[-6:]

    response = model.invoke(messages)
    pred = response.content
    messages.append(response)

    SCG_extractor_template = ChatPromptTemplate.from_messages([
        ("system", Path('agents/SCG/SCG_extractor_system.txt').read_text()),
        ("human", "{text}")
    ])

    SCG_extractor = SCG_extractor_template | model.with_structured_output(schema=SCGString)

    SCG = SCG_extractor.invoke({'text': pred})
    scg_string = SCG.scg_string
    # short_scg_string = SCG.short_scg_string

    G, nodes = parse_graph_from_text(scg_string)
    draw_graph(G, nodes)

    nodes_str = '\n'.join([f'{n}: {l}' for n, l in nodes.items()])
    edge_list = [f'{s} -> {t}' for s, t in G.edges]
    edge_str = '\n'.join(edge_list)
    merged_edge_list = [', '.join(list(edge[0] for edge in G.in_edges(node))) + ' -> ' + node for node in
                        nodes.keys() if len(G.in_edges(node)) > 0]
    merged_edge_str = '\n'.join(merged_edge_list)

    verifier_messages = [
        SystemMessage(Path('agents/verifier/verifier_system.txt').read_text()),
        HumanMessage(partial_replace(Path('agents/verifier/verifier_user.txt').read_text(),
                                     {'solution': f'Nodes:\n\n{nodes_str}\n\nEdges:\n\n{edge_str}',
                                      'requirements': Path('agents/SCG/scg_requirements.txt').read_text()})),
    ]
    verifier_agent = model.with_structured_output(schema=Verifier)
    async def SC_verify(messages):
        tasks = [asyncio.create_task(verifier_agent.ainvoke(messages)) for i in range(10)]
        return await asyncio.gather(*tasks)
    verifier_results = asyncio.run(SC_verify(verifier_messages))
    # run a LLM to summarize the verifier_results, whether it is valid? What is the common feedback if not valid?
    summarizer_message = [
        SystemMessage(Path('agents/summarizer/summarizer_system.txt').read_text()),
        HumanMessage(partial_replace(Path('agents/summarizer/summarizer_user.txt').read_text(),
                                     {'outputs': '\n-----------------\n'.join([f'{i+1}. Valid: {result.valid} - Feedback: {result.feedback}' for i, result in enumerate(verifier_results)]),
                                      'keys': 'valid and feedback'}),),
    ]

    sc_verifier_result = verifier_agent.invoke(summarizer_message)

    return {'scg_design_results': [{'valid': sc_verifier_result.valid, 'feedback': sc_verifier_result.feedback, 'scg_nx': G, 'nodes': nodes, 'nodes_str': nodes_str, 'merged_edge_str': merged_edge_str}],
            'last_node': [config["metadata"]["langgraph_node"]],
            'scg_messages': messages,
            'scg_window': scg_window}

def scg_verifier_route(states: SCGState, config: RunnableConfig):
    if states['scg_design_results'][-1]['valid']:
        return 'good scg'
    else:
        return 'bad scg'

def prompter_generation(states: SCGState, config: RunnableConfig):
    model = models[config["configurable"].get("model", 'gpt-4o')]
    scg_design_results = states['scg_design_results'][-1]
    nodes_str, merged_edge_str = scg_design_results['nodes_str'], scg_design_results['merged_edge_str']

    prompter_window = states['prompter_window']
    if states['last_node'][-1] == 'scg_design':
        # new prompt generation
        prompter_messages = [
            SystemMessage(Path('agents/prompter/prompter_system.txt').read_text()),
            HumanMessage(partial_replace(Path('agents/prompter/prompter_user.txt').read_text(),
                                         {'nodes': nodes_str, 'edges': merged_edge_str})),
        ]  # Can use template but need to use {{}} to excaped the curly braces replacement
    elif states['last_node'][-1] == 'prompter_generation':
        # self-loop feedback
        prompter_messages = states['prompter_messages']
        prompter_messages.append(HumanMessage(states['prompter_results'][-1]['feedback'] + '\n\n Please try again according to the feedback. Note that please output all agents prompts all over again. When I say you do not follow the causal structure, that means your generated prompts are wrong (e.g., you should not have an agent to predict X, it is given). The causal structure is extracted from your generated prompts, you should modify your generated prompts to correct it. Explicitly generating causal structure to cheat is prohibited.'))
    elif states['last_node'][-1] == 'execution_trials':
        prompter_messages = states['prompter_messages']
        prompter_messages.append(HumanMessage('Here are some feedbacks by executing the SCG over several samples: \n\n' + states['execution_results'][-1]['feedback'] + '\n\n Please try again according to the feedback. Note that please output all agents prompts all over again.'))
        prompter_messages = prompter_messages[:prompter_window] + prompter_messages[-2:]
        prompter_window += 2

    if len(prompter_messages) > prompter_window + 6:
        prompter_messages = prompter_messages[:prompter_window] + prompter_messages[-6:]

    response = model.invoke(prompter_messages)
    pred_prompts = response.content
    prompter_messages.append(response)

    extractor_template = ChatPromptTemplate.from_messages([
        ("system", Path('agents/general/extractor_system.txt').read_text()),
        ("human", "{text}")
    ])

    extractor = extractor_template | model.with_structured_output(schema=MultiAgentData)
    extracted_multi_agent_data = extractor.invoke({'text': pred_prompts})
    multi_agent_data = {st.subtask_id: st for st in extracted_multi_agent_data.subtasks}

    verifier_messages = [
        SystemMessage(Path('agents/verifier/verifier_system.txt').read_text()),
        HumanMessage(partial_replace(Path('agents/verifier/verifier_user.txt').read_text(),
                                     {'solution': '\n'.join([', '.join(st.input_variables) + ' -> ' + ', '.join(st.output_variables) for st in extracted_multi_agent_data.subtasks]),
                                      'requirements': partial_replace(Path(
                                          'agents/prompter/prompter_requirements.txt').read_text(), {'merged_edges': f'{merged_edge_str}'})}), ),
    ]
    verifier_agent = model.with_structured_output(schema=Verifier)
    verifier_result = verifier_agent.invoke(verifier_messages)
    return {'prompter_results': [{'valid': verifier_result.valid, 'feedback': verifier_result.feedback, 'multi_agent_data': multi_agent_data}],
            'last_node': [config["metadata"]["langgraph_node"]],
            'prompter_messages': prompter_messages,
            'prompter_window': prompter_window}

def prompt_verfier_route(states: SCGState, config: RunnableConfig):
    if states['prompter_results'][-1]['valid']:
        return 'good prompt'
    else:
        return 'bad prompt'

def scg_execution_trial(states: SCGState, config: RunnableConfig):

    # 1. Build the graph
    G, nodes = states['scg_design_results'][-1]['scg_nx'], states['scg_design_results'][-1]['nodes']
    builder = StateGraph(AgentState, ConfigSchema)
    for node in G.nodes:
        builder.add_node(node, executor)
    builder.add_edge(START, "X")
    for s, t in G.edges:
        builder.add_edge(s, t)
    builder.add_edge("Y", END)

    scg = builder.compile()
    visualize_langgraph(scg, Path(config['configurable']['exp_folder']) / f'langgraph{len(states["execution_results"])}.png')

    text_interface = config['configurable']['text_interface']
    write_out_file = config['configurable']['write_out_file']
    # is_trial_node = config["metadata"]["langgraph_node"] == 'execution_trials'
    data = config['configurable']['data_10']
    multi_agent_data = states['prompter_results'][-1]['multi_agent_data']

    tqdm_desc = f'Model={config["configurable"].get("model", "gpt-4o")}, Data={write_out_file}'
    print(tqdm_desc)

    async def run_sample(datum):
        variables = {'X': datum['raw_prompt']}
        agent_response = dict()
        agent_states = await scg.ainvoke({'variables': variables, 'agent_response': agent_response},
                              {'configurable': {"model": "gpt-4o-mini", 'multi_agent_data': multi_agent_data}})
        state_dict = dict(agent_states)

        datum['pred'] = agent_states['variables']['Y']
        datum['pred_norm'] = text_interface.convert_to_norm(datum['pred'])
        if datum['pred_norm'] != datum['truth_norm']:
            state_dict['variables']['truth'] = datum['truth']
            return state_dict

    async def run_all_samples(data):
        tasks = [asyncio.create_task(run_sample(datum)) for datum in data]
        return await asyncio.gather(*tasks)
    errors = asyncio.run(run_all_samples(data))
    errors = [x for x in errors if x is not None]

    df = pd.DataFrame(data)
    df.to_csv(write_out_file, index=False)


    refl_messages = []
    scg_design_results = states['scg_design_results'][-1]
    nodes_str, merged_edge_str = scg_design_results['nodes_str'], scg_design_results['merged_edge_str']
    input_1 = f'### Input component 1: SCG design\n\nThe predicted solution causal graph that requires the agents to follow is:\n\nVariables:\n\n{nodes_str}\n\nEdges:\n\n{merged_edge_str}\n\n\n'
    input_2 = f'### Input component 2: Prompt generation\n\nThe generated prompts for agents (one agent one variable) are:\n\n{multi_agent_data}\n\n\n'
    input2output = f'Then the output is produced by using the agents with the prompts to execute the solution causal graph in a topological order. The graph input variable X is given as "Sample".\n\n'
    input_components = input_1 + input_2 + input2output
    model = models[config["configurable"].get("model", 'gpt-4o')]
    for sample_states in errors:
        wrong_output = 'The wrongly predicted sample is:\n'
        wrong_output += f"Sample: {sample_states['variables']['X']}\n\n"
        wrong_output += f"Truth: {sample_states['variables']['truth']}\n\n"
        wrong_output += f"Prediction: {sample_states['variables']['Y']}\n\n"
        wrong_output += f"Error: {sample_states['variables']['Y']} != {sample_states['variables']['truth']}\n\n"
        # add details including all variables' contents and agent_response
        wrong_output += f"Details:\n\n"
        wrong_output += f"Predicted Variables in SCG: {sample_states['variables']}\n\n"
        wrong_output += f"Detailed of Agent Response: {sample_states['agent_response']}\n"
        reflector_message = [
            SystemMessage(Path('agents/reflector/system.txt').read_text()),
            HumanMessage(partial_replace(Path('agents/reflector/user.txt').read_text(),
                                         {'input_components': input_components,
                                          'wrong_output': wrong_output})),
        ]
        refl_messages.append(reflector_message)

    reflector_agent = model.with_structured_output(schema=Reflector)

    async def error_reflection(all_error_samples):
        tasks = [asyncio.create_task(reflector_agent.ainvoke(message)) for message in all_error_samples]
        return await asyncio.gather(*tasks)

    reflector_results = asyncio.run(error_reflection(refl_messages))
    # run a LLM to summarize the verifier_results, whether it is valid? What is the common feedback if not valid?
    summarizer_message = [
        SystemMessage(Path('agents/summarizer/summarizer_system.txt').read_text()),
        HumanMessage(partial_replace(Path('agents/summarizer/summarizer_user.txt').read_text(),
                                     {'outputs': '\n-----------------\n'.join(
                                         [f'{i + 1}. Input component that causes the error: {result.input_component} - Feedback: {result.feedback}' for i, result
                                          in enumerate(reflector_results)]),
                                      'keys': 'input_component and feedback'}), ),
    ]

    sc_reflector_result = reflector_agent.invoke(summarizer_message)

    return {'execution_results': [{'num_error': len(errors), 'input_component': sc_reflector_result.input_component, 'feedback': sc_reflector_result.feedback, 'scg': scg, 'multi_agent_data': multi_agent_data}],
            'last_node': [config["metadata"]["langgraph_node"]]}


def post_trial_route(states: SCGState, config: RunnableConfig):
    if states['execution_results'][-1]['num_error'] == 0 or len(states['execution_results']) > 3:
        return 'good execution'
    elif states['execution_results'][-1]['input_component'] == 'SCG design':
        return 'reflect on scg'
    elif states['execution_results'][-1]['input_component'] == 'Prompt generation':
        return 'reflect on prompt'


def scg_full_execution(states: SCGState, config: RunnableConfig):
    best_execution_result = min(states['execution_results'], key=lambda x: x['num_error'])
    scg = best_execution_result['scg']
    visualize_langgraph(scg, Path(config['configurable']['exp_folder']) / f'langgraph_best.png')

    text_interface = config['configurable']['text_interface']
    write_out_file = config['configurable']['write_out_file']
    data = config['configurable']['data_100']
    multi_agent_data = best_execution_result['multi_agent_data']

    tqdm_desc = f'Model={config["configurable"].get("model", "gpt-4o")}, Data={write_out_file}'
    print(tqdm_desc)

    async def run_sample(datum):
        variables = {'X': datum['raw_prompt']}
        agent_response = dict()
        agent_states = await scg.ainvoke({'variables': variables, 'agent_response': agent_response},
                                         {'configurable': {"model": "gpt-4o-mini",
                                                           'multi_agent_data': multi_agent_data}})
        state_dict = dict(agent_states)

        datum['pred'] = agent_states['variables']['Y']
        datum['pred_norm'] = text_interface.convert_to_norm(datum['pred'])
        if datum['pred_norm'] != datum['truth_norm']:
            state_dict['variables']['truth'] = datum['truth']
            return state_dict

    async def run_all_samples(data):
        tasks = [asyncio.create_task(run_sample(datum)) for datum in data]
        return await asyncio.gather(*tasks)

    errors = asyncio.run(run_all_samples(data))
    errors = [x for x in errors if x is not None]

    df = pd.DataFrame(data)
    df.to_csv(write_out_file, index=False)

    return {}



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

            from datetime import datetime
            START_TIME = datetime.now().strftime("%Y%m%d%H%M%S")

            exp_folder = f'{ROOT_PATH}/outputs/{model_version}{cot_suffix}' \
                f'{fewshot_suffix}{ask_about_suffix}{missing_step_suffix}{para_suffix}{majority_vote_suffix}{START_TIME}'
            Path(exp_folder).mkdir(parents=True, exist_ok=True)

            write_out_file = \
                f'{exp_folder}/output.csv'
            write_out_files.append(write_out_file)
            # == make file name end ==

            # --- Create LLM Text interface taking charge of template/prompt composer/response processor ---
            text_interface = TextInterfaceForLLMs(write_out_file, ask_about=ask_about, enable_fewshot=enable_fewshot,
                                                  enable_cot=enable_cot, given_cot_until_step=given_cot_until_step)

            # graph, multi_agent_data = scg_design(just_scoring, 'gpt-4o')

            df_list = DataFileList(ask_about=ask_about)
            for data_file_obj in df_list.data_objs:
                if not just_scoring:
                    text_interface.prepare_prompt(data_file_obj.data)
                    data = text_interface.data_in

                    tqdm_desc = f'Model={model_version}, Data={write_out_file}'

                    data_10 = data[:10]
                    data_100 = data[10:110]

                    # -- trials here --

                    # model = models[model_version]
                    # async def trial_model():
                    #     tasks = [asyncio.create_task(model.ainvoke([SystemMessage(system_prompt), HumanMessage(data_10[i]['raw_prompt'])])) for i in range(10)]
                    #     return await asyncio.gather(*tasks)
                    # output = asyncio.run(trial_model())


                    # --- meta graph ---
                    meta_graph_builder = StateGraph(SCGState, ConfigSchema)
                    meta_graph_builder.add_node('scg_design', scg_design)
                    meta_graph_builder.add_node('prompter_generation', prompter_generation)
                    meta_graph_builder.add_node('execution_trials', scg_execution_trial)
                    meta_graph_builder.add_node('full_execution', scg_full_execution)

                    # - edges -
                    meta_graph_builder.add_edge(START, 'scg_design')
                    meta_graph_builder.add_conditional_edges('scg_design', scg_verifier_route, {'good scg': 'prompter_generation', 'bad scg': 'scg_design'})
                    meta_graph_builder.add_conditional_edges('prompter_generation', prompt_verfier_route, {'good prompt': 'execution_trials', 'bad prompt': 'prompter_generation'})
                    meta_graph_builder.add_conditional_edges('execution_trials', post_trial_route, {'good execution': 'full_execution', 'reflect on scg': 'scg_design', 'reflect on prompt': 'prompter_generation'})
                    meta_graph_builder.add_edge('full_execution', END)

                    meta_graph = meta_graph_builder.compile()
                    visualize_langgraph(meta_graph, Path(exp_folder) / f'langgraph_controller.png')

                    meta_graph.invoke({'scg_messages': [], 'scg_window': 2, 'scg_design_results': [], 'prompter_window': 2, 'prompter_messages': [], 'prompter_results': [], 'execution_results': [], 'last_node': []},
                                      {'configurable': {'model': 'o3-mini',
                                                        'exp_folder': exp_folder,
                                                        'write_out_file': write_out_file,
                                                        'text_interface': text_interface,
                                                        'data_10': data_10,
                                                        'data_100': data_100}})

                    # print(tqdm_desc)
                    # for datum_i, datum in tqdm(enumerate(data), desc=tqdm_desc):
                    #     # query = datum['prompt']
                    #     variables = {'X': datum['raw_prompt']}
                    #     agent_response = dict()
                    #     states = graph.invoke({'variables': variables, 'agent_response': agent_response},
                    #                           {'configurable': {"model": "gpt-4o-mini", 'multi_agent_data': multi_agent_data}})
                    #     state_dict = dict(states)
                    #
                    #     datum['pred'] = states['variables']['Y']
                    #
                    #     df = pd.DataFrame(data[:datum_i + 1])
                    #     df.to_csv(write_out_file, index=False)
                    #     if datum_i > 100:
                    #         break

                text_interface.response_processor(model_version=f"{model_version}{cot_suffix}")

        scorer = Scorer(write_out_files, ask_about)


from jsonargparse import CLI

# def main():
#
#     tester = Tester()
#     tester.run_default_test()


if __name__ == '__main__':
    CLI(Tester, parser_mode='omegaconf')
