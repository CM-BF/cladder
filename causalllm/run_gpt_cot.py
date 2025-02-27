import operator
from pathlib import Path
from typing import Literal, Any, List, TypedDict, Annotated, Dict, Tuple

import asyncio

from langchain.chains.question_answering.map_reduce_prompt import messages
from pydantic import BaseModel
from sympy.assumptions.satask import satask
from sympy.polys.polyconfig import query
from tqdm import tqdm
import pandas as pd
from itertools import product
import networkx as nx
from tqdm.asyncio import tqdm_asyncio

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
from causalllm.structured_data_template import MultiAgentData, SCGString, Verifier, Reflector, BinaryClassifier
from causalllm.graph_parser import parse_graph_from_text, draw_graph
from causalllm.langgraph_builder import AgentState, ConfigSchema, executor, extractor_template
from graph_parser import visualize_langgraph
from langgraph_builder import models





class ExecState(TypedDict):
    datum: Any
    pred: str

def cot_sample_run(states: ExecState, config: RunnableConfig):
    model = models[config["configurable"].get("model", 'gpt-4o-mini')]
    extractor_model = models['gpt-4o-mini']

    datum = states['datum']
    query = datum['raw_prompt']

    messages = [
        SystemMessage(
            "You are an expert in causal inference. The following question is not a typical commonsense query, but rather a meticulously designed question created by a professor specializing in causal inference, intended to assess the students' mastery of the course content."),
    ]
    if config["configurable"]["model"] == 'o3-mini':
        messages.append(HumanMessage(
            f"The question is: {query}\n\nPlease answer whether the hypothesis is true or not? Reply 'yes' or 'no'."))
        response = model.invoke(messages)
        extractor = extractor_template | extractor_model.with_structured_output(schema=BinaryClassifier)
        classification = extractor.invoke({'text': response.content})
    else:

        messages.append(HumanMessage(f"The question is: {query}\n\nPlease answer the question by proposing chain of thoughts and do reasoning step by step."))

        response = model.invoke(messages)
        messages.append(response)
        messages.append(
            HumanMessage("According to the above reasoning, whether the hypothesis is true or not? Reply 'yes' or 'no'."))
        final_response = model.invoke(messages)
        extractor = extractor_template | extractor_model.with_structured_output(schema=BinaryClassifier)
        classification = extractor.invoke({'text': final_response.content})

    datum['pred'] = classification.answer

    return {'pred': classification.answer}

class CotState(TypedDict):
    finished: bool

def cot_executaion(states: CotState, config: RunnableConfig):
    data = config["configurable"]['data']
    write_out_file = config['configurable']['write_out_file']

    graph_builder = StateGraph(ExecState, ConfigSchema)
    graph_builder.add_node('cot_sample_run', cot_sample_run)
    graph_builder.add_edge(START, 'cot_sample_run')
    graph_builder.add_edge('cot_sample_run', END)
    graph = graph_builder.compile()

    async def run_sample(datum):
        agent_states = await graph.ainvoke({'datum': datum, 'pred': 'null'},
                                         {'configurable': {"model": config['configurable']['model']}})
        datum['pred'] = agent_states['pred']
        return None

    async def run_all_samples(data):
        tasks = [run_sample(datum) for datum in data]
        return await tqdm_asyncio.gather(*tasks, desc=f'Model={config["configurable"].get("model", "gpt-4o")}, Data={write_out_file}', total=len(tasks))

    asyncio.run(run_all_samples(data))
    df = pd.DataFrame(data)
    df.to_csv(write_out_file, index=False)

    return {'finished': True}



class Tester:

    def __init__(self):
        from efficiency.function import set_seed
        set_seed()

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

            exp_folder = f'{ROOT_PATH}/outputs/COT{model_version}{cot_suffix}' \
                f'{fewshot_suffix}{ask_about_suffix}{missing_step_suffix}{para_suffix}{majority_vote_suffix}{START_TIME}'
            Path(exp_folder).mkdir(parents=True, exist_ok=True)
            print(f'Exp folder: {exp_folder}')

            write_out_file = \
                f'{exp_folder}/output.csv'
            write_out_files.append(write_out_file)
            # == make file name end ==

            # --- Create LLM Text interface taking charge of template/prompt composer/response processor ---
            text_interface = TextInterfaceForLLMs(write_out_file, ask_about=ask_about, enable_fewshot=enable_fewshot,
                                                  enable_cot=enable_cot, given_cot_until_step=given_cot_until_step)


            df_list = DataFileList(ask_about=ask_about)
            for data_file_obj in df_list.data_objs:
                if not just_scoring:
                    text_interface.prepare_prompt(data_file_obj.data)
                    data = text_interface.data_in

                    tqdm_desc = f'Model={model_version}, Data={write_out_file}'


                    # --- meta graph ---
                    graph_builder = StateGraph(CotState, ConfigSchema)
                    graph_builder.add_node('cot_execution', cot_executaion)

                    # - edges -
                    graph_builder.add_edge(START, 'cot_execution')
                    graph_builder.add_edge('cot_execution', END)

                    cot_graph = graph_builder.compile()
                    visualize_langgraph(cot_graph, Path(exp_folder) / f'cot_graph.png')

                    cot_graph.invoke({'finished': False},
                                      {'configurable': {'model': model_version,
                                                        'write_out_file': write_out_file,
                                                        'data': data}})

                text_interface.response_processor(model_version=f"{model_version}{cot_suffix}")

        scorer = Scorer(write_out_files, ask_about)


from jsonargparse import CLI


if __name__ == '__main__':
    CLI(Tester, parser_mode='omegaconf')
