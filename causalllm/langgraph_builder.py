from typing import Dict, TypedDict, Optional, Annotated
import operator

from langchain.chat_models import init_chat_model
from langgraph.graph import END, StateGraph, START
from langchain_core.runnables.config import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
from causalllm.prompt_utils import partial_replace
from pydantic import Field, create_model

models = {'gpt-4o-mini': init_chat_model("gpt-4o-mini", model_provider="openai"),
          'gpt-4o': init_chat_model("gpt-4o", model_provider="openai"),
          'o1': init_chat_model("o1", model_provider="openai"),
          'o1-mini': init_chat_model("o1-mini", model_provider="openai"),
          'o3-mini': init_chat_model("o3-mini", model_provider="openai"),
          'gpt-4': init_chat_model("gpt-4", model_provider="openai"),
          'gpt-3.5-turbo': init_chat_model("gpt-3.5-turbo", model_provider="openai")}
extractor_template = ChatPromptTemplate.from_messages([
                    ("system", Path('agents/general/extractor_system.txt').read_text()),
                    ("human", "{text}")
                ])

class ConfigSchema(TypedDict):
    model: Optional[str]
    system_message: Optional[str]

class AgentState(TypedDict):
    variables: Annotated[Dict[str, any], operator.or_]
    # multi_agent_data: Dict[str, any]
    agent_response: Annotated[Dict[str, any], operator.or_]

def executor(states: AgentState, config: RunnableConfig):
    model = models[config["configurable"].get("model", 'gpt-4o-mini')]
    subtask = config["metadata"]["langgraph_node"]
    multi_agent_data = config["configurable"]['multi_agent_data']

    if subtask == 'X' and multi_agent_data.get('X') is None:
        return {'agent_response': {'X': states['variables']['X']}}
    system_prompt = multi_agent_data[subtask].system_prompt
    user_prompt = partial_replace(multi_agent_data[subtask].user_prompt, states['variables'])
    st_prompt = [('system', system_prompt), ('human', user_prompt)]

    response = model.invoke(st_prompt)


    subtask_response = [system_prompt, user_prompt, response.content]
    if subtask == 'Y':
        schema = create_model(f'{subtask}subtask', **{
            'Y': (str, Field(None, description="Final answer for the query: yes/no", enum=['yes', 'no']))})
    else:
        schema = create_model(f'{subtask}subtask',
                              **{output_variable: (str, Field(None, description="One of the final concluded answer(s)"))
                                 for output_variable in multi_agent_data[subtask].output_variables})
    st_extractor = extractor_template | model.with_structured_output(schema=schema)
    extracted_data = st_extractor.invoke({'text': response.content})
    return {'variables': extracted_data.__dict__, 'agent_response': {subtask: subtask_response}}

if __name__ == '__main__':
    builder = StateGraph(AgentState, ConfigSchema)
    builder.add_node("A", executor)
    builder.add_edge(START, "A")
    builder.add_edge("A", END)

    graph = builder.compile()

    st_prompt = [
        ('system', 'You are a helpful assistant.'),
        ('human', 'Please output the result of 1 + 1.')
    ]

    graph.invoke({'keys': {'X': 'sample_X'}, 'st_template': {'A': st_prompt}}, {'configurable': {"model": "gpt-4o-mini"}})