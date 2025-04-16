from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class SubTask(BaseModel):
    subtask_id: Optional[str] = Field(None, description="The unique variable name for the subtask: X, Y, graph, data, results, etc. In most cases, it is the same as the output variable name.")
    system_prompt: Optional[str] = Field(None, description="The system prompt of the subtask for the agent.")
    user_prompt: Optional[str] = Field(None, description="The user prompt for the agent to respond to.")
    input_variables: Optional[List[str]] = Field(None, description="The names of input variables for the subtask. Those variables are wrapped in {variable_name}")
    output_variables: Optional[List[str]] = Field(None, description="The names of output variables for the subtask. Those variables are wrapped in {variable_name}")

class MultiAgentData(BaseModel):
    subtasks: List[SubTask]

class SCGString(BaseModel):
    scg_string: str = Field(None, description="The solution causal graph.")
    # short_scg_string: str = Field(None, description="The solution causal graph where for nodes, we omit the long descriptions and only keep the subtask's short descriptions.")


class Variable2Name(BaseModel):
    variable: Optional[str] = Field(None, description="The variable symbol")
    name: Optional[str] = Field(None, description="The original name")
    set_description: Optional[str] = Field(None, description="The description of the variable=1")
    unset_description: Optional[str] = Field(None, description="The description of the variable=0")

class Anonymizer(BaseModel):
    variable2name: List[Variable2Name]
    background_question: str = Field(None, description="The anonymized background and question descriptions.")
    reasoning: str = Field(None, description="The anonymized reasoning steps.")
    # plain_reasoning: str = Field(None, description="The reasoning steps with real event/variable names.")

class Verifier(BaseModel):
    valid: bool = Field(None, description="Whether the solution satisfies all the requirements.")
    feedback: Optional[str] = Field(None, description="The feedback message for the user if the solution does not satisfy any of the requirements.")

class InputFeedback(BaseModel):
    name: str = Field(None, enum=['SCG design', 'Prompt generation'], description="The name of the input component.")
    fb: str = Field(None, description="The feedback message about how and why the input component can cause the error and how to fix it. Include at least one concrete examples to reproduce and explain the critical errors.")

class Reflector(BaseModel):
    # input_component: str = Field(None, enum=['SCG design', 'Prompt generation'], description="The name of the input component that takes the major responsibility for the error.")
    # feedback: Optional[str] = Field(None, description="The feedback message about how and why the input component is responsible for the error and how to fix it. Include some concrete examples to show the critical errors.")
    major_responsible_input_component: str = Field(None, enum=['SCG design', 'Prompt generation'], description="The name of the input component that takes the major responsibility for the error.")
    input_feedbacks: List[InputFeedback]

class BinaryClassifier(BaseModel):
    answer: str = Field(None, enum=['yes', 'no', 'true', 'false'], description="The final answer for the binary classification task.")