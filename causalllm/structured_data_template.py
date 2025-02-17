from pydantic import BaseModel, Field
from typing import List, Optional

class SubTask(BaseModel):
    subtask_id: Optional[str] = Field(None, description="The unique variable for the subtask: A, B, C, D, Y, etc.")
    system_prompt: Optional[str] = Field(None, description="The system prompt of the subtask for the agent.")
    user_prompt: Optional[str] = Field(None, description="The user prompt for the agent to respond to.")
    input_variables: Optional[List[str]] = Field(None, description="The names of input variables for the subtask. Those variables are wrapped in {variable_name}")
    output_variables: Optional[List[str]] = Field(None, description="The names of output variables for the subtask. Those variables are wrapped in {variable_name}")

class MultiAgentData(BaseModel):
    subtasks: List[SubTask]

class SCGString(BaseModel):
    scg_string: str = Field(None, description="The solution causal graph.")
    # short_scg_string: str = Field(None, description="The solution causal graph where for nodes, we omit the long descriptions and only keep the subtask's short descriptions.")
