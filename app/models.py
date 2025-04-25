from typing import List, Dict, Optional, Union, Literal, Any
from pydantic import BaseModel, Field

# --- Tool Call related models ---
class FunctionCall(BaseModel):
    name: str
    arguments: str # JSON string format

class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall

class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: Dict[str, Any] # Function declaration as expected by Gemini API

# --- Message model update ---
class Message(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]], None] = None # Allow content to be None for tool calls
    tool_calls: Optional[List[ToolCall]] = None # For assistant messages with tool calls
    tool_call_id: Optional[str] = None # For tool messages responding to a call

# --- Request model update ---
class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = 0.7
    top_p: Optional[float] = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    thinking_budget: Optional[int] = 4096
    tools: Optional[List[Tool]] = None # Add tools field
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None # Add tool_choice field

# --- Response model updates ---
class ChoiceDelta(BaseModel): # For streaming delta
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None

class Choice(BaseModel):
    index: int
    message: Message
    delta: Optional[ChoiceDelta] = None # Add delta for streaming
    finish_reason: Optional[str] = None
    # Note: Gemini API might not return tool_calls directly in the choice message in non-streaming,
    # but rather as a separate part. We'll handle this in the main logic.
    # Let's keep the message structure consistent with OpenAI for the response.

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion", "chat.completion.chunk"] # Allow chunk type for streaming
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = Field(default_factory=Usage) # Usage might be null in streaming chunks

class ErrorResponse(BaseModel):
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None

class ModelList(BaseModel):
    object: str = "list"
    data: List[Dict]