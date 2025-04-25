import os
import asyncio
import httpx
import logging
import json
import requests
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union
from app.models import ChatCompletionRequest, Message, Tool, ToolCall, FunctionCall

logger = logging.getLogger('my_logger')


@dataclass
class GeneratedText:
    text: str
    finish_reason: Optional[str] = None


class ResponseWrapper:
    def __init__(self, data: Dict[Any, Any]):  # 正确的初始化方法名
        self._data = data
        self._text = self._extract_text()
        self._finish_reason = self._extract_finish_reason()
        self._prompt_token_count = self._extract_prompt_token_count()
        self._candidates_token_count = self._extract_candidates_token_count()
        self._total_token_count = self._extract_total_token_count()
        self._thoughts = self._extract_thoughts()
        self._tool_calls = self._extract_tool_calls() # Add tool calls extraction
        self._json_dumps = json.dumps(self._data, indent=4, ensure_ascii=False)

    def _extract_thoughts(self) -> Optional[str]:
        try:
            for part in self._data['candidates'][0]['content']['parts']:
                if 'thought' in part:
                    return part['text']
            return ""
        except (KeyError, IndexError):
            return ""

    def _extract_text(self) -> str:
        try:
            for part in self._data['candidates'][0]['content']['parts']:
                if 'thought' not in part:
                    return part['text']
            return ""
        except (KeyError, IndexError):
            return ""
        except (KeyError, IndexError, TypeError): # Added TypeError
            return ""

    def _extract_tool_calls(self) -> Optional[List[Dict[str, Any]]]:
        """Extracts function calls from the response."""
        try:
            tool_calls = []
            parts = self._data['candidates'][0]['content']['parts']
            for i, part in enumerate(parts):
                if 'functionCall' in part:
                    # Gemini returns functionCall directly in parts
                    fc = part['functionCall']
                    tool_calls.append({
                        "id": f"call_{i}_{fc.get('name', 'unknown_func')}", # Generate an ID
                        "type": "function",
                        "function": {
                            "name": fc.get('name'),
                            "arguments": json.dumps(fc.get('args', {})) # Gemini uses 'args'
                        }
                    })
            return tool_calls if tool_calls else None
        except (KeyError, IndexError, TypeError):
             # logger.warning("Could not extract tool calls from response.", exc_info=True)
             return None

    def _extract_finish_reason(self) -> Optional[str]:
        try:
            # Check for tool calls finish reason first
            if self._data['candidates'][0].get('finishReason') == 'TOOL_CALLS':
                 return 'tool_calls'
            return self._data['candidates'][0].get('finishReason')
        except (KeyError, IndexError):
            return None

    def _extract_prompt_token_count(self) -> Optional[int]:
        try:
            return self._data['usageMetadata'].get('promptTokenCount')
        except (KeyError):
            return None

    def _extract_candidates_token_count(self) -> Optional[int]:
        try:
            return self._data['usageMetadata'].get('candidatesTokenCount')
        except (KeyError):
            return None

    def _extract_total_token_count(self) -> Optional[int]:
        try:
            return self._data['usageMetadata'].get('totalTokenCount')
        except (KeyError):
            return None

    @property
    def text(self) -> str:
        return self._text

    @property
    def finish_reason(self) -> Optional[str]:
        return self._finish_reason

    @property
    def prompt_token_count(self) -> Optional[int]:
        return self._prompt_token_count

    @property
    def candidates_token_count(self) -> Optional[int]:
        return self._candidates_token_count

    @property
    def total_token_count(self) -> Optional[int]:
        return self._total_token_count

    @property
    def thoughts(self) -> Optional[str]:
        return self._thoughts

    @property
    def tool_calls(self) -> Optional[List[Dict[str, Any]]]:
        """Property to access extracted tool calls."""
        return self._tool_calls

    @property
    def json_dumps(self) -> str:
        return self._json_dumps


class GeminiClient:

    AVAILABLE_MODELS = []
    EXTRA_MODELS = os.environ.get("EXTRA_MODELS", "").split(",")

    def __init__(self, api_key: str):
        self.api_key = api_key

<<<<<<< HEAD
<<<<<<< HEAD
    async def stream_chat(self, request: ChatCompletionRequest, contents, safety_settings, system_instruction, tools: Optional[List[Tool]] = None, tool_choice: Optional[Union[str, Dict[str, Any]]] = None, api_version: str = "v1alpha", use_thinking_budget: bool = False):
        logger.info(f"流式开始 (API Version: {api_version})", extra={'request_type': 'stream', 'model': request.model, 'api_version': api_version, 'key': self.api_key[-6:] if self.api_key else 'N/A', 'status_code': 'N/A', 'error_message': ''})
=======
=======
>>>>>>> parent of f44bd50 (更新thinking模型list)
    async def stream_chat(self, request: ChatCompletionRequest, contents, safety_settings, system_instruction, api_version: str = "v1alpha"):
        log_msg = format_log_message('INFO', f"流式开始 (API Version: {api_version})", extra={'request_type': 'stream', 'model': request.model, 'api_version': api_version})
        logger.info(log_msg)
        # api_version = "v1alpha" # Removed, now passed as parameter
>>>>>>> parent of f44bd50 (更新thinking模型list)
        url = f"https://generativelanguage.googleapis.com/{api_version}/models/{request.model}:streamGenerateContent?key={self.api_key}&alt=sse"
        headers = {
            "Content-Type": "application/json",
        }
        data = {
            "contents": contents,
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_tokens,
                "thinkingConfig": {
                    "thinking_budget": request.thinking_budget
                }
            },
            "safetySettings": safety_settings,
        }
        # Validate and adjust thinking_budget within generationConfig
        budget = request.thinking_budget
        if budget is not None:
            if budget < 0:
                budget = 0
            elif 0 < budget <= 1024:
                budget = 1024
            elif budget > 24576:
                budget = 24576
            # Update the nested thinking_budget
            if "generationConfig" in data and "thinkingConfig" in data["generationConfig"]:
                data["generationConfig"]["thinkingConfig"]["thinking_budget"] = budget
        else:
            # If budget is None, remove thinkingConfig from generationConfig if it exists
            if "generationConfig" in data and "thinkingConfig" in data["generationConfig"]:
                 del data["generationConfig"]["thinkingConfig"]

        if system_instruction:
            data["system_instruction"] = system_instruction

        # Add tools and tool_config (mapping from tool_choice)
        if tools:
            # Gemini expects tools in a specific format (list of function declarations)
            gemini_tools = [{"function_declarations": [t.function for t in tools if t.type == "function"]}]
            data["tools"] = gemini_tools

        if tool_choice:
             # Basic mapping for 'auto', 'none', or specific function
             if tool_choice == "auto":
                 data["tool_config"] = {"function_calling_config": {"mode": "AUTO"}}
             elif tool_choice == "none":
                 data["tool_config"] = {"function_calling_config": {"mode": "NONE"}}
             elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
                 func_name = tool_choice.get("function", {}).get("name")
                 if func_name:
                     data["tool_config"] = {
                         "function_calling_config": {
                             "mode": "ANY",
                             "allowed_function_names": [func_name]
                         }
                     }
             # Add more complex tool_choice mappings if needed

        logger.debug(f"Gemini Request Payload: {json.dumps(data, indent=2)}")

        async with httpx.AsyncClient() as client:
            # Use response = await client.post(...) for debugging non-stream errors if needed
            async with client.stream("POST", url, headers=headers, json=data, timeout=300) as response:
                buffer = b""
                try:
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        if line.startswith("data: "):
                            line = line[len("data: "):] 
                        buffer += line.encode('utf-8')
                        try:
                            # logger.debug(f"Raw Gemini Stream Data: {line}") # Debug raw data
                            data = json.loads(buffer.decode('utf-8'))
                            buffer = b"" # Clear buffer after successful parse
                            # logger.debug(f"Parsed Gemini Stream Chunk: {json.dumps(data, indent=2)}") # Debug parsed data

                            if 'candidates' in data and data['candidates']:
                                candidate = data['candidates'][0]
                                finish_reason = candidate.get("finishReason")
                                finish_reason_mapped = None
                                if finish_reason == "STOP":
                                    finish_reason_mapped = "stop"
                                elif finish_reason == "MAX_TOKENS":
                                    finish_reason_mapped = "length"
                                elif finish_reason == "TOOL_CALLS":
                                     finish_reason_mapped = "tool_calls"
                                elif finish_reason: # Other reasons like SAFETY, RECITATION, etc.
                                     finish_reason_mapped = "stop" # Map others to stop for now, or handle specifically
                                     logger.warning(f"Gemini finish reason: {finish_reason}")
                                     # Optionally raise error for safety blocks if needed here

                                if 'content' in candidate and 'parts' in candidate['content']:
                                    parts = candidate['content']['parts']
                                    text_chunk = ""
                                    tool_calls_chunk = []
                                    # Aggregate text and tool calls from parts in the chunk
                                    for i, part in enumerate(parts):
                                        if 'text' in part:
                                            text_chunk += part['text']
                                        elif 'functionCall' in part:
                                            fc = part['functionCall']
                                            # Generate a unique ID for the tool call for this chunk
                                            # Note: A persistent ID across chunks might be needed if a call spans multiple chunks.
                                            # For simplicity, generate per-chunk ID for now.
                                            tool_call_id = f"call_{fc.get('name', 'unknown')}_{int(time.time()*1000)}_{i}"
                                            tool_calls_chunk.append({
                                                "index": len(tool_calls_chunk), # Index within this chunk's tool calls
                                                "id": tool_call_id,
                                                "type": "function",
                                                "function": {
                                                    "name": fc.get('name'),
                                                    # Arguments might stream; handle potential partial JSON if necessary.
                                                    # For simplicity now, assume args are complete in the part.
                                                    "arguments": json.dumps(fc.get('args', {}))
                                                }
                                            })

                                    # Yield text delta chunk if present
                                    if text_chunk:
                                        yield {"type": "delta", "content": {"role": "assistant", "content": text_chunk}, "finish_reason": None}

                                    # Yield tool calls delta chunk if present
                                    if tool_calls_chunk:
                                         yield {"type": "delta", "content": {"role": "assistant", "content": None, "tool_calls": tool_calls_chunk}, "finish_reason": None}


                                # Yield final chunk with finish reason if applicable
                                if finish_reason_mapped:
                                     yield {"type": "final", "content": None, "finish_reason": finish_reason_mapped}
                                     break # Stop processing after final chunk

                                # Handle safety ratings immediately if needed (optional, could also be handled by finish_reason)
                                if 'safetyRatings' in candidate:
                                    for rating in candidate['safetyRatings']:
                                        if rating['probability'] == 'HIGH':
                                            logger.warning(f"模型的响应因高概率被标记为 {rating['category']}")
                                            # Decide how to handle safety blocks in stream (e.g., raise error, yield special message)
                                            # Raising error might be best to trigger retry logic if applicable
                                            raise ValueError(f"模型的响应被截断: {rating['category']}")

                        except json.JSONDecodeError:
                            # logger.debug(f"JSON解析错误, 当前缓冲区内容: {buffer}")
                            # Incomplete JSON, wait for more data
                            continue
                        except Exception as e:
                            logger.error(f"流式处理内部错误: {e}", exc_info=True)
                            raise e # Re-raise to be caught by outer handler
                except Exception as e:
                    logger.error(f"流式处理错误: {e}", exc_info=True)
                    raise e # Ensure outer handler catches this
                finally:
<<<<<<< HEAD
<<<<<<< HEAD
                    logger.info("流式结束", extra={'request_type': 'stream', 'model': request.model, 'key': self.api_key[-6:] if self.api_key else 'N/A', 'status_code': 'N/A', 'error_message': ''})


 
    def complete_chat(self, request: ChatCompletionRequest, contents, safety_settings, system_instruction, tools: Optional[List[Tool]] = None, tool_choice: Optional[Union[str, Dict[str, Any]]] = None, api_version: str = "v1alpha", use_thinking_budget: bool = False):
        logger.info(f"非流式请求开始 (API Version: {api_version})", extra={'request_type': 'non-stream', 'model': request.model, 'api_version': api_version, 'key': self.api_key[-6:] if self.api_key else 'N/A', 'status_code': 'N/A', 'error_message': ''})
=======
                    log_msg = format_log_message('INFO', "流式结束", extra={'request_type': 'stream', 'model': request.model})
        logger.info(log_msg)
        # Validate and adjust thinking_budget within generationConfig
        budget = request.thinking_budget
        if budget is not None:
            if budget < 0:
                budget = 0
            elif 0 < budget <= 1024:
                budget = 1024
            elif budget > 24576:
                budget = 24576
            # Update the nested thinking_budget
            if "generationConfig" in data and "thinkingConfig" in data["generationConfig"]:
                data["generationConfig"]["thinkingConfig"]["thinking_budget"] = budget
        else:
            # If budget is None, remove thinkingConfig from generationConfig if it exists
            if "generationConfig" in data and "thinkingConfig" in data["generationConfig"]:
                 del data["generationConfig"]["thinkingConfig"]


=======
                    log_msg = format_log_message('INFO', "流式结束", extra={'request_type': 'stream', 'model': request.model})
        logger.info(log_msg)
        # Validate and adjust thinking_budget within generationConfig
        budget = request.thinking_budget
        if budget is not None:
            if budget < 0:
                budget = 0
            elif 0 < budget <= 1024:
                budget = 1024
            elif budget > 24576:
                budget = 24576
            # Update the nested thinking_budget
            if "generationConfig" in data and "thinkingConfig" in data["generationConfig"]:
                data["generationConfig"]["thinkingConfig"]["thinking_budget"] = budget
        else:
            # If budget is None, remove thinkingConfig from generationConfig if it exists
            if "generationConfig" in data and "thinkingConfig" in data["generationConfig"]:
                 del data["generationConfig"]["thinkingConfig"]


>>>>>>> parent of f44bd50 (更新thinking模型list)

    def complete_chat(self, request: ChatCompletionRequest, contents, safety_settings, system_instruction, api_version: str = "v1alpha"):
        log_msg = format_log_message('INFO', f"非流式请求开始 (API Version: {api_version})", extra={'request_type': 'non-stream', 'model': request.model, 'api_version': api_version})
        logger.info(log_msg)
        # api_version = "v1alpha" # Removed, now passed as parameter
>>>>>>> parent of f44bd50 (更新thinking模型list)
        url = f"https://generativelanguage.googleapis.com/{api_version}/models/{request.model}:generateContent?key={self.api_key}"
        headers = {
            "Content-Type": "application/json",
        }
        data = {
            "contents": contents,
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_tokens,
                "thinkingConfig": {  # Move thinking config inside generationConfig
                    "thinking_budget": request.thinking_budget
                }
            },
            "safetySettings": safety_settings,
        }
        if system_instruction:
            data["system_instruction"] = system_instruction

        # Add tools and tool_config (mapping from tool_choice)
        if tools:
            gemini_tools = [{"function_declarations": [t.function for t in tools if t.type == "function"]}]
            data["tools"] = gemini_tools

        if tool_choice:
             # Basic mapping for 'auto', 'none', or specific function
             if tool_choice == "auto":
                 data["tool_config"] = {"function_calling_config": {"mode": "AUTO"}}
             elif tool_choice == "none":
                 data["tool_config"] = {"function_calling_config": {"mode": "NONE"}}
             elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
                 func_name = tool_choice.get("function", {}).get("name")
                 if func_name:
                     data["tool_config"] = {
                         "function_calling_config": {
                             "mode": "ANY",
                             "allowed_function_names": [func_name]
                         }
                     }
             # Add more complex tool_choice mappings if needed

        logger.debug(f"Gemini Request Payload (non-stream): {json.dumps(data, indent=2)}")
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return ResponseWrapper(response.json())

    def convert_messages(self, messages: List[Message], use_system_prompt=False):
        gemini_history = []
        errors = []
        system_instruction_text = ""
        is_system_phase = use_system_prompt

        for i, message in enumerate(messages):
            role = message.role
            content = message.content
            tool_calls = message.tool_calls
            tool_call_id = message.tool_call_id

            # Handle tool role first
            if role == 'tool':
                 if tool_call_id and isinstance(content, str):
                     # Find the corresponding function call to get the function name
                     # This requires looking back at the previous assistant message,
                     # which is complex here. Assuming the caller provides enough info
                     # or we can infer it. For now, let's structure the part simply.
                     # Gemini expects role: "function" for tool responses.
                     # It needs 'name' (function name) and 'response' (content).
                     # We need to find the function name associated with tool_call_id.
                     # This conversion logic might need adjustment based on how tool_call_id maps back.
                     # Placeholder: Assuming we can retrieve the function name somehow.
                     # A better approach might be to pass the function name along with the tool message.
                     # For now, let's create a placeholder structure.
                     # We need the function name that was called. Let's assume it's passed implicitly or findable.
                     # This part is tricky without context of the previous assistant message.
                     # Let's assume for now the 'content' contains the result and we need the function name.
                     # We'll add a placeholder name. The calling code in main.py might need adjustment.

                     # Find the function name from the previous assistant message's tool_calls
                     function_name = "unknown_function" # Default
                     if i > 0 and messages[i-1].role == 'assistant' and messages[i-1].tool_calls:
                         for tc in messages[i-1].tool_calls:
                             if tc.id == tool_call_id:
                                 function_name = tc.function.name
                                 break

                     gemini_history.append({
                         "role": "function", # Gemini uses 'function' role for tool results
                         "parts": [{
                             "functionResponse": {
                                 "name": function_name,
                                 "response": {"content": content} # Gemini expects response content here
                             }
                         }]
                     })
                 else:
                     errors.append(f"Tool message missing tool_call_id or content: {message}")
                 continue # Move to next message

            # Handle other roles (user, assistant, system)
            elif isinstance(content, str) or content is None: # Allow content to be None for assistant tool calls
                if is_system_phase and role == 'system' and content:
                    if system_instruction_text:
                        system_instruction_text += "\n" + content
                    else:
                        system_instruction_text = content
                else:
                    is_system_phase = False

                    if role in ['user', 'system'] and content: # Ensure content exists for user/system text
                        role_to_use = 'user'
                    elif role == 'assistant':
                        role_to_use = 'model'
                    else:
                        errors.append(f"Invalid role: {role}")
                        continue

                    current_parts = []
                    if content: # Add text part if content exists
                         current_parts.append({"text": content})

                    # Add function call parts if they exist (for assistant messages)
                    if role == 'assistant' and tool_calls:
                        for tc in tool_calls:
                            current_parts.append({
                                "functionCall": {
                                    "name": tc.function.name,
                                    "args": json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                                }
                            })

                    if not current_parts: # Skip if message has no content and no tool calls
                         continue

                    # Append or create new history entry
                    if gemini_history and gemini_history[-1]['role'] == role_to_use:
                         # Check if the last part of the previous message and the first part of the current one are both text
                         can_merge = False
                         if content and gemini_history[-1]['parts'] and "text" in gemini_history[-1]['parts'][-1] and "text" in current_parts[0]:
                             can_merge = True

                         if can_merge:
                             gemini_history[-1]['parts'][-1]["text"] += "\n" + content # Merge text content
                             if len(current_parts) > 1: # Add remaining parts (e.g., tool calls)
                                 gemini_history[-1]['parts'].extend(current_parts[1:])
                         else:
                             gemini_history[-1]['parts'].extend(current_parts) # Append all parts
                    elif current_parts: # Only append if there are parts
                        gemini_history.append(
                            {"role": role_to_use, "parts": current_parts})

            elif isinstance(content, list): # Handling list content (e.g., multimodal)
                parts = []
                for item in content:
                    if item.get('type') == 'text':
                        parts.append({"text": item.get('text')})
                    elif item.get('type') == 'image_url':
                        image_data = item.get('image_url', {}).get('url', '')
                        if image_data.startswith('data:image/'):
                            try:
                                mime_type, base64_data = image_data.split(';')[0].split(':')[1], image_data.split(',')[1]
                                parts.append({
                                    "inline_data": {
                                        "mime_type": mime_type,
                                        "data": base64_data
                                    }
                                })
                            except (IndexError, ValueError):
                                errors.append(
                                    f"Invalid data URI for image: {image_data}")
                        else:
                            errors.append(
                                f"Invalid image URL format for item: {item}")

                if parts:
                    # Multimodal messages typically don't have tool calls, but check just in case
                    if role in ['user', 'system']:
                        role_to_use = 'user'
                    elif role == 'assistant':
                        # Assistant multimodal messages are less common, handle if needed
                        role_to_use = 'model'
                    else:
                        errors.append(f"Invalid role: {role}")
                        continue
                    if gemini_history and gemini_history[-1]['role'] == role_to_use:
                        gemini_history[-1]['parts'].extend(parts)
                    else:
                        gemini_history.append(
                            {"role": role_to_use, "parts": parts})
            else:
                 # Handle cases where content is neither str, list, nor None (or role is tool but invalid)
                 if role != 'tool': # Avoid erroring on already handled tool role
                     errors.append(f"Unsupported message content type or structure: {message}")


        if errors:
            # Log errors or handle them appropriately
            logger.error(f"Errors converting messages: {errors}")
            # Decide whether to raise an exception or return partial conversion
            # Returning errors might be problematic for the caller expecting (history, system_instruction)
            # Let's raise an exception for now
            raise ValueError(f"Failed to convert messages: {'; '.join(errors)}")
        else:
            system_instruction_payload = {"parts": [{"text": system_instruction_text}]} if system_instruction_text else None
            # Filter out empty history entries just in case
            gemini_history = [entry for entry in gemini_history if entry.get("parts")]
            return gemini_history, system_instruction_payload

    @staticmethod
    async def list_available_models(api_key) -> list:
        url = "https://generativelanguage.googleapis.com/v1alpha/models?key={}".format(
            api_key)
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            models = [model["name"].replace("models/", "") for model in data.get("models", [])]
            models.extend(GeminiClient.EXTRA_MODELS)
            return models
