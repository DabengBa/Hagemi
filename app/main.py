from fastapi import FastAPI, HTTPException, Request, Depends, status, Response
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from .models import ChatCompletionRequest, ChatCompletionResponse, ErrorResponse, ModelList, Message, ToolCall, ChoiceDelta
from .gemini import GeminiClient, ResponseWrapper
from .utils import handle_gemini_error, protect_from_abuse, APIKeyManager, test_api_key
import os
import json
import asyncio
from typing import Literal
import random
import requests
from datetime import datetime, timedelta, timezone
from apscheduler.schedulers.background import BackgroundScheduler
import sys
import logging

logging.getLogger("uvicorn").disabled = True
logging.getLogger("uvicorn.access").disabled = True

# 配置 logger
logger = logging.getLogger("my_logger")
logger.setLevel(logging.DEBUG)

def translate_error(message: str) -> str:
    if "quota exceeded" in message.lower():
        return "API 密钥配额已用尽"
    if "invalid argument" in message.lower():
        return "无效参数"
    if "internal server error" in message.lower():
        return "服务器内部错误"
    if "service unavailable" in message.lower():
        return "服务不可用"
    return message


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.excepthook(exc_type, exc_value, exc_traceback)
        return
    error_message = translate_error(str(exc_value))
    logger.error(f"未捕获的异常: %s" % error_message, extra={'status_code': 500, 'error_message': error_message, 'key': 'N/A', 'request_type': 'N/A', 'model': 'N/A'})


sys.excepthook = handle_exception

app = FastAPI()

PASSWORD = os.environ.get("PASSWORD", "123")
SECOND_MODEL = os.environ.get("SECOND_MODEL", "gemini-2.5-flash-preview-04-17")
MAX_RETRY = int(os.environ.get("MAX_RETRY", "3")) 
MAX_REQUESTS_PER_MINUTE = int(os.environ.get("MAX_REQUESTS_PER_MINUTE", "4"))
MAX_REQUESTS_PER_DAY_PER_IP = int(
    os.environ.get("MAX_REQUESTS_PER_DAY_PER_IP", "200"))
RETRY_DELAY = 3
safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": 'HARM_CATEGORY_CIVIC_INTEGRITY',
        "threshold": 'BLOCK_NONE'
    }
]
safety_settings_g2 = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "OFF"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "OFF"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "OFF"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "OFF"
    },
    {
        "category": 'HARM_CATEGORY_CIVIC_INTEGRITY',
        "threshold": 'OFF'
    }
]
GOOGLE_SEARCH_MODELS = {
    "gemini-2.0-flash-exp",
    "gemini-2.0-flash",
    "gemini-2.5-pro-preview-03-25",
    "gemini-2.5-pro-exp-03-25",
}

THINKING_BUDGET_MODELS = {
    "gemini-2.5-flash-preview-04-17",
}

<<<<<<< HEAD
<<<<<<< HEAD
logger.info("即将实例化 APIKeyManager", extra={'key': 'N/A', 'request_type': 'N/A', 'model': 'N/A', 'status_code': 'N/A', 'error_message': ''})
key_manager = APIKeyManager() # 实例化 APIKeyManager，栈会在 __init__ 中初始化
logger.info("APIKeyManager 实例化完成", extra={'key': 'N/A', 'request_type': 'N/A', 'model': 'N/A', 'status_code': 'N/A', 'error_message': ''})
current_api_key = key_manager.get_available_key()
logger.info("获取可用密钥完成", extra={'key': 'N/A', 'request_type': 'N/A', 'model': 'N/A', 'status_code': 'N/A', 'error_message': ''})
=======
key_manager = APIKeyManager() # 实例化 APIKeyManager，栈会在 __init__ 中初始化
current_api_key = key_manager.get_available_key()
>>>>>>> parent of f44bd50 (更新thinking模型list)
=======
key_manager = APIKeyManager() # 实例化 APIKeyManager，栈会在 __init__ 中初始化
current_api_key = key_manager.get_available_key()
>>>>>>> parent of f44bd50 (更新thinking模型list)


def switch_api_key():
    global current_api_key
    key = key_manager.get_available_key() # get_available_key 会处理栈的逻辑
    if key:
        current_api_key = key
        logger.info(f"API key 替换为 → {current_api_key[-6:]}...", extra={'key': current_api_key[-6:], 'request_type': 'switch_key', 'model': 'N/A', 'status_code': 'N/A', 'error_message': ''})
    else:
        logger.error("API key 替换失败，所有API key都已尝试，请重新配置或稍后重试", extra={'key': 'N/A', 'request_type': 'switch_key', 'model': 'N/A', 'status_code': 'N/A', 'error_message': ''})


async def check_keys():
    available_keys = []
    for key in key_manager.api_keys:
        is_valid = await test_api_key(key)
        status_msg = "有效" if is_valid else "无效"
        logger.info(f"API Key {key[:10]}... {status_msg}.", extra={'key': key[-6:], 'request_type': 'startup', 'model': 'N/A', 'status_code': 'N/A', 'error_message': ''})
        if is_valid:
            available_keys.append(key)
    if not available_keys:
        logger.error("没有可用的 API 密钥！", extra={'key': 'N/A', 'request_type': 'startup', 'model': 'N/A', 'status_code': 'N/A', 'error_message': ''})
    return available_keys


@app.on_event("startup")
async def startup_event():
    logger.info("Starting Gemini API proxy...", extra={'key': 'N/A', 'request_type': 'startup', 'model': 'N/A', 'status_code': 'N/A', 'error_message': ''})
    available_keys = await check_keys()
    if available_keys:
        key_manager.api_keys = available_keys
        key_manager._reset_key_stack() # 启动时也确保创建随机栈
        key_manager.show_all_keys()
        logger.info(f"可用 API 密钥数量：{len(key_manager.api_keys)}", extra={'key': 'N/A', 'request_type': 'startup', 'model': 'N/A', 'status_code': 'N/A', 'error_message': ''})
        # MAX_RETRIES = len(key_manager.api_keys)
        retry_attempts = min(MAX_RETRY, len(key_manager.api_keys)) if key_manager.api_keys else 1
        logger.info(f"最大重试次数设置为：{retry_attempts}", extra={'key': 'N/A', 'request_type': 'startup', 'model': 'N/A', 'status_code': 'N/A', 'error_message': ''}) # 添加日志
        if key_manager.api_keys:
            all_models = await GeminiClient.list_available_models(key_manager.api_keys[0])
            GeminiClient.AVAILABLE_MODELS = all_models
            logger.info(f"Available models loaded: {len(all_models)} models", extra={'key': 'N/A', 'request_type': 'startup', 'model': 'N/A', 'status_code': 'N/A', 'error_message': ''})

@app.get("/v1/models", response_model=ModelList)
def list_models():
    logger.info("Received request to list models", extra={'request_type': 'list_models', 'status_code': 200, 'key': 'N/A', 'model': 'N/A', 'error_message': ''})
    return ModelList(data=[{"id": model, "object": "model", "created": 1678888888, "owned_by": "organization-owner"} for model in GeminiClient.AVAILABLE_MODELS])


async def verify_password(request: Request):
    if PASSWORD:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=401, detail="Unauthorized: Missing or invalid token")
        token = auth_header.split(" ")[1]
        if token != PASSWORD:
            raise HTTPException(
                status_code=401, detail="Unauthorized: Invalid token")


async def process_request(chat_request: ChatCompletionRequest, http_request: Request, request_type: Literal['stream', 'non-stream']):
    global current_api_key
    protect_from_abuse(
        http_request, MAX_REQUESTS_PER_MINUTE, MAX_REQUESTS_PER_DAY_PER_IP)

    key_manager.reset_tried_keys_for_request() # 在每次请求处理开始时重置 tried_keys 集合

    try:
        contents, system_instruction = GeminiClient.convert_messages(
            GeminiClient, chat_request.messages)
    except ValueError as e:
         logger.error(f"消息转换失败: {e}", extra={'key': 'N/A', 'request_type': request_type, 'model': chat_request.model, 'status_code': 400, 'error_message': str(e)})
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to convert messages: {e}")

    retry_attempts = min(MAX_RETRY, len(key_manager.api_keys)) if key_manager.api_keys else 1 # 重试次数等于密钥数量和MAX_RETRY之中最小值，至少尝试 1 次
<<<<<<< HEAD
<<<<<<< HEAD
    api_version_to_use = "v1beta" # Start with v1beta as it's generally more stable for tools
    use_thinking_budget = chat_request.model in THINKING_BUDGET_MODELS
    tools = chat_request.tools
    tool_choice = chat_request.tool_choice
=======
    api_version_to_use = "v1alpha" # Start with v1alpha
>>>>>>> parent of f44bd50 (更新thinking模型list)
=======
    api_version_to_use = "v1alpha" # Start with v1alpha
>>>>>>> parent of f44bd50 (更新thinking模型list)

    for attempt in range(1, retry_attempts + 1):
        # Get a key only if it's the first attempt or after a key switch (not after version switch)
        if attempt == 1 or 'switch_key_occurred' in locals() and switch_key_occurred:
             current_api_key = key_manager.get_available_key()
             api_version_to_use = "v1alpha" # Reset to v1alpha when getting a new key
             switch_key_occurred = False # Reset flag

        if current_api_key is None: # 检查是否获取到 API 密钥
            logger.warning("没有可用的 API 密钥，跳过本次尝试", extra={'request_type': request_type, 'model': chat_request.model, 'status_code': 'N/A', 'key': 'N/A', 'error_message': ''})
            break  # 如果没有可用密钥，跳出循环

        extra_log = {'key': current_api_key[-6:], 'request_type': request_type, 'model': chat_request.model, 'status_code': 'N/A', 'error_message': '', 'api_version': api_version_to_use}
        logger.info(f"第 {attempt}/{retry_attempts} 次尝试 (API Version: {api_version_to_use})", extra=extra_log)

        gemini_client = GeminiClient(current_api_key)
        try:
            if chat_request.stream:
                async def stream_generator():
                    chunk_id_counter = 0
                    try:
<<<<<<< HEAD
<<<<<<< HEAD
                        async for chunk_data in gemini_client.stream_chat(
                            chat_request, contents,
                            safety_settings_g2 if 'gemini-2.0-flash-exp' in chat_request.model else safety_settings,
                            system_instruction,
                            tools=tools, # Pass tools
                            tool_choice=tool_choice, # Pass tool_choice
                            api_version=api_version_to_use,
                            use_thinking_budget=use_thinking_budget
                        ):
                            chunk_id_counter += 1
                            chunk_id = f"chatcmpl-chunk-{chunk_id_counter}"
                            created_time = int(time.time())

                            if chunk_data["type"] == "delta":
                                delta_content = chunk_data["content"]
                                # Map Gemini delta to OpenAI delta format
                                choice_delta = ChoiceDelta(**delta_content) # Directly use if structure matches
                                choice = Choice(index=0, delta=choice_delta, finish_reason=None, message=None) # message is not needed for delta

                            elif chunk_data["type"] == "final":
                                finish_reason = chunk_data["finish_reason"]
                                choice = Choice(index=0, delta=ChoiceDelta(), finish_reason=finish_reason, message=None) # Empty delta for final chunk

                            else:
                                logger.warning(f"Unknown chunk type received: {chunk_data.get('type')}")
                                continue

                            # Create OpenAI compatible chunk response
                            formatted_chunk = ChatCompletionResponse(
                                id=chunk_id,
                                object="chat.completion.chunk",
                                created=created_time,
                                model=chat_request.model,
                                choices=[choice],
                                usage=None # Usage is typically not in chunks
                            )
                            yield f"data: {formatted_chunk.model_dump_json(exclude_unset=True)}\n\n"

                        # Send DONE message after the loop finishes successfully
=======
=======
>>>>>>> parent of f44bd50 (更新thinking模型list)
                        async for chunk in gemini_client.stream_chat(chat_request, contents, safety_settings_g2 if 'gemini-2.0-flash-exp' in chat_request.model else safety_settings, system_instruction, api_version=api_version_to_use):
                            formatted_chunk = {"id": "chatcmpl-someid", "object": "chat.completion.chunk", "created": 1234567,
                                               "model": chat_request.model, "choices": [{"delta": {"role": "assistant", "content": chunk}, "index": 0, "finish_reason": None}]}
                            yield f"data: {json.dumps(formatted_chunk)}\n\n"
>>>>>>> parent of f44bd50 (更新thinking模型list)
                        yield "data: [DONE]\n\n"

                    except asyncio.CancelledError:
                        extra_log_cancel = {'key': current_api_key[-6:], 'request_type': request_type, 'model': chat_request.model, 'error_message': '客户端已断开连接'}
                        logger.info("客户端连接已中断", extra=extra_log_cancel)
                    except Exception as e:
                        # Handle error within the stream generator needs careful thought,
                        # For now, let the outer exception handler catch it to trigger retry/version switch
                        raise e
                        # error_detail = handle_gemini_error(
                        #     e, current_api_key, key_manager)
                        # yield f"data: {json.dumps({'error': {'message': error_detail, 'type': 'gemini_error'}})}\n\n"
                return StreamingResponse(stream_generator(), media_type="text/event-stream")
            else:
                async def run_gemini_completion():
                    try:
<<<<<<< HEAD
<<<<<<< HEAD
                        # Pass tools and tool_choice to complete_chat
                        response_wrapper: ResponseWrapper = await asyncio.to_thread(
                            gemini_client.complete_chat,
                            chat_request, contents,
                            safety_settings_g2 if 'gemini-2.0-flash-exp' in chat_request.model else safety_settings,
                            system_instruction,
                            tools=tools, # Pass tools
                            tool_choice=tool_choice, # Pass tool_choice
                            api_version=api_version_to_use,
                            use_thinking_budget=use_thinking_budget
                        )
                        return response_wrapper
=======
=======
>>>>>>> parent of f44bd50 (更新thinking模型list)
                        response_content = await asyncio.to_thread(gemini_client.complete_chat, chat_request, contents, safety_settings_g2 if 'gemini-2.0-flash-exp' in chat_request.model else safety_settings, system_instruction, api_version=api_version_to_use)
                        return response_content
>>>>>>> parent of f44bd50 (更新thinking模型list)
                    except asyncio.CancelledError:
                        extra_log_gemini_cancel = {'key': current_api_key[-6:], 'request_type': request_type, 'model': chat_request.model, 'error_message': '客户端断开导致API调用取消', 'api_version': api_version_to_use}
                        logger.info("API调用因客户端断开而取消", extra=extra_log_gemini_cancel)
                        raise

                gemini_task = asyncio.create_task(run_gemini_completion())

                try:
                    done, pending = await asyncio.wait(
                        [gemini_task],
                        return_when=asyncio.FIRST_COMPLETED
                    )

                    if gemini_task in done:
                        response_wrapper = gemini_task.result() # Now it's a ResponseWrapper

                        # Check for empty response or only thoughts
                        if not response_wrapper.text and not response_wrapper.tool_calls:
                            extra_log_empty_response = {'key': current_api_key[-6:], 'request_type': request_type, 'model': chat_request.model, 'status_code': 204, 'finish_reason': response_wrapper.finish_reason}
                            logger.info(f"Gemini API 返回空响应或仅包含思考过程 (Finish Reason: {response_wrapper.finish_reason})", extra=extra_log_empty_response)

                            # Handle empty response retry logic (existing logic seems okay)
                            if response_wrapper.finish_reason != 'STOP' and response_wrapper.finish_reason != 'tool_calls': # Don't retry if it stopped normally or for tool calls
                                if chat_request.model != SECOND_MODEL:
                                    logger.info(f"尝试切换到备用模型 {SECOND_MODEL}", extra={'key': current_api_key[-6:], 'request_type': request_type, 'model': SECOND_MODEL, 'status_code': 'N/A', 'error_message': ''})
                                    chat_request.model = SECOND_MODEL
                                    continue # Retry with the second model

                            # If still empty or stopped normally/tool_calls, return an empty message or handle as needed
                            # For now, let's return an empty assistant message if no text/tools
                            assistant_message = Message(role="assistant", content="")
                            finish_reason = response_wrapper.finish_reason if response_wrapper.finish_reason else "stop"
                            response = ChatCompletionResponse(
                                id="chatcmpl-empty", object="chat.completion", created=int(time.time()), model=chat_request.model,
                                choices=[Choice(index=0, message=assistant_message, finish_reason=finish_reason)]
                                # Usage might be available even for empty responses
                                # usage=Usage(prompt_tokens=response_wrapper.prompt_token_count or 0, completion_tokens=response_wrapper.candidates_token_count or 0, total_tokens=response_wrapper.total_token_count or 0)
                            )
                            # Log success even for empty response if finish reason is STOP/tool_calls
                            extra_log_success = {'key': current_api_key[-6:], 'request_type': request_type, 'model': chat_request.model, 'status_code': 200, 'error_message': '', 'finish_reason': finish_reason}
                            logger.info("请求处理成功 (空响应)", extra=extra_log_success)
                            # Fall through to return the empty response

                        else:
                            # Create assistant message with text and/or tool calls
                            assistant_content = response_wrapper.text
                            tool_calls_list = None
                            if response_wrapper.tool_calls:
                                tool_calls_list = [ToolCall(**tc) for tc in response_wrapper.tool_calls] # Validate with Pydantic

                            assistant_message = Message(
                                role="assistant",
                                content=assistant_content if assistant_content else None, # Content is None if only tool calls
                                tool_calls=tool_calls_list
                            )

                            # Determine finish reason
                            finish_reason = response_wrapper.finish_reason
                            if finish_reason == "TOOL_CALLS":
                                finish_reason = "tool_calls" # Map to OpenAI's reason
                            elif not finish_reason or finish_reason == "STOP":
                                finish_reason = "stop"
                            elif finish_reason == "MAX_TOKENS":
                                finish_reason = "length"
                            # Add other mappings if necessary (e.g., SAFETY -> stop)

                            response = ChatCompletionResponse(
                                id="chatcmpl-someid", object="chat.completion", created=int(time.time()), model=chat_request.model,
                                choices=[Choice(index=0, message=assistant_message, finish_reason=finish_reason)],
                                usage=Usage(prompt_tokens=response_wrapper.prompt_token_count or 0, completion_tokens=response_wrapper.candidates_token_count or 0, total_tokens=response_wrapper.total_token_count or 0)
                            )
                            extra_log_success = {'key': current_api_key[-6:], 'request_type': request_type, 'model': chat_request.model, 'status_code': 200, 'error_message': '', 'finish_reason': finish_reason}
                            logger.info("请求处理成功", extra=extra_log_success)

                        # --- Existing disconnection check and return logic ---
                        try:
                            is_disconnected = await http_request.is_disconnected()
                        except Exception as e_disconnect:
                            # 记录 is_disconnected 调用本身的错误
                            extra_log_disconnect_error = {'key': current_api_key[-6:] if current_api_key else 'N/A', 'request_type': request_type, 'model': chat_request.model, 'error_message': f"Error calling http_request.is_disconnected: {e_disconnect}"}
                            logger.error("调用 http_request.is_disconnected 时出错", extra=extra_log_disconnect_error)
                            # 假设未断开，让后续逻辑继续尝试返回数据
                            is_disconnected = False

                        if is_disconnected:
                            extra_log_client_disconnect = {'key': current_api_key[-6:] if current_api_key else 'N/A', 'request_type': request_type, 'model': chat_request.model, 'error_message': '检测到客户端断开连接'}
                            logger.info("客户端连接已中断 (但仍尝试返回)", extra=extra_log_client_disconnect)
                            if response_content.text:
                                extra_log_response = {
                                    'key': current_api_key[-6:] if current_api_key else 'N/A',
                                    'request_type': request_type,
                                    'model': chat_request.model,
                                    'response_content': response_content.text
                                }
                                logger.info(f"获取到响应内容 (客户端已断开): {extra_log_response['response_content']}", extra=extra_log_response)
                            # 注意：即使客户端断开，我们仍然尝试返回响应
                            # 如果返回失败，FastAPI/Uvicorn 会处理底层的连接错误
                        try:
                            return response # 无论 is_disconnected 检查是否出错或结果如何，都尝试返回
                        except Exception as e_return:
                            # 记录在返回响应时发生的错误
                            extra_log_return_error = {'key': current_api_key[-6:] if current_api_key else 'N/A', 'request_type': request_type, 'model': chat_request.model, 'error_message': f"Error during return response: {e_return}"}
                            logger.error("返回响应时发生错误", extra=extra_log_return_error)
                            # 重新抛出异常，让 FastAPI 或 ASGI 服务器处理，避免触发外层重试
                            raise e_return

                except asyncio.CancelledError:
                    extra_log_request_cancel = {'key': current_api_key[-6:] if current_api_key else 'N/A', 'request_type': request_type, 'model': chat_request.model, 'error_message':"请求被取消" }
                    logger.info("请求取消", extra=extra_log_request_cancel)
                    raise

        except HTTPException as e:
            if e.status_code == status.HTTP_408_REQUEST_TIMEOUT:
                extra_log = {'key': current_api_key[-6:], 'request_type': request_type, 'model': chat_request.model,
                            'status_code': 408, 'error_message': '客户端连接中断'}
                logger.error("客户端连接中断，终止后续重试", extra=extra_log)
                raise  
            else:
                raise  
        except Exception as e:
            error_type = handle_gemini_error(e, current_api_key, key_manager)
            
            # Check if it's a 500 error and we are using v1alpha
            # Adjust error handling for API version switching if needed
            # Since we start with v1beta now, maybe remove the v1alpha -> v1beta switch logic
            # Or adjust based on specific errors encountered with v1beta
            if error_type == "GEMINI_500_ERROR": # Handle 500 errors (potentially try switching key)
                logger.warning(f"遇到 500 错误 (API Version: {api_version_to_use})", extra={'key': current_api_key[-6:], 'request_type': request_type, 'model': chat_request.model, 'status_code': 500, 'error_message': 'Gemini 500 Error'})
                # Fall through to standard retry logic (switch key)
            elif error_type == "无效的 API 密钥": # Handle invalid key specifically
                 logger.error(f"API 密钥无效: ...{current_api_key[-6:]}", extra={'key': current_api_key[-6:], 'request_type': request_type, 'model': chat_request.model, 'status_code': 400, 'error_message': 'Invalid API Key'})
                 # Key is likely bad, proceed to switch key immediately in retry logic
            elif error_type == "API 密钥配额已用尽或其他原因": # Handle 429 specifically
                 logger.warning(f"API 密钥限流: ...{current_api_key[-6:]}", extra={'key': current_api_key[-6:], 'request_type': request_type, 'model': chat_request.model, 'status_code': 429, 'error_message': 'Rate Limit/Quota Exceeded'})
                 # Key is rate-limited, proceed to switch key immediately in retry logic

            # Standard retry logic: switch key if attempts remain
            if attempt < retry_attempts:
                logger.info(f"尝试切换 API Key (当前: ...{current_api_key[-6:]})", extra={'key': current_api_key[-6:], 'request_type': request_type, 'model': chat_request.model, 'status_code': 'N/A', 'error_message': f'Attempt {attempt} failed, switching key'})
                switch_api_key()
                switch_key_occurred = True # Flag that key was switched
                api_version_to_use = "v1beta" # Reset to v1beta when switching key
                await asyncio.sleep(RETRY_DELAY) # 添加重试延迟
                continue # Continue to the next attempt in the loop
            else:
                # This is the final attempt, and it failed. Log it.
                logger.error(f"最后一次尝试失败 (Attempt {attempt}/{retry_attempts})，错误: {error_type}", extra={'key': current_api_key[-6:] if current_api_key else 'N/A', 'request_type': request_type, 'model': chat_request.model, 'status_code': 'N/A', 'error_message': f'Final attempt failed: {error_type}'})
                # No 'continue' here. The loop will terminate after this block,
                # and the code will proceed to raise the final HTTPException outside the loop.
                pass # Explicitly pass to make the else block valid

    msg = "所有API密钥均失败,请稍后重试"
    extra_log_all_fail = {'key': "ALL", 'request_type': request_type, 'model': chat_request.model, 'status_code': 500, 'error_message': msg}
    logger.error(msg, extra=extra_log_all_fail)
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=msg)


# Allow response_model to be Union[ChatCompletionResponse, StreamingResponse] ? No, FastAPI handles StreamingResponse separately.
# The return type hint should reflect what's actually returned.
@app.post("/v1/chat/completions") # Remove response_model here as it varies
async def chat_completions(request: ChatCompletionRequest, http_request: Request, _: None = Depends(verify_password)) -> Response: # Return type is Response
    response = await process_request(request, http_request, "stream" if request.stream else "non-stream")
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_message = translate_error(str(exc))
    extra_log_unhandled_exception = {'status_code': 500, 'error_message': error_message}
    logger.error(f"Unhandled exception: {error_message}", extra=extra_log_unhandled_exception)
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=ErrorResponse(message=str(exc), type="internal_error").dict())


@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Gemini API 代理服务</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                line-height: 1.6;
            }}
            h1 {{
                color: #333;
                text-align: center;
                margin-bottom: 30px;
            }}
            .info-box {{
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 20px;
                margin-bottom: 20px;
            }}
            .status {{
                color: #28a745;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <h1>🤖 Gemini API 代理服务</h1>
        
        <div class="info-box">
            <h2>🟢 运行状态</h2>
            <p class="status">服务运行中</p>
            <p>API密钥数量: {len(key_manager.api_keys)}</p>
            <p>可用模型数量: {len(GeminiClient.AVAILABLE_MODELS)}</p>
            <div style='margin-top:15px;'>
                <h3>API密钥状态：</h3>
                <ul style='list-style:none; padding-left:0;'>
                    {"".join([f'<li style="margin-bottom:8px;">🔑 ...{key[-6:]} | 最近429错误: {(datetime.fromtimestamp(error_time).astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S") if (error_time := key_manager.key_error_times.get(key)) else "无错误记录")}</li>' for key in key_manager.api_keys])}
                </ul>
            </div>
        </div>

        <div class="info-box">
            <h2>⚙️ 环境配置</h2>
            <p>每分钟请求限制: {MAX_REQUESTS_PER_MINUTE}</p>
            <p>每IP每日请求限制: {MAX_REQUESTS_PER_DAY_PER_IP}</p>
            <p>最大重试次数: {min(MAX_RETRY, len(key_manager.api_keys)) if key_manager.api_keys else 1}</p>
        </div>
    </body>
    </html>
    """
    return html_content
