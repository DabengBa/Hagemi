from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from .models import ChatCompletionRequest, ChatCompletionResponse, ErrorResponse, ModelList
from .gemini import GeminiClient, ResponseWrapper
from .utils import handle_gemini_error, protect_from_abuse, APIKeyManager, test_api_key, format_log_message
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
    log_msg = format_log_message('ERROR', f"未捕获的异常: %s" % error_message, extra={'status_code': 500, 'error_message': error_message})
    logger.error(log_msg)


sys.excepthook = handle_exception

app = FastAPI()

PASSWORD = os.environ.get("PASSWORD", "123")
SECOND_MODEL = os.environ.get("SECOND_MODEL", "gemini-2.0-flash")
MAX_RETRY = int(os.environ.get("MAX_RETRY", "3"))  # 默认3次重试
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

key_manager = APIKeyManager() # 实例化 APIKeyManager，栈会在 __init__ 中初始化
current_api_key = key_manager.get_available_key()


def switch_api_key():
    global current_api_key
    key = key_manager.get_available_key() # get_available_key 会处理栈的逻辑
    if key:
        current_api_key = key
        log_msg = format_log_message('INFO', f"API key 替换为 → {current_api_key[-6:]}...", extra={'key': current_api_key[-6:], 'request_type': 'switch_key'})
        logger.info(log_msg)
    else:
        log_msg = format_log_message('ERROR', "API key 替换失败，所有API key都已尝试，请重新配置或稍后重试", extra={'key': 'N/A', 'request_type': 'switch_key', 'status_code': 'N/A'})
        logger.error(log_msg)


async def check_keys():
    available_keys = []
    for key in key_manager.api_keys:
        is_valid = await test_api_key(key)
        status_msg = "有效" if is_valid else "无效"
        log_msg = format_log_message('INFO', f"API Key {key[:10]}... {status_msg}.")
        logger.info(log_msg)
        if is_valid:
            available_keys.append(key)
    if not available_keys:
        log_msg = format_log_message('ERROR', "没有可用的 API 密钥！", extra={'key': 'N/A', 'request_type': 'startup', 'status_code': 'N/A'})
        logger.error(log_msg)
    return available_keys


@app.on_event("startup")
async def startup_event():
    log_msg = format_log_message('INFO', "Starting Gemini API proxy...")
    logger.info(log_msg)
    available_keys = await check_keys()
    if available_keys:
        key_manager.api_keys = available_keys
        key_manager._reset_key_stack() # 启动时也确保创建随机栈
        key_manager.show_all_keys()
        log_msg = format_log_message('INFO', f"可用 API 密钥数量：{len(key_manager.api_keys)}")
        logger.info(log_msg)
        # MAX_RETRIES = len(key_manager.api_keys)
        retry_attempts = min(MAX_RETRY, len(key_manager.api_keys)) if key_manager.api_keys else 1
        log_msg = format_log_message('INFO', f"最大重试次数设置为：{retry_attempts}") # 添加日志
        logger.info(log_msg)
        if key_manager.api_keys:
            all_models = await GeminiClient.list_available_models(key_manager.api_keys[0])
            GeminiClient.AVAILABLE_MODELS = all_models
            log_msg = format_log_message('INFO', f"Available models loaded: {len(all_models)} models")
            logger.info(log_msg)

@app.get("/v1/models", response_model=ModelList)
def list_models():
    log_msg = format_log_message('INFO', "Received request to list models", extra={'request_type': 'list_models', 'status_code': 200})
    logger.info(log_msg)
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

    contents, system_instruction = GeminiClient.convert_messages(
        GeminiClient, chat_request.messages)

    retry_attempts = min(MAX_RETRY, len(key_manager.api_keys)) if key_manager.api_keys else 1 # 重试次数等于密钥数量和MAX_RETRY之中最小值，至少尝试 1 次
    api_version_to_use = "v1alpha" # Start with v1alpha

    for attempt in range(1, retry_attempts + 1):
        # Get a key only if it's the first attempt or after a key switch (not after version switch)
        if attempt == 1 or 'switch_key_occurred' in locals() and switch_key_occurred:
             current_api_key = key_manager.get_available_key()
             api_version_to_use = "v1alpha" # Reset to v1alpha when getting a new key
             switch_key_occurred = False # Reset flag

        if current_api_key is None: # 检查是否获取到 API 密钥
            log_msg_no_key = format_log_message('WARNING', "没有可用的 API 密钥，跳过本次尝试", extra={'request_type': request_type, 'model': chat_request.model, 'status_code': 'N/A'})
            logger.warning(log_msg_no_key)
            break  # 如果没有可用密钥，跳出循环

        extra_log = {'key': current_api_key[-6:], 'request_type': request_type, 'model': chat_request.model, 'status_code': 'N/A', 'error_message': '', 'api_version': api_version_to_use}
        log_msg = format_log_message('INFO', f"第 {attempt}/{retry_attempts} 次尝试 (API Version: {api_version_to_use})", extra=extra_log)
        logger.info(log_msg)

        gemini_client = GeminiClient(current_api_key)
        try:
            if chat_request.stream:
                async def stream_generator():
                    try:
                        async for chunk in gemini_client.stream_chat(chat_request, contents, safety_settings_g2 if 'gemini-2.0-flash-exp' in chat_request.model else safety_settings, system_instruction, api_version=api_version_to_use):
                            formatted_chunk = {"id": "chatcmpl-someid", "object": "chat.completion.chunk", "created": 1234567,
                                               "model": chat_request.model, "choices": [{"delta": {"role": "assistant", "content": chunk}, "index": 0, "finish_reason": None}]}
                            yield f"data: {json.dumps(formatted_chunk)}\n\n"
                        yield "data: [DONE]\n\n"

                    except asyncio.CancelledError:
                        extra_log_cancel = {'key': current_api_key[-6:], 'request_type': request_type, 'model': chat_request.model, 'error_message': '客户端已断开连接'}
                        log_msg = format_log_message('INFO', "客户端连接已中断", extra=extra_log_cancel)
                        logger.info(log_msg)
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
                        response_content = await asyncio.to_thread(gemini_client.complete_chat, chat_request, contents, safety_settings_g2 if 'gemini-2.0-flash-exp' in chat_request.model else safety_settings, system_instruction, api_version=api_version_to_use)
                        return response_content
                    except asyncio.CancelledError:
                        extra_log_gemini_cancel = {'key': current_api_key[-6:], 'request_type': request_type, 'model': chat_request.model, 'error_message': '客户端断开导致API调用取消', 'api_version': api_version_to_use}
                        log_msg = format_log_message('INFO', "API调用因客户端断开而取消", extra=extra_log_gemini_cancel)
                        logger.info(log_msg)
                        raise

                gemini_task = asyncio.create_task(run_gemini_completion())

                try:
                    done, pending = await asyncio.wait(
                        [gemini_task],
                        return_when=asyncio.FIRST_COMPLETED
                    )

                    if gemini_task in done:
                        response_content = gemini_task.result()
                        if response_content.text == "":
                            extra_log_empty_response = {'key': current_api_key[-6:], 'request_type': request_type, 'model': chat_request.model, 'status_code': 204}
                            log_msg = format_log_message('INFO', "Gemini API 返回空响应", extra=extra_log_empty_response)
                            logger.info(log_msg)
                            
                            # 如果当前不是SECOND_MODEL，则切换到SECOND_MODEL重试
                            if chat_request.model != SECOND_MODEL:
                                log_msg = format_log_message('INFO', f"尝试切换到备用模型 {SECOND_MODEL}", extra={'key': current_api_key[-6:], 'request_type': request_type, 'model': SECOND_MODEL, 'status_code': 'N/A'})
                                logger.info(log_msg)
                                chat_request.model = SECOND_MODEL
                                continue
                            
                            # 继续循环
                            continue
                        response = ChatCompletionResponse(id="chatcmpl-someid", object="chat.completion", created=1234567890, model=chat_request.model,
                                                        choices=[{"index": 0, "message": {"role": "assistant", "content": response_content.text}, "finish_reason": "stop"}])
                        extra_log_success = {'key': current_api_key[-6:], 'request_type': request_type, 'model': chat_request.model, 'status_code': 200}
                        log_msg = format_log_message('INFO', "请求处理成功", extra=extra_log_success)
                        logger.info(log_msg)
                        try:
                            is_disconnected = await http_request.is_disconnected()
                        except Exception as e_disconnect:
                            # 记录 is_disconnected 调用本身的错误
                            extra_log_disconnect_error = {'key': current_api_key[-6:] if current_api_key else 'N/A', 'request_type': request_type, 'model': chat_request.model, 'error_message': f"Error calling http_request.is_disconnected: {e_disconnect}"}
                            log_msg_disconnect_error = format_log_message('ERROR', "调用 http_request.is_disconnected 时出错", extra=extra_log_disconnect_error)
                            logger.error(log_msg_disconnect_error)
                            # 假设未断开，让后续逻辑继续尝试返回数据
                            is_disconnected = False

                        if is_disconnected:
                            extra_log_client_disconnect = {'key': current_api_key[-6:] if current_api_key else 'N/A', 'request_type': request_type, 'model': chat_request.model, 'error_message': '检测到客户端断开连接'}
                            log_msg = format_log_message('INFO', "客户端连接已中断 (但仍尝试返回)", extra=extra_log_client_disconnect)
                            logger.info(log_msg)
                            if response_content.text:
                                extra_log_response = {
                                    'key': current_api_key[-6:] if current_api_key else 'N/A',
                                    'request_type': request_type,
                                    'model': chat_request.model,
                                    'response_content': response_content.text
                                }
                                log_msg = format_log_message('INFO', f"获取到响应内容 (客户端已断开): {extra_log_response['response_content']}", extra=extra_log_response)
                                logger.info(log_msg)
                            # 注意：即使客户端断开，我们仍然尝试返回响应
                            # 如果返回失败，FastAPI/Uvicorn 会处理底层的连接错误
                        try:
                            return response # 无论 is_disconnected 检查是否出错或结果如何，都尝试返回
                        except Exception as e_return:
                            # 记录在返回响应时发生的错误
                            extra_log_return_error = {'key': current_api_key[-6:] if current_api_key else 'N/A', 'request_type': request_type, 'model': chat_request.model, 'error_message': f"Error during return response: {e_return}"}
                            log_msg_return_error = format_log_message('ERROR', "返回响应时发生错误", extra=extra_log_return_error)
                            logger.error(log_msg_return_error)
                            # 重新抛出异常，让 FastAPI 或 ASGI 服务器处理，避免触发外层重试
                            raise e_return

                except asyncio.CancelledError:
                    extra_log_request_cancel = {'key': current_api_key[-6:] if current_api_key else 'N/A', 'request_type': request_type, 'model': chat_request.model, 'error_message':"请求被取消" }
                    log_msg = format_log_message('INFO', "请求取消", extra=extra_log_request_cancel)
                    logger.info(log_msg)
                    raise

        except HTTPException as e:
            if e.status_code == status.HTTP_408_REQUEST_TIMEOUT:
                extra_log = {'key': current_api_key[-6:], 'request_type': request_type, 'model': chat_request.model, 
                            'status_code': 408, 'error_message': '客户端连接中断'}
                log_msg = format_log_message('ERROR', "客户端连接中断，终止后续重试", extra=extra_log)
                logger.error(log_msg)
                raise  
            else:
                raise  
        except Exception as e:
            error_type = handle_gemini_error(e, current_api_key, key_manager)
            
            # Check if it's a 500 error and we are using v1alpha
            if error_type == "GEMINI_500_ERROR" and api_version_to_use == "v1alpha":
                if attempt < retry_attempts:
                    api_version_to_use = "v1beta"
                    log_msg_version_switch = format_log_message('WARNING', f"遇到 500 错误，尝试切换到 API 版本 v1beta 进行重试 (Key: ...{current_api_key[-6:]})", extra={'key': current_api_key[-6:], 'request_type': request_type, 'model': chat_request.model, 'status_code': 500, 'error_message': 'Switching to v1beta'})
                    logger.warning(log_msg_version_switch)
                    switch_key_occurred = False # Ensure we don't switch key on next loop iteration
                    await asyncio.sleep(RETRY_DELAY) # 添加重试延迟
                    continue # Retry with the same key but v1beta
                else:
                    # Last attempt failed even after trying v1alpha
                    pass # Let it fall through to the final error

            # Handle other errors or 500 error when already using v1beta
            elif attempt < retry_attempts:
                switch_api_key()
                switch_key_occurred = True # Flag that key was switched
                # api_version_to_use is reset at the start of the loop when key is switched
                continue
            
                await asyncio.sleep(RETRY_DELAY) # 添加重试延迟
            # If it's the last attempt and it failed, let it fall through

    msg = "所有API密钥均失败,请稍后重试"
    extra_log_all_fail = {'key': "ALL", 'request_type': request_type, 'model': chat_request.model, 'status_code': 500, 'error_message': msg}
    log_msg = format_log_message('ERROR', msg, extra=extra_log_all_fail)
    logger.error(log_msg)
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=msg)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest, http_request: Request, _: None = Depends(verify_password)):
    return await process_request(request, http_request, "stream" if request.stream else "non-stream")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_message = translate_error(str(exc))
    extra_log_unhandled_exception = {'status_code': 500, 'error_message': error_message}
    log_msg = format_log_message('ERROR', f"Unhandled exception: {error_message}", extra=extra_log_unhandled_exception)
    logger.error(log_msg)
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
