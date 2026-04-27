"""
Railway 部署版：企业智能客服 Web 聊天助手

本项目不包含 RAG，只演示：
1. FastAPI 后端
2. 静态前端页面
3. 后端调用大模型
4. 部署到 Railway 后获得公网访问地址

本地运行：
    python3 -m uvicorn app:app --reload

Railway 启动：
    python -m uvicorn app:app --host 0.0.0.0 --port $PORT
"""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

QWEN_CHAT_MODEL = "qwen-plus"
DEEPSEEK_CHAT_MODEL = "deepseek-chat"

# Railway 里可以配置 LLM_PROVIDER=qwen 或 LLM_PROVIDER=deepseek。
# 不配置时默认使用 qwen。
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "qwen")


SYSTEM_PROMPT = """
你是星河科技有限公司的企业智能客服。
你的职责是回答客户关于产品、价格、售后、账号、发票、使用问题的咨询。

请遵守以下规则：
1. 回答要礼貌、简洁、专业。
2. 如果用户询问具体订单、个人隐私、实时库存、公司内部资料等你无法确认的信息，
   请说明“目前无法查询，需要人工客服进一步处理”。
3. 如果用户的问题和企业客服无关，请简短回答，并引导用户提出产品或服务相关问题。
4. 本项目还没有接入企业知识库，所以不要编造具体政策、价格、订单状态或内部数据。
"""


app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str


def check_api_key(api_key, name):
    if not api_key:
        raise RuntimeError(
            f"没有读取到环境变量 {name}。请在 Railway Variables 中配置它。"
        )


def get_llm_client():
    if LLM_PROVIDER == "qwen":
        api_key = os.getenv("DASHSCOPE_API_KEY")
        check_api_key(api_key, "DASHSCOPE_API_KEY")
        return OpenAI(api_key=api_key, base_url=QWEN_BASE_URL), QWEN_CHAT_MODEL

    if LLM_PROVIDER == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        check_api_key(api_key, "DEEPSEEK_API_KEY")
        return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL), DEEPSEEK_CHAT_MODEL

    raise RuntimeError('LLM_PROVIDER 只能是 "qwen" 或 "deepseek"')


def call_llm(user_message):
    client, model = get_llm_client()

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        temperature=0.3,
    )

    return completion.choices[0].message.content


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest):
    user_message = request.message.strip()
    if not user_message:
        return ChatResponse(answer="请输入您的问题。")

    try:
        answer = call_llm(user_message)
    except Exception as error:
        answer = f"调用大模型失败：{error}"

    return ChatResponse(answer=answer)
