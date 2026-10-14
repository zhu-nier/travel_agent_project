"""Project-local configuration shared by all entry points."""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from mcp import StdioServerParameters

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")

SUPPORTED_CITIES = ("北京", "上海", "西安", "成都", "重庆", "广州", "南京", "杭州", "苏州")
MAX_MESSAGE_LENGTH = 4000
MAX_TRAVEL_DAYS = 30
TOOL_TIMEOUT = 30
TURN_TIMEOUT = 180
MAX_TOOL_ROUNDS = 6


def mcp_parameters():
    return StdioServerParameters(
        command=sys.executable,
        args=[str(PROJECT_DIR / "travel_mcp_server_v2.py")],
        cwd=str(PROJECT_DIR),
        env=dict(os.environ),
    )


def create_model():
    if not os.getenv("DEEPSEEK_API_KEY", "").strip():
        raise ValueError("请在项目 .env 中配置 DEEPSEEK_API_KEY 后重试。")
    from langchain_deepseek import ChatDeepSeek
    return ChatDeepSeek(
        model=os.getenv("TRAVEL_MODEL", "deepseek-chat"),
        temperature=0.3,
        timeout=45,
        max_retries=2,
    )
