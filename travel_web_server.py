"""FastAPI / WebSocket entry point with bounded async turns and disconnect cleanup."""
import asyncio
import json
import logging
from contextlib import suppress
from uuid import uuid4

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from langchain_core.messages import HumanMessage

from travel_config import MAX_MESSAGE_LENGTH, PROJECT_DIR, TURN_TIMEOUT
from travel_core import agent_session, message_text

logger = logging.getLogger(__name__)
app = FastAPI(title="旅游助手 Web")


@app.get("/")
async def index():
    return HTMLResponse((PROJECT_DIR / "static" / "index.html").read_text("utf-8"))


@app.get("/health")
async def health():
    return {"status": "ok"}


def validate_payload(raw):
    if len(raw) > MAX_MESSAGE_LENGTH * 6 + 256:
        raise ValueError("消息过长。")
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        raise ValueError("消息格式无效，请发送 JSON 对象。") from None
    if not isinstance(payload, dict) or payload.get("type") not in ("message", "reset"):
        raise ValueError("不支持的消息类型。")
    if payload["type"] == "reset":
        return payload
    content = payload.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("请输入旅行需求。")
    if len(content) > MAX_MESSAGE_LENGTH:
        raise ValueError(f"每条消息最多 {MAX_MESSAGE_LENGTH} 个字符。")
    return {"type": "message", "content": content.strip()}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    origin = websocket.headers.get("origin")
    if origin and origin != f"{websocket.url.scheme.replace('ws', 'http')}://{websocket.headers.get('host')}":
        await websocket.close(code=1008)
        return
    await websocket.accept()
    turn_task = None
    try:
        async with agent_session(websocket.send_json) as graph:
            config = {"configurable": {"thread_id": uuid4().hex}, "recursion_limit": 20}
            await websocket.send_json({"type": "ready"})

            async def run_turn(content):
                try:
                    async with asyncio.timeout(TURN_TIMEOUT):
                        async for event in graph.astream(
                            {"messages": [HumanMessage(content=content)], "tool_rounds": 0},
                            config=config, stream_mode="updates",
                        ):
                            for update in event.values():
                                for message in update.get("messages", []):
                                    if message.type == "ai" and not getattr(message, "tool_calls", []):
                                        await websocket.send_json({"type": "ai_message", "content": message_text(message)})
                except WebSocketDisconnect:
                    raise
                except Exception:
                    logger.exception("Conversation failed")
                    config["configurable"]["thread_id"] = uuid4().hex
                    await websocket.send_json({"type": "error", "content": "本轮处理失败或超时，已开始新对话，请重试。"})
                else:
                    await websocket.send_json({"type": "turn_done"})

            try:
                while True:
                    if turn_task and turn_task.done():
                        await turn_task
                        turn_task = None
                    raw = await websocket.receive_text()
                    try:
                        payload = validate_payload(raw)
                    except ValueError as error:
                        await websocket.send_json({"type": "error", "content": str(error)})
                        continue
                    if turn_task and not turn_task.done():
                        await websocket.send_json({"type": "busy", "content": "正在规划，请等待当前回复。"})
                        continue
                    if turn_task:
                        await turn_task
                        turn_task = None
                    if payload["type"] == "reset":
                        config["configurable"]["thread_id"] = uuid4().hex
                        await websocket.send_json({"type": "reset_done"})
                        continue
                    if payload["content"].lower() in ("exit", "quit"):
                        await websocket.send_json({"type": "done"})
                        await websocket.close(code=1000)
                        break
                    turn_task = asyncio.create_task(run_turn(payload["content"]))
            finally:
                if turn_task:
                    turn_task.cancel()
                    with suppress(asyncio.CancelledError, WebSocketDisconnect, RuntimeError, OSError):
                        await turn_task
    except WebSocketDisconnect:
        pass
    except Exception as error:
        logger.exception("WebSocket session failed")
        with suppress(WebSocketDisconnect, RuntimeError, OSError):
            message = str(error) if isinstance(error, ValueError) else "服务初始化失败，请检查配置并稍后重试。"
            await websocket.send_json({"type": "error", "content": message})
            await websocket.close(code=1011)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, ws_max_size=32 * 1024)
