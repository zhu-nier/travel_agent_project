"""旅游助手 V2：共享异步工作流的命令行入口。"""
import asyncio
import logging
import sys
from uuid import uuid4

from langchain_core.messages import HumanMessage

from travel_config import MAX_MESSAGE_LENGTH, TURN_TIMEOUT
from travel_core import agent_session, message_text


async def show_tool(event):
    if event["type"] == "tool_call":
        print(f"  [查询 {event['tool']}] {event['args']}", flush=True)


async def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    async with agent_session(show_tool) as graph:
        config = {"configurable": {"thread_id": uuid4().hex}, "recursion_limit": 20}
        print("旅游助手 V2 · 输入 exit 退出，输入 reset 开始新对话。")
        while True:
            try:
                question = (await asyncio.to_thread(input, "\n> ")).strip()
            except EOFError:
                break
            if question.lower() in ("exit", "quit"):
                break
            if question.lower() == "reset":
                config["configurable"]["thread_id"] = uuid4().hex
                print("已开始新对话。")
                continue
            if not question:
                continue
            if len(question) > MAX_MESSAGE_LENGTH:
                print(f"每条消息最多 {MAX_MESSAGE_LENGTH} 个字符。")
                continue
            try:
                async with asyncio.timeout(TURN_TIMEOUT):
                    async for event in graph.astream(
                        {"messages": [HumanMessage(content=question)], "tool_rounds": 0},
                        config=config, stream_mode="updates",
                    ):
                        for update in event.values():
                            for message in update.get("messages", []):
                                if message.type == "ai" and not getattr(message, "tool_calls", []):
                                    print("\n" + message_text(message))
            except Exception:
                logging.exception("Conversation failed")
                config["configurable"]["thread_id"] = uuid4().hex
                print("本轮处理失败或超时，已开始新对话，请重试。")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except ValueError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
