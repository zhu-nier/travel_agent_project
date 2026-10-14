"""旅游助手 Web Server — FastAPI + WebSocket 流式输出"""
import asyncio
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain.chat_models import init_chat_model
from langchain_core.tools import StructuredTool
from langchain.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, TypedDict
import operator


class State(TypedDict):
    messages: Annotated[list, operator.add]


def convert(t, s):
    async def call(**kw):
        r = await s.call_tool(t.name, kw)
        return r.content[0].text
    return StructuredTool(name=t.name, description=t.description or t.name, args_schema=t.inputSchema, coroutine=call)


def check_missing_info(msgs_list):
    text = "\n".join(m.content or "" for m in msgs_list if hasattr(m, "content") and m.content)
    checks = {
        "城市": r"(北京|上海|广州|深圳|成都|杭州|南京|重庆|武汉|西安|苏州|天津|长沙|厦门|青岛|大连)|\w+市",
        "月份": r"\d+月|下个月|这个月|本月|六月|七月|八月|九月",
        "天数": r"\d+天|\d+日",
        "预算": r"\d+[块元]|预算.*\d|\d+.*预算|预算",
    }
    return [f for f, p in checks.items() if not re.search(p, text)]


app = FastAPI(title="旅游助手 Web")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
async def index():
    html = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html.read_text("utf-8"))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # 通知前端准备就绪
    await websocket.send_json({"type": "ready"})

    mcp_script = str(Path(__file__).parent / "travel_mcp_server_v2.py")
    params = StdioServerParameters(command=sys.executable, args=[mcp_script], env={**os.environ})

    try:
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                tools_def = await s.list_tools()
                lc_tools = [convert(t, s) for t in tools_def.tools]
                tm = {t.name: t for t in lc_tools}

                llm = init_chat_model("deepseek-chat", temperature=0.3)
                llm_no_tools = init_chat_model("deepseek-chat", temperature=0.3)
                agent = llm.bind_tools(lc_tools)

                def agent_node(state: State):
                    missing = check_missing_info(state["messages"])
                    system = SystemMessage(content=f"""你是旅游规划助手。
必须收集完以下信息才能查工具：城市、月份、天数、预算。
当前缺失：{', '.join(missing) if missing else '无'}。
缺任何一项就直接问用户，不要调用工具。
完整工作流：
  1. get_weather(查天气)
  2. find_events(搜当月活动)
  3. plan_budget_tiers(生成三档方案) 或 recommend_itinerary(单方案)
  4. optimize_route(优化各天路线)
  5. weather_adjust_schedule(雨天调整室内外)
  6. get_food_recommendation(美食)
  7. estimate_daily_cost(消费估算)
  8. packing_list(生成行李清单)
生成完整行程，按天展示，每项含价格、时长和间距，末尾算总预算。""")
                    msg = agent.invoke([system] + state["messages"])
                    if missing and hasattr(msg, "tool_calls") and msg.tool_calls:
                        question = llm_no_tools.invoke([
                            SystemMessage(content=f"用户提供了部分信息，但还缺少：{', '.join(missing)}。请用中文礼貌地问用户这些信息。不要调用任何工具，不要回答其他问题。"),
                            HumanMessage(content=state["messages"][-1].content or "请帮我规划行程")
                        ])
                        return {"messages": [AIMessage(content=question.content)]}
                    return {"messages": [msg]}

                async def tool_node(state: State):
                    last = state["messages"][-1]
                    results = []
                    for tc in last.tool_calls:
                        t = tm.get(tc["name"])
                        if t:
                            await websocket.send_json({
                                "type": "tool_call",
                                "tool": tc["name"],
                                "args": tc["args"],
                            })
                            r = await t.ainvoke(tc["args"])
                            await websocket.send_json({
                                "type": "tool_result",
                                "tool": tc["name"],
                                "result": r[:300],
                            })
                            results.append(ToolMessage(content=r, tool_call_id=tc["id"]))
                    return {"messages": results}

                def router(state: State):
                    last = state["messages"][-1] if state["messages"] else None
                    if last and hasattr(last, "tool_calls") and last.tool_calls:
                        return "tools"
                    return END

                b = StateGraph(State)
                b.add_node("agent", agent_node)
                b.add_node("tools", tool_node)
                b.add_edge(START, "agent")
                b.add_conditional_edges("agent", router, {"tools": "tools", END: END})
                b.add_edge("tools", "agent")
                g = b.compile(checkpointer=MemorySaver())
                cfg = {"configurable": {"thread_id": "web-session"}}

                while True:
                    data = await websocket.receive_text()
                    payload = json.loads(data)
                    if payload.get("type") != "message":
                        continue
                    q = payload.get("content", "")
                    if q.lower() in ("exit", "quit"):
                        await websocket.send_json({"type": "done"})
                        break

                    async for ev in g.astream(
                        {"messages": [HumanMessage(content=q)]},
                        config=cfg,
                        stream_mode="updates",
                    ):
                        for node_name, up in ev.items():
                            if "messages" in up:
                                m = up["messages"][-1]
                                if m.type == "ai" and not (hasattr(m, "tool_calls") and m.tool_calls):
                                    await websocket.send_json({
                                        "type": "ai_message",
                                        "content": m.content,
                                    })

                    await websocket.send_json({"type": "turn_done"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "content": str(e)})
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
