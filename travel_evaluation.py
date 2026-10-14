"""旅游助手评测：测试集 + 自动化评估"""
import asyncio
import re
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

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


def check_missing_info(state):
    """Check for missing required info in conversation. Returns list of missing fields."""
    msgs = state["messages"]
    text = "\n".join(m.content or "" for m in msgs if hasattr(m, "content") and m.content)

    checks = {
        "城市": r"(北京|上海|广州|深圳|成都|杭州|南京|重庆|武汉|西安|苏州|天津|长沙|厦门|青岛|大连)|\w+市",
        "月份": r"\d+月|下个月|这个月|本月|六月|七月|八月|九月",
        "天数": r"\d+天|\d+日",
        "预算": r"\d+[块元]|预算.*\d|\d+.*预算|预算",
    }

    missing = []
    for field, pattern in checks.items():
        if not re.search(pattern, text):
            missing.append(field)
    return missing


# ====== 测试用例 ======
# question: 输入
# expect_tools: 期望调用的工具列表（空=不应调工具）
# expect_city: 期望城市参数
# expect_ask: 期望追问信息（True=应追问）
# score: 手动评分 0-2 的判据描述

TEST_CASES = [
    {
        "id": "单轮完整信息-北京",
        "question": "我想去北京玩，6月份，3天，喜欢历史，预算2000",
        "expect_tools": ["get_weather", "recommend_itinerary", "get_food_recommendation", "estimate_daily_cost"],
        "allow_extra": True,
        "alternatives": {"recommend_itinerary": {"plan_budget_tiers"}},
        "expect_city": "北京",
        "expect_ask": False,
        "desc": "完整信息应直接查工具生成行程",
    },
    {
        "id": "单轮完整信息-西安",
        "question": "计划去西安3天，6月去，喜欢美食，预算1500",
        "expect_tools": ["get_weather", "recommend_itinerary", "get_food_recommendation", "estimate_daily_cost"],
        "allow_extra": True,
        "alternatives": {"recommend_itinerary": {"plan_budget_tiers"}},
        "expect_city": "西安",
        "expect_ask": False,
        "desc": "西安完整信息应生成行程",
    },
    {
        "id": "信息不全-缺城市",
        "question": "我6月份想出去玩3天，预算2000",
        "expect_tools": [],
        "expect_ask": True,
        "desc": "缺城市应追问目的地",
    },
    {
        "id": "信息不全-缺日期",
        "question": "我想去北京玩",
        "expect_tools": [],
        "expect_ask": True,
        "desc": "缺日期应追问时间",
    },
    {
        "id": "信息不全-缺预算",
        "question": "去北京玩，6月份，3天",
        "expect_tools": [],
        "expect_ask": True,
        "desc": "缺预算应追问预算",
    },
    {
        "id": "城市名正确提取",
        "question": "我想去西安玩，6月去，3天，预算2000",
        "expect_tools": None,  # 至少调一个工具
        "expect_city": "西安",
        "allow_extra": True,  # 允许额外工具（如 search_web）
        "expect_ask": False,
        "desc": "确保工具调用参数中的城市是西安而不是其他城市",
    },
    {
        "id": "多轮对话-逐步补充",
        "turns": [
            "我想出去旅游",
            "去北京",
            "6月份，3天，预算2000",
        ],
        "expect_tools": [],
        "expect_final_tools": True,  # 最后一轮应调工具
        "expect_ask": True,
        "desc": "多轮对话逐步补充信息",
    },
]


# ====== 评估函数 ======

def check_city_in_tool_calls(messages, expected_city):
    """检查工具调用的城市参数是否正确"""
    for m in messages:
        if hasattr(m, "tool_calls") and m.tool_calls:
            for tc in m.tool_calls:
                args = tc.get("args", {})
                city = args.get("city", "")
                if city and city != expected_city:
                    return False, f"工具 {tc['name']} 传的城市是「{city}」，应为「{expected_city}」"
    return True, "城市参数正确"


def check_tools_called(messages, expected_tools, allow_extra=False, alternatives=None):
    """检查调用了哪些工具。alternatives: {缺失工具: [替代工具]} 允许替代"""
    called = set()
    for m in messages:
        if hasattr(m, "tool_calls") and m.tool_calls:
            for tc in m.tool_calls:
                called.add(tc["name"])

    if expected_tools is None:
        return len(called) > 0, f"调用了工具: {called}"

    called_list = list(called)
    expected_set = set(expected_tools)

    missing = expected_set - called
    extra = called - expected_set

    # 检查替代关系
    if alternatives and missing:
        still_missing = set()
        for t in missing:
            alt_set = alternatives.get(t, set())
            if not alt_set & called:  # 没有调用任何替代工具
                still_missing.add(t)
        missing = still_missing

    if missing:
        return False, f"缺少工具: {missing}"
    if extra and not allow_extra:
        return False, f"多余工具: {extra}"
    return True, f"工具正确: {called_list}"


def check_asked_question(messages, expected_ask):
    """检查 agent 是否追问了信息（区分信息收集 vs 礼貌性反问）"""
    # 找到工具调用前的 AI 消息（信息收集阶段）
    first_tool_idx = None
    for i, m in enumerate(messages):
        if hasattr(m, "tool_calls") and m.tool_calls:
            first_tool_idx = i
            break

    # 只看工具调用之前的 AI 消息
    for i, m in enumerate(messages):
        if first_tool_idx is not None and i >= first_tool_idx:
            break
        if m.type == "ai" and not (hasattr(m, "tool_calls") and m.tool_calls):
            content = m.content or ""
            # 信息收集关键词（问具体缺失字段，不含"还有什么"类反问）
            info_question_kw = ["去哪个城市", "去哪里", "什么时间", "哪天", "几天",
                                "预算", "多少", "偏好", "喜欢什么", "有没有"]
            if any(kw in content for kw in info_question_kw):
                return True if expected_ask else False, "追问了信息"
    # 没有工具调用就看所有 AI 消息
    if first_tool_idx is None:
        for m in messages:
            if m.type == "ai" and not (hasattr(m, "tool_calls") and m.tool_calls):
                content = m.content or ""
                if any(kw in content for kw in ["?", "？", "什么", "哪里"]):
                    return True if expected_ask else False, "追问了信息"
    return (not expected_ask), ("没有追问" if not expected_ask else "应该追问但没有追问")


async def evaluate():
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(Path(__file__).parent / "travel_mcp_server_v2.py")],
    )

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
                missing = check_missing_info(state)
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
生成完整行程，按天展示，每项含价格、时长和间距，末尾算总预算。
不要使用 emoji 符号。""")
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
                        r = await t.ainvoke(tc["args"])
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

            # ====== 跑评测 ======
            simple_cases = [tc for tc in TEST_CASES if "turns" not in tc]
            multi_cases = [tc for tc in TEST_CASES if "turns" in tc]

            print("=" * 80)
            print("                    旅游助手评测报告")
            print("=" * 80)

            all_passed = 0
            all_total = 0

            for tc in simple_cases:
                print(f"\n--- [{tc['id']}] ---")
                print(f"  输入: {tc['question'][:50]}")
                print(f"  期望: {tc['desc']}")

                cfg = {"configurable": {"thread_id": f"eval-{tc['id']}"}}

                msg_count_before = 0
                async for ev in g.astream(
                    {"messages": [HumanMessage(content=tc["question"])]},
                    config=cfg,
                    stream_mode="updates",
                ):
                    for node, up in ev.items():
                        if "messages" in up:
                            msg_count_before += 1

                result = g.get_state(cfg)
                messages = result.values["messages"]

                checks = []
                # 1. 工具调用检查
                if "expect_tools" in tc and tc["expect_tools"] is not None:
                    allow = tc.get("allow_extra", False)
                    alts = tc.get("alternatives", None)
                    ok, msg = check_tools_called(messages, tc["expect_tools"], allow_extra=allow, alternatives=alts)
                    checks.append((ok, f"工具调用: {msg}"))
                elif "expect_tools" in tc and tc["expect_tools"] is None:
                    # 至少调了一个工具
                    called = set()
                    for m in messages:
                        if hasattr(m, "tool_calls") and m.tool_calls:
                            for tc2 in m.tool_calls:
                                called.add(tc2["name"])
                    ok = len(called) > 0
                    checks.append((ok, f"至少调一个工具: {called if ok else '未调用任何工具'}"))

                # 2. 城市参数检查
                if tc.get("expect_city"):
                    ok, msg = check_city_in_tool_calls(messages, tc["expect_city"])
                    checks.append((ok, f"城市验证: {msg}"))

                # 3. 追问检查
                if "expect_ask" in tc:
                    ok, msg = check_asked_question(messages, tc["expect_ask"])
                    checks.append((ok, f"追问行为: {msg}"))

                # 评分
                passed = all(ok for ok, _ in checks)
                all_passed += 1 if passed else 0
                all_total += 1

                for ok, msg in checks:
                    status = "PASS" if ok else "FAIL"
                    print(f"  [{status}] {msg}")

                if passed:
                    print(f"  >> 结果: PASS")
                else:
                    print(f"  >> 结果: FAIL")

            # 多轮对话测试
            for tc in multi_cases:
                print(f"\n--- [{tc['id']}] ---")
                print(f"  多轮对话: {tc['turns']}")
                print(f"  期望: {tc['desc']}")

                cfg = {"configurable": {"thread_id": "eval-multi-turn"}}

                for i, turn in enumerate(tc["turns"]):
                    async for ev in g.astream(
                        {"messages": [HumanMessage(content=turn)]},
                        config=cfg,
                        stream_mode="updates",
                    ):
                        pass

                result = g.get_state(cfg)
                messages = result.values["messages"]
                tool_calls_count = sum(1 for m in messages if hasattr(m, "tool_calls") and m.tool_calls)

                checks = []
                final_asked = check_asked_question(messages, False)
                # 多轮最后一轮应该调工具
                if tc.get("expect_final_tools"):
                    ok = tool_calls_count > 0
                    checks.append((ok, f"多轮最终调工具: {tool_calls_count} 次工具调用"))

                passed = all(ok for ok, _ in checks)
                all_passed += 1 if passed else 0
                all_total += 1

                for ok, msg in checks:
                    status = "PASS" if ok else "FAIL"
                    print(f"  [{status}] {msg}")

                print(f"  >> 结果: {'PASS' if passed else 'FAIL'}")

            print("\n" + "=" * 80)
            print(f"  合计: {all_passed}/{all_total} 通过 ({all_passed/all_total*100:.0f}%)")
            print("=" * 80)


if __name__ == "__main__":
    asyncio.run(evaluate())
