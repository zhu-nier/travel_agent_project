"""旅游助手评测：测试集 + 自动化评估"""
import asyncio
import sys

from langchain_core.messages import HumanMessage

from travel_core import agent_session, check_missing_info

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
            info_question_kw = ["请补充", "去哪个城市", "去哪里", "什么时间", "哪天", "几天",
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
    async with agent_session() as g:
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

            cfg = {"configurable": {"thread_id": f"eval-{tc['id']}"}, "recursion_limit": 20}

            async for ev in g.astream(
                {"messages": [HumanMessage(content=tc["question"])], "tool_rounds": 0},
                config=cfg,
                stream_mode="updates",
            ):
                pass

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

            cfg = {"configurable": {"thread_id": f"eval-multi-{tc['id']}"}, "recursion_limit": 20}

            checks = []
            for i, turn in enumerate(tc["turns"]):
                previous = g.get_state(cfg).values.get("messages", [])
                previous_count = len(previous)
                async for ev in g.astream(
                    {"messages": [HumanMessage(content=turn)], "tool_rounds": 0},
                    config=cfg,
                    stream_mode="updates",
                ):
                    pass
                turn_messages = g.get_state(cfg).values["messages"][previous_count:]
                if i < len(tc["turns"]) - 1:
                    ok, detail = check_tools_called(turn_messages, [])
                    checks.append((ok, f"第{i+1}轮信息未全禁止工具: {detail}"))
                    ok, detail = check_asked_question(turn_messages, True)
                    checks.append((ok, f"第{i+1}轮追问: {detail}"))
                else:
                    ok, detail = check_asked_question(turn_messages, False)
                    checks.append((ok, f"最后一轮直接规划: {detail}"))

            result = g.get_state(cfg)
            messages = result.values["messages"]
            tool_calls_count = sum(1 for m in turn_messages if getattr(m, "tool_calls", []))

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
        return all_passed == all_total


if __name__ == "__main__":
    try:
        raise SystemExit(0 if asyncio.run(evaluate()) else 1)
    except ValueError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
