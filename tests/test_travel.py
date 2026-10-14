"""Offline regression tests: fake models, mock HTTP, and a real local MCP handshake."""
import asyncio
import json
import os
import re
import threading
import unittest
from contextlib import asynccontextmanager
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from mcp import ClientSession
from mcp.client.stdio import stdio_client

import travel_mcp_server_v2 as tools
import travel_web_server as web
from travel_config import MAX_TOOL_ROUNDS, mcp_parameters
from travel_core import build_graph, convert
from travel_request import check_missing_info, extract_request


class RequestTests(unittest.TestCase):
    def test_assistant_and_tool_text_cannot_supply_user_fields(self):
        messages = [HumanMessage(content="我想旅游"),
                    AIMessage(content="北京6月3天，预算2000元？"),
                    ToolMessage(content="上海6月3天2000元", tool_call_id="test")]
        self.assertEqual(check_missing_info(messages), ["城市", "月份", "天数", "预算"])

    def test_budget_word_is_not_an_amount(self):
        self.assertEqual(check_missing_info([HumanMessage(content="北京，6月，3天，预算")]), ["预算"])

    def test_chinese_and_numeric_forms(self):
        for text, expected in [("去成都，十二月，三天，预算两千元", 2000),
                               ("北京6月3天预算1.5万元", 15000),
                               ("北京6月3天2000块", 2000)]:
            with self.subTest(text=text):
                messages = [HumanMessage(content=text)]
                self.assertEqual(check_missing_info(messages), [])
                self.assertEqual(extract_request(messages)["预算"], expected)

    def test_multi_turn_and_latest_correction(self):
        messages = [HumanMessage(content="去北京"), HumanMessage(content="六月，三天"),
                    AIMessage(content="请提供总预算"), HumanMessage(content="2000"),
                    HumanMessage(content="改成去上海，9月，4天，预算3000元")]
        self.assertEqual(extract_request(messages), {"城市": "上海", "月份": "9月", "天数": 4, "预算": 3000})

    def test_invalid_numeric_fields(self):
        messages = [HumanMessage(content="北京13月0天预算-100元")]
        self.assertEqual(check_missing_info(messages), ["月份", "天数", "预算"])

    def test_unsupported_city_correction_does_not_keep_old_city(self):
        messages = [HumanMessage(content="北京6月3天2000元"), HumanMessage(content="改去长沙")]
        self.assertIn("城市", check_missing_info(messages))


class TravelToolTests(unittest.TestCase):
    def test_every_spot_is_scheduled_and_ticket_total_matches(self):
        result = tools.recommend_itinerary("北京", 3, budget=2000)
        for spot in tools.MOCK_SPOTS["北京"]:
            self.assertEqual(result.count(spot["name"]), 1)
        self.assertIn("145元", result)
        self.assertIn("1855元", result)

    def test_days_exceed_spots_produces_explicit_free_days(self):
        self.assertIn("自由活动", tools.recommend_itinerary("北京", 7))

    def test_unknown_budget_never_shows_negative_remaining_budget(self):
        result = tools.recommend_itinerary("北京", 3)
        self.assertIn("未指定预算", result)
        self.assertNotIn("剩余预算", result)

    def test_day_and_budget_boundaries(self):
        for days in (0, -1, 31):
            self.assertIn("天数必须", tools.recommend_itinerary("北京", days))
            self.assertIn("天数必须", tools.plan_budget_tiers("北京", days, 2000))
        self.assertIn("预算必须", tools.recommend_itinerary("北京", 3, budget=-10))

    def test_multiple_preferences(self):
        result = tools.recommend_itinerary("北京", 3, preference="历史和文化")
        self.assertIn("故宫博物院", result)
        self.assertIn("国家博物馆", result)
        self.assertNotIn("颐和园", result)

    def test_empty_route_and_partial_start_are_validation_errors(self):
        self.assertIn("至少提供", tools.optimize_route("北京", []))
        self.assertIn("同时提供", tools.optimize_route("北京", ["天坛公园"], start_lat=0))
        self.assertIn("有效范围", tools.optimize_route("北京", ["天坛公园"], start_lat=100, start_lng=0))

    def test_route_uses_start_and_includes_first_leg(self):
        spots = ["长城(八达岭)", "天坛公园"]
        result = tools.optimize_route("北京", spots, start_lat=39.882, start_lng=116.407)
        self.assertLess(result.index("天坛公园"), result.index("长城(八达岭)"))
        single = tools.optimize_route("北京", ["天坛公园"], start_lat=0, start_lng=0)
        total = float(re.search(r"合计约 ([\d.]+)", single).group(1))
        self.assertGreater(total, 1000)

    def test_deterministic_routes_remove_duplicate_stops(self):
        names = ["故宫博物院", "天坛公园", "天坛公园", "颐和园"]
        ordered = tools.nearest_neighbor(names, tools.build_coord_map("北京"))
        self.assertEqual(len(ordered), 3)
        self.assertEqual(ordered, tools.nearest_neighbor(names, tools.build_coord_map("北京")))

    def test_future_month_never_uses_current_weather(self):
        month = "6月" if date.today().month != 6 else "9月"
        with patch.object(tools, "fetch_weather_qweather") as forecast, patch.object(tools, "fetch_weather_wttr") as current:
            result = tools.get_weather("北京", month)
        forecast.assert_not_called()
        current.assert_not_called()
        self.assertIn("非真实预报", result)

    def test_weather_cache_and_offline_mode(self):
        tools._weather_cache.clear()
        fetch = unittest.mock.Mock(return_value="晴天")
        with patch.dict(os.environ, {"TRAVEL_OFFLINE": "0"}):
            self.assertEqual(tools._cached_weather("test", "北京", fetch), "晴天")
            self.assertEqual(tools._cached_weather("test", "北京", fetch), "晴天")
        self.assertEqual(fetch.call_count, 1)
        with patch.dict(os.environ, {"TRAVEL_OFFLINE": "1"}):
            self.assertIsNone(tools._cached_weather("test", "北京", fetch))
        tools._weather_cache.clear()

    def test_qweather_request_uses_private_host_coordinates_and_header(self):
        import httpx
        data = {"metadata": {"attributions": ["https://developer.qweather.com/attribution.html"]}, "days": [{
            "temperatureMax": {"value": 26}, "temperatureMin": {"value": 15},
            "daytime": {"forecastStartTime": "2026-10-14T07:00+08:00", "condition": {"text": "晴"}},
        }]}
        response = httpx.Response(200, json=data, request=httpx.Request("GET", "https://test.qweatherapi.com"))
        with patch.object(tools, "WEATHER_API_KEY", "offline-test-key"), patch.object(tools, "WEATHER_API_HOST", "test.qweatherapi.com"), patch.dict(os.environ, {"TRAVEL_OFFLINE": "0"}), patch("httpx.get", return_value=response) as get:
            result = tools.fetch_weather_qweather("北京")
        self.assertEqual(result[0]["fxDate"], "2026-10-14")
        self.assertEqual(result[0]["tempMin"], 15)
        self.assertIn("/weather/v1/daily/39.9/116.4", get.call_args.args[0])
        self.assertEqual(get.call_args.kwargs["headers"], {"X-QW-Api-Key": "offline-test-key"})
        self.assertNotIn("key", get.call_args.kwargs["params"])
        tools._weather_cache.clear()

    def test_qweather_malformed_response_falls_back(self):
        import httpx
        response = httpx.Response(200, json={"days": [{}]}, request=httpx.Request("GET", "https://test.qweatherapi.com"))
        with patch.object(tools, "WEATHER_API_KEY", "offline-test-key"), patch.object(tools, "WEATHER_API_HOST", "test.qweatherapi.com"), patch.dict(os.environ, {"TRAVEL_OFFLINE": "0"}), patch("httpx.get", return_value=response):
            self.assertIsNone(tools.fetch_weather_qweather("北京"))

    def test_rainy_schedule_preserves_indoor_priority(self):
        result = tools.weather_adjust_schedule("北京", "6月", ["长城(八达岭)", "国家博物馆", "天坛公园"])
        self.assertLess(result.index("国家博物馆"), result.index("长城(八达岭)"))
        self.assertIn("跳过", result)
        self.assertFalse(tools._indoor_friendly("长城(八达岭)"))

    def test_no_weather_does_not_claim_good_weather(self):
        self.assertIn("无法判断", tools.weather_adjust_schedule("北京", "1月", ["故宫博物院"]))

    def test_warm_temperature_is_not_misclassified_as_cold(self):
        result = tools.packing_list("北京", 3, "2026-10-14：25C，晴")
        self.assertNotIn("外套/毛衣", result)
        self.assertNotIn("短袖短裤", result)
        self.assertIn("外套/毛衣", tools.packing_list("北京", 3, "5-15C"))

    def test_tier_over_budget_is_explicit(self):
        self.assertIn("超出原预算", tools.plan_budget_tiers("北京", 3, 2000))


class FakeModel:
    def __init__(self, tool_name="test_tool", repeat=False):
        self.tool_name = tool_name
        self.repeat = repeat
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        if messages[-1].type == "tool" and not self.repeat:
            return AIMessage(content="参考方案已生成。")
        return AIMessage(content="", tool_calls=[{"name": self.tool_name, "args": {}, "id": f"call-{self.calls}"}])


def fake_tool(coroutine):
    return StructuredTool(name="test_tool", description="Offline test tool", args_schema={"type": "object", "properties": {}}, coroutine=coroutine)


class WorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_fields_skip_model_and_tools(self):
        model = FakeModel()
        graph = build_graph([], model)
        result = await graph.ainvoke({"messages": [HumanMessage(content="我想旅游")]}, {"configurable": {"thread_id": "missing"}})
        self.assertEqual(model.calls, 0)
        self.assertIn("请补充", result["messages"][-1].content)

    async def test_unknown_tool_gets_matching_response(self):
        graph = build_graph([], FakeModel("nonexistent"))
        result = await graph.ainvoke({"messages": [HumanMessage(content="北京6月3天预算2000元")]}, {"configurable": {"thread_id": "unknown"}})
        response = next(m for m in result["messages"] if m.type == "tool")
        self.assertEqual(response.tool_call_id, "call-1")
        self.assertIn("未知工具", response.content)

    async def test_failed_tool_does_not_crash_conversation(self):
        async def fail():
            raise RuntimeError("offline failure")
        graph = build_graph([fake_tool(fail)], FakeModel())
        with self.assertLogs("travel_core", level="ERROR"):
            result = await graph.ainvoke({"messages": [HumanMessage(content="北京6月3天预算2000元")]}, {"configurable": {"thread_id": "failed"}})
        self.assertEqual(result["messages"][-1].content, "参考方案已生成。")
        self.assertIn("查询失败", result["messages"][2].content)

    async def test_tool_timeout_is_recoverable(self):
        async def slow():
            await asyncio.sleep(2)
        with patch("travel_core.TOOL_TIMEOUT", .01):
            graph = build_graph([fake_tool(slow)], FakeModel())
            result = await graph.ainvoke({"messages": [HumanMessage(content="北京6月3天预算2000元")]}, {"configurable": {"thread_id": "timeout"}})
        self.assertIn("超时", result["messages"][2].content)

    async def test_repeated_tool_calls_stop_before_recursion_error(self):
        async def ok():
            return "ok"
        model = FakeModel(repeat=True)
        graph = build_graph([fake_tool(ok)], model)
        result = await graph.ainvoke({"messages": [HumanMessage(content="北京6月3天预算2000元")], "tool_rounds": 0}, {"configurable": {"thread_id": "limit"}, "recursion_limit": 20})
        self.assertEqual(model.calls, MAX_TOOL_ROUNDS)
        self.assertIn("上限", result["messages"][-1].content)

    async def test_multiturn_gate_and_session_isolation(self):
        async def ok():
            return "ok"
        model = FakeModel()
        graph = build_graph([fake_tool(ok)], model)
        config = {"configurable": {"thread_id": "one"}}
        await graph.ainvoke({"messages": [HumanMessage(content="北京")]}, config)
        await graph.ainvoke({"messages": [HumanMessage(content="6月3天预算2000元")], "tool_rounds": 0}, config)
        self.assertEqual(model.calls, 2)
        other = await graph.ainvoke({"messages": [HumanMessage(content="预算1000元")]}, {"configurable": {"thread_id": "two"}})
        self.assertIn("目的地城市", other["messages"][-1].content)
        self.assertEqual(model.calls, 2)

    async def test_mcp_adapter_keeps_all_text_blocks_and_error_flag(self):
        class Session:
            async def call_tool(self, name, args):
                return SimpleNamespace(content=[SimpleNamespace(type="text", text="first"), SimpleNamespace(type="text", text="second")], isError=True, structuredContent=None)
        definition = SimpleNamespace(name="demo", description="demo", inputSchema={"type": "object", "properties": {}})
        result = await convert(definition, Session()).ainvoke({})
        self.assertIn("first\nsecond", result)
        self.assertIn("失败", result)

    async def test_real_stdio_handshake_and_boundary_tool_call(self):
        with patch.dict(os.environ, {"TRAVEL_OFFLINE": "1"}):
            async with stdio_client(mcp_parameters()) as (reader, writer):
                async with ClientSession(reader, writer) as session:
                    await asyncio.wait_for(session.initialize(), 20)
                    definitions = await session.list_tools()
                    self.assertIn("search_web", [tool.name for tool in definitions.tools])
                    result = await session.call_tool("recommend_itinerary", {"city": "北京", "days": 0})
                    self.assertIn("天数必须", result.content[0].text)


@asynccontextmanager
async def offline_agent_session(on_tool_event=None):
    async def ok():
        return "ok"
    yield build_graph([fake_tool(ok)], FakeModel(), on_tool_event)


class WebTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(web.app)

    def test_http_home_and_health(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})
        self.assertIn("旅行需求", self.client.get("/").text)

    def test_invalid_messages_are_rejected_without_closing_session(self):
        with patch.object(web, "agent_session", offline_agent_session):
            with self.client.websocket_connect("/ws", headers={"origin": "http://testserver"}) as socket:
                self.assertEqual(socket.receive_json()["type"], "ready")
                for invalid in ("not json", "[]", '{"type":"message","content":12}', '{"type":"message","content":" "}'):
                    socket.send_text(invalid)
                    self.assertEqual(socket.receive_json()["type"], "error")
                socket.send_json({"type": "message", "content": "北京"})
                self.assertEqual(socket.receive_json()["type"], "ai_message")
                self.assertEqual(socket.receive_json()["type"], "turn_done")

    def test_tool_events_are_correlated_and_reset_clears_context(self):
        with patch.object(web, "agent_session", offline_agent_session):
            with self.client.websocket_connect("/ws") as socket:
                socket.receive_json()
                socket.send_json({"type": "message", "content": "北京6月3天2000元"})
                call = socket.receive_json()
                result = socket.receive_json()
                self.assertEqual(call["type"], "tool_call")
                self.assertEqual(result["call_id"], call["call_id"])
                self.assertEqual(socket.receive_json()["type"], "ai_message")
                self.assertEqual(socket.receive_json()["type"], "turn_done")
                socket.send_json({"type": "reset"})
                self.assertEqual(socket.receive_json()["type"], "reset_done")
                socket.send_json({"type": "message", "content": "预算1000元"})
                self.assertIn("目的地城市", socket.receive_json()["content"])
                self.assertEqual(socket.receive_json()["type"], "turn_done")

    def test_cross_site_websocket_is_rejected(self):
        from starlette.websockets import WebSocketDisconnect
        with self.assertRaises(WebSocketDisconnect) as raised:
            with self.client.websocket_connect("/ws", headers={"origin": "https://untrusted.example"}):
                pass
        self.assertEqual(raised.exception.code, 1008)

    def test_missing_key_is_reported_before_ready(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}), self.assertLogs("travel_web_server", level="ERROR"):
            with self.client.websocket_connect("/ws") as socket:
                result = socket.receive_json()
                self.assertEqual(result["type"], "error")
                self.assertIn("DEEPSEEK_API_KEY", result["content"])
                self.assertEqual(socket.receive()["code"], 1011)

    def test_disconnect_cancels_inflight_work_before_mcp_cleanup(self):
        started, cancelled, cleaned = threading.Event(), threading.Event(), threading.Event()

        class SlowGraph:
            async def astream(self, *args, **kwargs):
                started.set()
                try:
                    await asyncio.sleep(60)
                    yield {}
                finally:
                    cancelled.set()

        @asynccontextmanager
        async def slow_session(*args):
            try:
                yield SlowGraph()
            finally:
                self.assertTrue(cancelled.is_set())
                cleaned.set()

        with patch.object(web, "agent_session", slow_session):
            with self.client.websocket_connect("/ws") as socket:
                socket.receive_json()
                socket.send_json({"type": "message", "content": "北京"})
                self.assertTrue(started.wait(5))
                socket.send_json({"type": "message", "content": "再来一条"})
                self.assertEqual(socket.receive_json()["type"], "busy")
        self.assertTrue(cancelled.is_set())
        self.assertTrue(cleaned.is_set())


if __name__ == "__main__":
    unittest.main()
