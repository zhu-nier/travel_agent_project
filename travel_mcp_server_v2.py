"""旅游助手 MCP Server V2：真实 API + 模拟数据双模式"""
import os
import json
import logging
import math
import re
import sys
import time
from datetime import date
from urllib.parse import quote

from travel_config import MAX_TRAVEL_DAYS, SUPPORTED_CITIES
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("旅游数据服务 V2")

# ====== API 密钥 ======
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")
WEATHER_API_HOST = os.getenv("WEATHER_API_HOST", "").strip().lower()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
logger = logging.getLogger(__name__)
_weather_cache = {}
# Approximate city-center coordinates used for city-level forecasts.
CITY_COORDINATES = {
    "北京": (39.90, 116.40), "上海": (31.23, 121.47), "西安": (34.26, 108.94),
    "成都": (30.67, 104.07), "重庆": (29.56, 106.55), "广州": (23.13, 113.26),
    "南京": (32.06, 118.80), "杭州": (30.27, 120.15), "苏州": (31.30, 120.59),
}


def _cached_weather(provider, city, fetch):
    """Bounded ten-minute cache; failed responses are not cached."""
    if os.getenv("TRAVEL_OFFLINE", "").lower() in ("1", "true", "yes"):
        return None
    key = (provider, city)
    cached = _weather_cache.get(key)
    if cached and cached[0] > time.monotonic():
        return cached[1]
    value = fetch()
    if value:
        if len(_weather_cache) >= 64:
            _weather_cache.pop(next(iter(_weather_cache)))
        _weather_cache[key] = (time.monotonic() + 600, value)
    return value


def fetch_weather_wttr(city: str) -> str:
    """通过 wttr.in 查询天气（无需 API Key，全球城市）"""
    import httpx
    def fetch():
        try:
            resp = httpx.get(f"https://wttr.in/{quote(city, safe='')}?format=%C+%t&lang=zh", timeout=8)
            resp.raise_for_status()
            text = resp.text.strip()
            return text if len(text) < 300 and "<html" not in text.lower() else ""
        except httpx.HTTPError:
            return ""
    return _cached_weather("wttr", city, fetch) or ""


def fetch_weather_qweather(city: str) -> list | None:
    """通过和风天气 API 查询（需要 WEATHER_API_KEY）"""
    if not WEATHER_API_KEY or city not in CITY_COORDINATES:
        return None
    if not re.fullmatch(r"[a-z0-9]+(?:[.-][a-z0-9]+)*\.qweatherapi\.com", WEATHER_API_HOST):
        return None
    import httpx
    def fetch():
        try:
            latitude, longitude = CITY_COORDINATES[city]
            resp = httpx.get(
                f"https://{WEATHER_API_HOST}/weather/v1/daily/{latitude}/{longitude}",
                headers={"X-QW-Api-Key": WEATHER_API_KEY},
                params={"days": 3, "localTime": "true", "lang": "zh"}, timeout=8,
            )
            resp.raise_for_status()
            data = resp.json()
            days = data.get("days")
            if isinstance(days, list) and days:
                attributions = data.get("metadata", {}).get("attributions", [])
                return [{
                    "tempMax": day["temperatureMax"]["value"],
                    "tempMin": day["temperatureMin"]["value"],
                    "textDay": day["daytime"]["condition"]["text"],
                    "fxDate": day["daytime"]["forecastStartTime"][:10],
                    "attributions": attributions,
                } for day in days]
        except (httpx.HTTPError, ValueError, TypeError, AttributeError, KeyError):
            pass
        return None
    return _cached_weather("qweather", city, fetch)

# ====== 模拟数据（兜底） ======
MOCK_WEATHER = {
    ("北京", "6月"): "22-32C, 炎热, 偶有雷阵雨",
    ("北京", "3月"): "5-15C, 干燥",
    ("北京", "9月"): "15-25C, 凉爽",
    ("上海", "6月"): "23-30C, 梅雨季, 潮湿",
    ("上海", "3月"): "8-18C, 湿润",
    ("上海", "9月"): "20-28C, 舒适",
    ("西安", "6月"): "22-33C, 炎热",
    ("西安", "3月"): "5-16C, 干燥",
    ("西安", "9月"): "15-25C, 凉爽",
    ("成都", "6月"): "22-28C, 闷热",
    ("成都", "3月"): "10-18C, 阴雨",
    ("成都", "9月"): "18-25C, 舒适",
    ("广州", "6月"): "26-34C, 湿热多雨",
    ("重庆", "6月"): "24-33C, 闷热潮湿",
    ("重庆", "3月"): "12-20C, 多雾",
    ("重庆", "9月"): "20-27C, 舒适",
}

MOCK_SPOTS = {
    "北京": [
        {"name": "故宫博物院", "price": 60, "hours": 3, "type": "历史", "lat": 39.916, "lng": 116.397},
        {"name": "长城(八达岭)", "price": 40, "hours": 5, "type": "历史", "lat": 40.354, "lng": 116.021},
        {"name": "天坛公园", "price": 15, "hours": 2, "type": "历史", "lat": 39.882, "lng": 116.407},
        {"name": "颐和园", "price": 30, "hours": 3, "type": "自然", "lat": 39.999, "lng": 116.275},
        {"name": "国家博物馆", "price": 0, "hours": 3, "type": "文化", "lat": 39.905, "lng": 116.397},
    ],
    "上海": [
        {"name": "迪士尼乐园", "price": 475, "hours": 10, "type": "娱乐", "lat": 31.144, "lng": 121.658},
        {"name": "外滩", "price": 0, "hours": 1, "type": "城市", "lat": 31.240, "lng": 121.490},
        {"name": "豫园", "price": 30, "hours": 2, "type": "历史", "lat": 31.227, "lng": 121.487},
        {"name": "上海博物馆", "price": 0, "hours": 3, "type": "文化", "lat": 31.230, "lng": 121.472},
    ],
    "西安": [
        {"name": "兵马俑", "price": 120, "hours": 4, "type": "历史", "lat": 34.385, "lng": 109.273},
        {"name": "西安城墙", "price": 54, "hours": 2, "type": "历史", "lat": 34.266, "lng": 108.942},
        {"name": "大雁塔", "price": 50, "hours": 2, "type": "历史", "lat": 34.218, "lng": 108.959},
        {"name": "回民街", "price": 0, "hours": 2, "type": "美食", "lat": 34.265, "lng": 108.942},
    ],
}

MOCK_FOOD = {
    "北京": ["北京烤鸭(约150元/只)", "炸酱面(约20元)", "涮羊肉(约80元/人)"],
    "上海": ["小笼包(约30元)", "生煎包(约15元)", "红烧肉(约60元)"],
    "西安": ["肉夹馍(约12元)", "羊肉泡馍(约35元)", "凉皮(约10元)"],
    "成都": ["火锅(约100元/人)", "串串香(约60元/人)", "担担面(约15元)"],
}

VALID_CITIES = set(SUPPORTED_CITIES)
INDOOR_SPOTS = {"国家博物馆", "上海博物馆"}


def validate_city(city: str) -> str | None:
    """返回 None 表示城市有效，否则返回错误提示"""
    if city in VALID_CITIES:
        return None
    return f"不支持的城市「{city}」，可选: {', '.join(sorted(VALID_CITIES))}"


def validate_trip(days, budget=0):
    if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= MAX_TRAVEL_DAYS:
        return f"天数必须是 1—{MAX_TRAVEL_DAYS} 的整数。"
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        return "预算必须是非负整数，0 表示未指定预算。"
    return None


def validate_month(month):
    return None if re.fullmatch(r"(?:[1-9]|1[0-2])月", month) else "月份格式应为 1月—12月。"


def haversine(lat1, lng1, lat2, lng2):
    """计算两点间球面距离（公里）"""
    import math
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng / 2) ** 2
    return R * 2 * math.asin(math.sqrt(min(1.0, max(0.0, a))))


def build_coord_map(city):
    """根据城市名构建 {景点名: (lat, lng)} 字典"""
    spots = MOCK_SPOTS.get(city, [])
    return {s["name"]: (s["lat"], s["lng"]) for s in spots if "lat" in s and "lng" in s}


def nearest_neighbor(spots_list, coord_map, start_coord=None):
    """贪心最近邻算法优化游览顺序。spots_list: 景点名列表, coord_map: {名: (lat,lng)}"""
    remaining = list(dict.fromkeys(spots_list))
    ordered = []
    current_coord = start_coord
    while remaining:
        nearest = remaining[0] if current_coord is None else min(
            remaining,
            key=lambda name: haversine(*current_coord, *coord_map[name]) if name in coord_map else float("inf"),
        )
        ordered.append(nearest)
        remaining.remove(nearest)
        current_coord = coord_map.get(nearest)
    return ordered


def _indoor_friendly(spot_name):
    """判断景点类型是否适合雨天室内活动"""
    return spot_name in INDOOR_SPOTS


def _get_climate(city, month):
    """获取城市月份的气候描述"""
    key = (city, month)
    w = MOCK_WEATHER.get(key, "")
    if w:
        return w
    # Current weather must not stand in for an arbitrary travel month.
    return ""


def _is_rainy(climate):
    return any(kw in climate for kw in ["雨", "湿", "雷", "潮"])


def _generate_itinerary_text(city, days, budget, preference):
    """核心行程生成逻辑（被 recommend_itinerary 和 plan_budget_tiers 共用）"""
    error = validate_trip(days, budget)
    if error:
        return error
    spots = MOCK_SPOTS.get(city, [])
    if not spots:
        return f"暂无{city}的景点数据。"

    if preference:
        filtered = [s for s in spots if s["type"] in preference]
        if filtered:
            spots = filtered

    if budget > 0:
        per_day = budget // days
        spots = [s for s in spots if s["price"] <= per_day * 0.6]
        selected = []
        remaining_budget = budget
        for spot in spots:
            if spot["price"] <= remaining_budget:
                selected.append(spot)
                remaining_budget -= spot["price"]
        spots = selected

    if not spots:
        return f"{city}没有符合「{preference}」偏好且在预算内的景点。"

    budget_label = f"总预算{budget}元" if budget else "未指定预算"
    lines = [f"【{city} {days}日行程推荐】{budget_label}", "数据来源：本地模拟景点与参考票价，实际费用和开放信息需核对。"]
    schedule = [[] for _ in range(days)]
    durations = [0] * days
    for spot in spots:
        day_index = min(range(days), key=lambda index: durations[index])
        schedule[day_index].append(spot)
        durations[day_index] += spot["hours"]
    for d, day_spots in enumerate(schedule):
        lines.append(f"\n第{d+1}天:")
        if not day_spots:
            lines.append("  - 自由活动或休息，暂无更多本地景点数据。")
        for s in day_spots:
            lines.append(f"  - {s['name']} ({s['type']}, 门票{s['price']}元, 建议{s['hours']}小时)")
        if durations[d] > 8:
            lines.append("  提醒：当天游览超过8小时，建议减少景点或调整天数。")

    total = sum(s["price"] for s in spots)
    lines.append(f"\n景点门票合计: {total}元")
    if budget:
        lines.append(f"剩余预算(食宿交通): {budget - total}元（未包含城际交通）")
    return "\n".join(lines)
@mcp.tool()
def search_web(query: str) -> str:
    """搜索网络获取实时信息。query: 搜索关键词"""
    if os.getenv("TRAVEL_OFFLINE", "").lower() in ("1", "true", "yes"):
        return "离线模式：未执行网络搜索。"
    if not TAVILY_API_KEY:
        return "未配置 TAVILY_API_KEY，无法搜索。"
    try:
        from tavily import TavilyClient
        c = TavilyClient(api_key=TAVILY_API_KEY)
        r = c.search(query=query, search_depth="advanced", max_results=5)
        snippets = [f"- {x.get('title', '')}: {x.get('content', '')[:1000]}\n  来源: {x.get('url', '')}" for x in r.get("results", [])]
        return "搜索结果:\n" + "\n".join(snippets) if snippets else "无结果。"
    except Exception:
        return "搜索失败，请检查配置或稍后重试。"


@mcp.tool()
def optimize_route(city: str, spot_names: list[str], start_lat: float | None = None, start_lng: float | None = None) -> str:
    """用最近邻算法优化景点游览顺序，减少走回头路。city: 城市名, spot_names: 景点名称列表, start_lat/start_lng: 起点坐标(可选，默认第一个景点)"""
    err = validate_city(city)
    if err:
        return err
    if not spot_names:
        return "请至少提供一个景点。"
    if (start_lat is None) != (start_lng is None):
        return "起点经纬度必须同时提供。"
    if start_lat is not None and (not math.isfinite(start_lat) or not math.isfinite(start_lng)
                                 or not -90 <= start_lat <= 90 or not -180 <= start_lng <= 180):
        return "起点经纬度超出有效范围。"
    coord_map = build_coord_map(city)
    if not coord_map:
        return f"暂无{city}的坐标数据。"
    unknown = [s for s in spot_names if s not in coord_map]
    if unknown:
        return f"未知景点: {unknown}"

    start = (start_lat, start_lng) if start_lat is not None else None
    ordered = nearest_neighbor(spot_names, coord_map, start)
    lines = [f"【{city}路线优化】按游览顺序:"]
    dists = []
    prev_coord = start if start is not None else coord_map[ordered[0]]
    for s in ordered:
        cur = coord_map[s]
        d = haversine(*prev_coord, *cur)
        dists.append(d)
        lines.append(f"  {len(lines)}. {s} (距上一点 {d:.1f}km)")
        prev_coord = cur
    total = sum(dists)
    lines.append(f"\n景点间直线距离合计约 {total:.1f}km（含指定起点；非导航里程）")
    return "\n".join(lines)


@mcp.tool()
def find_events(city: str, month: str, year: int | None = None) -> str:
    """搜索目的地当月演唱会/展览/节日活动。city: 城市名, month: 月份(如6月), year: 年份"""
    error = validate_city(city) or validate_month(month)
    if error:
        return error
    year = date.today().year if year is None else year
    if not 2000 <= year <= 2100:
        return "年份必须在 2000—2100 之间。"
    if os.getenv("TRAVEL_OFFLINE", "").lower() in ("1", "true", "yes"):
        return "离线模式：未查询实时活动。"
    if not TAVILY_API_KEY:
        return "未配置 TAVILY_API_KEY，无法搜索活动。"
    try:
        from tavily import TavilyClient
        c = TavilyClient(api_key=TAVILY_API_KEY)
        query = f"{year}年{month} {city} 演唱会 展览 活动 演出"
        r = c.search(query=query, search_depth="advanced", max_results=8)
        snippets = [f"- {x.get('title', '')}: {x['content'][:200]}\n  来源: {x.get('url', '')}" for x in r.get("results", []) if x.get("content")]
        return f"【{city}{year}年{month}活动】\n" + "\n".join(snippets) if snippets else f"未找到{city}{month}的活动信息。"
    except Exception:
        return "搜索活动失败，请检查配置或稍后重试。"


@mcp.tool()
def get_weather(city: str, month: str) -> str:
    """查询城市天气。city: 城市名, month: 月份(如6月)

优先用真实 API（和风天气需要 WEATHER_API_KEY），其次 wttr.in（无需 Key），最后回退模拟数据。"""
    error = validate_city(city) or validate_month(month)
    if error:
        return error
    target_month = int(month[:-1])
    climate = MOCK_WEATHER.get((city, month))
    if target_month != date.today().month:
        return (f"{city}{month}模拟气候参考: {climate}（非真实预报）" if climate else
                f"暂无{city}{month}的气候数据，无法提供该月份的真实预报。")
    # Short-term providers are used only for the current month.
    daily = fetch_weather_qweather(city)
    if daily:
        temps = [str(d["tempMin"]) + "—" + str(d["tempMax"]) + "°C" for d in daily]
        texts = [d["textDay"] for d in daily]
        attribution = ", ".join(str(value) for value in daily[0].get("attributions", []))
        return f"{city}近期天气预报（来源：和风天气；城市中心附近；非整月预报）:\n" + "\n".join(
            f"  {daily[i]['fxDate']}: {temps[i]}, {texts[i]}" for i in range(len(daily))
        ) + (f"\n数据归属: {attribution}" if attribution else "")

    # 2. wttr.in（无需 Key）
    wttr = fetch_weather_wttr(city)
    if wttr:
        return f"{city}当前天气（来源：wttr.in；非整月预报）: {wttr}"

    # 3. 回退：模拟数据
    key = (city, month)
    w = MOCK_WEATHER.get(key)
    return f"{city}{month}模拟气候参考: {w}（非真实预报）" if w else f"暂无{city}{month}天气数据。"


@mcp.tool()
def recommend_itinerary(city: str, days: int, preference: str = "", budget: int = 0) -> str:
    """根据预算和偏好生成行程建议。city:城市, days:天数, preference:偏好(历史/自然/美食/文化/娱乐/城市), budget:总预算(元)

返回按天分配的行程，每项含门票价格和游玩时长。"""
    err = validate_city(city)
    if err:
        return err
    return _generate_itinerary_text(city, days, budget, preference)


@mcp.tool()
def plan_budget_tiers(city: str, days: int, budget: int, preference: str = "") -> str:
    """生成穷游/标准/舒适三档行程方案。city:城市, days:天数, budget:总预算(元), preference:偏好"""
    err = validate_city(city) or validate_trip(days, budget)
    if err:
        return err

    tiers = [
        ("穷游版", 0.6, "青旅/民宿", "公交+步行", "简餐/小吃"),
        ("标准版", 1.0, "经济型酒店", "地铁+公交", "当地餐厅"),
        ("舒适版", 1.5, "舒适型酒店", "打车/包车", "特色餐厅"),
    ]

    results = [f"【{city} {days}日三档方案】预算{budget}元\n"]
    for label, ratio, hotel, transport, meal in tiers:
        tier_budget = int(budget * ratio)
        spots_text = _generate_itinerary_text(city, days, tier_budget, preference)
        warning = "，超出原预算" if tier_budget > budget else ""
        results.append(f"===== {label} (约{tier_budget}元{warning}) =====")
        results.append(f"住宿: {hotel}")
        results.append(f"交通: {transport}")
        results.append(f"餐饮: {meal}")
        results.append(spots_text + "\n")

    return "\n".join(results)


@mcp.tool()
def weather_adjust_schedule(city: str, month: str, spot_names: list[str], days: int = 1) -> str:
    """根据天气调整景点安排，雨天优先室内景点。city:城市, month:月份, spot_names:景点名列表, days:天数"""
    err = validate_city(city) or validate_trip(days) or validate_month(month)
    if err:
        return err

    climate = _get_climate(city, month)
    rainy = _is_rainy(climate)

    coord_map = build_coord_map(city)
    spots_data = {s["name"]: s for s in MOCK_SPOTS.get(city, [])}
    unknown = [name for name in spot_names if name not in spots_data]
    if unknown:
        return f"未知景点: {unknown}"
    if not spot_names:
        return "请至少提供一个景点。"
    if not climate:
        return "暂无目标月份的气候参考，无法判断是否需要雨天调整。"

    indoor = []
    outdoor = []
    for name in spot_names:
        s = spots_data.get(name)
        if s and _indoor_friendly(s["name"]):
            indoor.append(name)
        else:
            outdoor.append(name)

    lines = [f"【{city}{month}天气调整】模拟气候参考: {climate}（非真实预报）"]
    if rainy:
        lines.append("预计有雨，已按雨天室内(文化/美食/娱乐)优先安排：")
        optimized = indoor + outdoor[:max(1, days)]
        skipped = outdoor[max(1, days):]
    else:
        lines.append("天气良好，按原计划即可：")
        optimized = spot_names
        skipped = []

    # Keep indoor-first grouping when it rains; optimize within each group.
    ordered = (nearest_neighbor(indoor, coord_map) + nearest_neighbor(outdoor[:max(1, days)], coord_map)) if rainy else nearest_neighbor(optimized, coord_map)
    for i, s in enumerate(ordered):
        tag = "🏠室内" if s in indoor else "🌳室外"
        lines.append(f"  {i+1}. {s} {tag}")

    if skipped:
        lines.append(f"\n跳过(天气不适宜): {', '.join(skipped)}")
    return "\n".join(lines)


@mcp.tool()
def packing_list(city: str, days: int, weather_summary: str = "", activities: str = "") -> str:
    """根据目的地天气和活动生成行李清单。city:城市, days:天数, weather_summary:天气描述, activities:活动类型(逗号分隔，如爬山,城市,美食)"""
    error = validate_city(city) or validate_trip(days)
    if error:
        return error
    rainy = _is_rainy(weather_summary)
    ranges = re.findall(r"(?<![\d.])([+-]?\d+(?:\.\d+)?)\s*(?:[-—~至]\s*([+-]?\d+(?:\.\d+)?))?\s*(?:°?C|℃|度)", weather_summary, re.I)
    temperatures = [float(value) for pair in ranges for value in pair if value]
    hot = any(kw in weather_summary for kw in ("热", "晒")) or any(value >= 30 for value in temperatures)
    cold = any(kw in weather_summary for kw in ("冷", "寒")) or any(value <= 10 for value in temperatures)

    lines = [f"【{city} {days}日行李清单】"]

    # 衣物
    lines.append("\n[衣物]")
    base = days
    lines.append(f"  - 上衣 x{max(2, base)}")
    lines.append(f"  - 裤子 x{max(2, max(1, base-1))}")
    lines.append(f"  - 内衣裤 x{days+1}")
    if hot:
        lines.append("  - 防晒衣/帽子/墨镜")
        lines.append("  - 短袖短裤为主")
    if cold:
        lines.append("  - 外套/毛衣")
    if rainy:
        lines.append("  - 雨伞/雨衣")

    # 鞋
    lines.append("\n[鞋履]")
    has_hike = "爬山" in activities or "徒步" in activities or "自然" in activities
    if has_hike:
        lines.append("  - 登山鞋/运动鞋")
    else:
        lines.append("  - 舒适步行鞋")
    lines.append("  - 拖鞋（酒店用）")

    # 洗漱
    lines.append("\n[洗漱用品]")
    lines.append("  - 牙刷/牙膏/洗面奶")
    lines.append("  - 毛巾/浴巾")
    lines.append("  - 护肤品/防晒霜")

    # 电子
    lines.append("\n[电子产品]")
    lines.append("  - 手机/充电器/充电宝")
    lines.append("  - 耳机")
    if days > 2:
        lines.append("  - 相机(可选)")

    # 证件
    lines.append("\n[证件]")
    lines.append("  - 身份证")
    lines.append("  - 手机支付/少量现金")

    # 活动专属
    if activities:
        acts = [a.strip() for a in activities.split(",")]
        lines.append("\n[活动专属]")
        if "爬山" in acts or "徒步" in acts:
            lines.append("  - 登山杖/背包/水壶")
        if "海边" in acts or "海滩" in acts:
            lines.append("  - 泳衣/沙滩鞋/防晒")
        if "美食" in acts or "吃货" in acts:
            lines.append("  - 健胃消食片")
        if "博物馆" in acts or "文化" in acts:
            lines.append("  - 笔记本(记录用)")

    return "\n".join(lines)


@mcp.tool()
def get_food_recommendation(city: str) -> str:
    """查询目的地特色美食及参考价格。city: 城市名"""
    err = validate_city(city)
    if err:
        return err
    foods = MOCK_FOOD.get(city)
    if not foods:
        return f"暂无{city}的美食数据。"
    return f"{city}美食及参考价格（本地模拟数据）:\n" + "\n".join(f"  - {f}" for f in foods)


@mcp.tool()
def estimate_daily_cost(city: str) -> str:
    """估算城市每日基本消费（住宿+餐饮+交通）。city: 城市名"""
    err = validate_city(city)
    if err:
        return err
    estimates = {
        "北京": "住宿200-400元/晚, 餐饮80-150元/天, 交通20-50元/天",
        "上海": "住宿250-500元/晚, 餐饮100-200元/天, 交通20-50元/天",
        "西安": "住宿150-300元/晚, 餐饮60-120元/天, 交通15-30元/天",
        "成都": "住宿150-350元/晚, 餐饮70-150元/天, 交通15-30元/天",
        "广州": "住宿200-400元/晚, 餐饮80-150元/天, 交通20-40元/天",
    }
    estimate = estimates.get(city)
    return f"{estimate}（本地模拟估算，不含门票和城际交通）" if estimate else f"暂无{city}的消费估算数据。请根据实际情况调整。"


if __name__ == "__main__":
    has_api = "已配置" if WEATHER_API_KEY and WEATHER_API_HOST else "未完整配置（wttr.in 或模拟数据兜底）"
    print(f"[天气API] {has_api}", file=sys.stderr)
    mcp.run(transport="stdio")
