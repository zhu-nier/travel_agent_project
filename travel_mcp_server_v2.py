"""旅游助手 MCP Server V2：真实 API + 模拟数据双模式"""
import os
import json
from dotenv import load_dotenv
load_dotenv()
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("旅游数据服务 V2")

# ====== API 密钥 ======
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")


def fetch_weather_wttr(city: str) -> str:
    """通过 wttr.in 查询天气（无需 API Key，全球城市）"""
    import httpx
    try:
        resp = httpx.get(
            f"https://wttr.in/{city}?format=%C+%t&lang=zh",
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.text.strip()
    except Exception:
        pass
    return ""


def fetch_weather_qweather(city: str) -> list | None:
    """通过和风天气 API 查询（需要 WEATHER_API_KEY）"""
    if not WEATHER_API_KEY:
        return None
    import httpx
    try:
        resp = httpx.get(
            "https://devapi.qweather.com/v7/weather/3d",
            params={"location": city, "key": WEATHER_API_KEY},
            timeout=10,
        )
        data = resp.json()
        if data.get("code") == "200":
            return data["daily"]
    except Exception:
        pass
    return None

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

VALID_CITIES = set(MOCK_SPOTS.keys()) | {"重庆", "广州", "南京", "杭州", "苏州"}  # 支持的额外城市


def validate_city(city: str) -> str | None:
    """返回 None 表示城市有效，否则返回错误提示"""
    if city in VALID_CITIES:
        return None
    return f"不支持的城市「{city}」，可选: {', '.join(sorted(VALID_CITIES))}"


def haversine(lat1, lng1, lat2, lng2):
    """计算两点间球面距离（公里）"""
    import math
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def build_coord_map(city):
    """根据城市名构建 {景点名: (lat, lng)} 字典"""
    spots = MOCK_SPOTS.get(city, [])
    return {s["name"]: (s["lat"], s["lng"]) for s in spots if "lat" in s and "lng" in s}


def nearest_neighbor(spots_list, coord_map):
    """贪心最近邻算法优化游览顺序。spots_list: 景点名列表, coord_map: {名: (lat,lng)}"""
    if not spots_list or len(spots_list) <= 1:
        return spots_list

    remaining = set(spots_list)
    # 以第一个景点为起点
    current = spots_list[0]
    remaining.discard(current)
    ordered = [current]

    while remaining:
        cur_coord = coord_map.get(current)
        if not cur_coord:
            current = remaining.pop()
            ordered.append(current)
            continue
        # 找最近的未访问景点
        nearest = min(remaining, key=lambda s: haversine(*cur_coord, *coord_map.get(s, (0, 0))))
        ordered.append(nearest)
        remaining.discard(nearest)
        current = nearest

    return ordered


def _indoor_friendly(spot_type):
    """判断景点类型是否适合雨天室内活动"""
    return spot_type in ("文化", "美食", "娱乐", "历史")


def _get_climate(city, month):
    """获取城市月份的气候描述"""
    key = (city, month)
    w = MOCK_WEATHER.get(key, "")
    if w:
        return w
    wttr = fetch_weather_wttr(city)
    return wttr if wttr else ""


def _is_rainy(climate):
    return any(kw in climate for kw in ["雨", "湿", "雷", "潮"])


def _generate_itinerary_text(city, days, budget, preference):
    """核心行程生成逻辑（被 recommend_itinerary 和 plan_budget_tiers 共用）"""
    spots = MOCK_SPOTS.get(city, [])
    if not spots:
        return f"暂无{city}的景点数据。"

    if preference:
        filtered = [s for s in spots if preference in s["type"]]
        if filtered:
            spots = filtered

    if budget > 0:
        per_day = budget // days
        spots = [s for s in spots if s["price"] <= per_day * 0.6]

    if not spots:
        return f"{city}没有符合「{preference}」偏好且在预算内的景点。"

    lines = [f"【{city} {days}日行程推荐】预算约{budget}元"]
    per_day = max(1, len(spots) // days)
    for d in range(days):
        lines.append(f"\n第{d+1}天:")
        day_spots = spots[d * per_day: (d + 1) * per_day]
        for s in day_spots:
            lines.append(f"  - {s['name']} ({s['type']}, 门票{s['price']}元, 建议{s['hours']}小时)")

    total = sum(s["price"] for s in spots)
    lines.append(f"\n景点门票合计: {total}元")
    lines.append(f"剩余预算(食宿交通): {budget - max(0, total)}元")
    return "\n".join(lines)
def search_web(query: str) -> str:
    """搜索网络获取实时信息。query: 搜索关键词"""
    if not TAVILY_API_KEY:
        return "未配置 TAVILY_API_KEY，无法搜索。"
    try:
        from tavily import TavilyClient
        c = TavilyClient(api_key=TAVILY_API_KEY)
        r = c.search(query=query, search_depth="advanced", max_results=5)
        snippets = [f"- {x['title']}: {x['content']}" for x in r.get("results", [])]
        return "搜索结果:\n" + "\n".join(snippets) if snippets else "无结果。"
    except Exception as e:
        return f"搜索失败: {e}"


@mcp.tool()
def optimize_route(city: str, spot_names: list[str], start_lat: float = 0, start_lng: float = 0) -> str:
    """用最近邻算法优化景点游览顺序，减少走回头路。city: 城市名, spot_names: 景点名称列表, start_lat/start_lng: 起点坐标(可选，默认第一个景点)"""
    err = validate_city(city)
    if err:
        return err
    coord_map = build_coord_map(city)
    if not coord_map:
        return f"暂无{city}的坐标数据。"
    unknown = [s for s in spot_names if s not in coord_map]
    if unknown:
        return f"未知景点: {unknown}"

    ordered = nearest_neighbor(spot_names, coord_map)
    lines = [f"【{city}路线优化】按游览顺序:"]
    dists = []
    prev_coord = (start_lat, start_lng) if start_lat and start_lng else coord_map[ordered[0]]
    for s in ordered:
        cur = coord_map[s]
        d = haversine(*prev_coord, *cur)
        dists.append(d)
        lines.append(f"  {len(lines)}. {s} (距上一点 {d:.1f}km)")
        prev_coord = cur
    total = sum(dists[1:])  # 跳过起点到第一点的距离
    lines.append(f"\n总路程约 {total:.1f}km")
    return "\n".join(lines)


@mcp.tool()
def find_events(city: str, month: str, year: int = 2026) -> str:
    """搜索目的地当月演唱会/展览/节日活动。city: 城市名, month: 月份(如6月), year: 年份"""
    if not TAVILY_API_KEY:
        return "未配置 TAVILY_API_KEY，无法搜索活动。"
    try:
        from tavily import TavilyClient
        c = TavilyClient(api_key=TAVILY_API_KEY)
        query = f"{year}年{month} {city} 演唱会 展览 活动 演出"
        r = c.search(query=query, search_depth="advanced", max_results=8)
        snippets = [f"- {x['title']}: {x['content'][:200]}" for x in r.get("results", []) if x.get("content")]
        return f"【{city}{year}年{month}活动】\n" + "\n".join(snippets) if snippets else f"未找到{city}{month}的活动信息。"
    except Exception as e:
        return f"搜索活动失败: {e}"


@mcp.tool()
def get_weather(city: str, month: str) -> str:
    """查询城市天气。city: 城市名, month: 月份(如6月)

优先用真实 API（和风天气需要 WEATHER_API_KEY），其次 wttr.in（无需 Key），最后回退模拟数据。"""
    # 1. 和风天气（需 API Key）
    daily = fetch_weather_qweather(city)
    if daily:
        temps = [d["tempMax"] + "-" + d["tempMin"] + "C" for d in daily]
        texts = [d["textDay"] for d in daily]
        return f"{city}{month}天气预报:\n" + "\n".join(
            f"  第{i+1}天: {temps[i]}, {texts[i]}" for i in range(len(daily))
        )

    # 2. wttr.in（无需 Key）
    wttr = fetch_weather_wttr(city)
    if wttr:
        return f"{city}当前天气: {wttr}"

    # 3. 回退：模拟数据
    key = (city, month)
    w = MOCK_WEATHER.get(key)
    return f"{city}{month}气候: {w}" if w else f"暂无{city}{month}天气数据。"


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
    err = validate_city(city)
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
        results.append(f"===== {label} (约{tier_budget}元) =====")
        results.append(f"住宿: {hotel}")
        results.append(f"交通: {transport}")
        results.append(f"餐饮: {meal}")
        results.append(spots_text + "\n")

    return "\n".join(results)


@mcp.tool()
def weather_adjust_schedule(city: str, month: str, spot_names: list[str], days: int = 1) -> str:
    """根据天气调整景点安排，雨天优先室内景点。city:城市, month:月份, spot_names:景点名列表, days:天数"""
    err = validate_city(city)
    if err:
        return err

    climate = _get_climate(city, month)
    rainy = _is_rainy(climate)

    coord_map = build_coord_map(city)
    spots_data = {s["name"]: s for s in MOCK_SPOTS.get(city, [])}

    indoor = []
    outdoor = []
    for name in spot_names:
        s = spots_data.get(name)
        if s and _indoor_friendly(s["type"]):
            indoor.append(name)
        else:
            outdoor.append(name)

    lines = [f"【{city}{month}天气调整】{climate}"]
    if rainy:
        lines.append("预计有雨，已按雨天室内(文化/美食/娱乐)优先安排：")
        optimized = indoor + outdoor[:max(1, days)]
        skipped = outdoor[max(1, days):]
    else:
        lines.append("天气良好，按原计划即可：")
        optimized = spot_names
        skipped = []

    ordered = nearest_neighbor(optimized, coord_map) if coord_map else optimized
    for i, s in enumerate(ordered):
        tag = "🏠室内" if s in indoor else "🌳室外"
        lines.append(f"  {i+1}. {s} {tag}")

    if skipped:
        lines.append(f"\n跳过(天气不适宜): {', '.join(skipped)}")
    return "\n".join(lines)


@mcp.tool()
def packing_list(city: str, days: int, weather_summary: str = "", activities: str = "") -> str:
    """根据目的地天气和活动生成行李清单。city:城市, days:天数, weather_summary:天气描述, activities:活动类型(逗号分隔，如爬山,城市,美食)"""
    rainy = _is_rainy(weather_summary)
    hot = any(kw in weather_summary for kw in ["热", "晒", "30"])
    cold = any(kw in weather_summary for kw in ["冷", "寒", "5"])

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
    return f"{city}必尝美食及参考价格:\n" + "\n".join(f"  - {f}" for f in foods)


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
    return estimates.get(city, f"暂无{city}的消费估算数据。请根据实际情况调整。")


if __name__ == "__main__":
    has_api = "已配置" if WEATHER_API_KEY else "未配置(使用模拟数据)"
    print(f"[天气API] {has_api}")
    mcp.run(transport="stdio")
