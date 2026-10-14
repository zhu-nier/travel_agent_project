"""Extract required fields only from user messages, never assistant/tool text."""
import re

from travel_config import MAX_TRAVEL_DAYS, SUPPORTED_CITIES

CITY_PATTERN = re.compile("|".join(SUPPORTED_CITIES + ("深圳", "武汉", "天津", "长沙", "厦门", "青岛", "大连", "纽约", "东京", "香港", "台北", "巴黎", "伦敦")))
NUMBER = r"-?\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千万]+"
DAYS_PATTERN = re.compile(rf"(?<![\d.])({NUMBER})\s*(?:天|日)(?!元)")
MONTH_PATTERN = re.compile(r"(?<!\d)(\d+|[一二三四五六七八九十]+)\s*月")
BUDGET_PATTERN = re.compile(
    rf"预算\s*(?:是|为|大约|约|大概|只有|有|:|：)?\s*({NUMBER})\s*(万|千|百)?"
    rf"|(?<![\d.])({NUMBER})\s*(万|千|百)?\s*(?:元|块)"
)
DIGITS = dict(zip("零〇一二两三四五六七八九", (0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9)))


def chinese_number(text):
    if re.fullmatch(r"-?\d+(?:\.\d+)?", text):
        return float(text)
    total = section = digit = 0
    for char in text:
        if char in DIGITS:
            digit = DIGITS[char]
        elif char == "万":
            total += (section + digit) * 10000
            section = digit = 0
        else:
            section += (digit or 1) * {"十": 10, "百": 100, "千": 1000}[char]
            digit = 0
    return float(total + section + digit)


def message_text(message):
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(block if isinstance(block, str) else block.get("text", "")
                         for block in content if isinstance(block, (str, dict)))
    return ""


def extract_request(messages):
    fields = {"城市": None, "月份": None, "天数": None, "预算": None}
    for message in messages:
        if getattr(message, "type", None) != "human":
            continue
        text = message_text(message)
        cities = list(CITY_PATTERN.finditer(text))
        if cities:
            fields["城市"] = cities[-1].group()
        months = list(MONTH_PATTERN.finditer(text))
        if months:
            value = chinese_number(months[-1].group(1))
            fields["月份"] = f"{int(value)}月" if 1 <= value <= 12 else None
        elif re.search(r"下个月|这个月|本月", text):
            from datetime import date
            value = date.today().month
            if "下个月" in text:
                value = value % 12 + 1
            fields["月份"] = f"{value}月"
        days = list(DAYS_PATTERN.finditer(text))
        if days:
            value = chinese_number(days[-1].group(1))
            fields["天数"] = int(value) if value.is_integer() and 1 <= value <= MAX_TRAVEL_DAYS else None
        budgets = list(BUDGET_PATTERN.finditer(text))
        if budgets:
            match = budgets[-1]
            value = chinese_number(match.group(1) or match.group(3))
            value *= {None: 1, "百": 100, "千": 1000, "万": 10000}[match.group(2) or match.group(4)]
            fields["预算"] = int(value) if value.is_integer() and value > 0 else None
        # A bare number is unambiguous only when budget is the sole missing field.
        elif all(fields[key] is not None for key in ("城市", "月份", "天数")) and fields["预算"] is None:
            if re.fullmatch(r"\d+", text.strip()) and int(text) > 0:
                fields["预算"] = int(text)
    return fields


def check_missing_info(state):
    messages = state["messages"] if isinstance(state, dict) else state
    fields = extract_request(messages)
    return [field for field, value in fields.items()
            if value is None or (field == "城市" and value not in SUPPORTED_CITIES)]


def missing_question(missing):
    prompts = {"城市": "目的地城市", "月份": "出行月份（1—12月）", "天数": f"旅行天数（1—{MAX_TRAVEL_DAYS}天）", "预算": "总预算（具体金额，元）"}
    city_hint = f"支持的城市：{'、'.join(SUPPORTED_CITIES)}。" if "城市" in missing else ""
    return f"请补充{'、'.join(prompts[field] for field in missing)}。{city_hint}"
