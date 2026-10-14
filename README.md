# 旅游助手 V2

基于 LangChain、LangGraph 和 MCP 的中文旅游规划助手，支持命令行对话与 FastAPI / WebSocket 网页界面。

## 功能

- 收集目的地、月份、天数、预算后生成旅游方案。
- 查询天气与活动，规划预算档位、景点路线和雨天安排。
- 提供美食推荐、消费估算与行李清单。
- 天气和搜索接口结合本地模拟数据；结果可能使用兜底数据，请核对实际价格与营业信息。

## 安装

使用 Python 3.11 或更高版本（推荐 3.11 / 3.12），在项目目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

在项目目录的 `.env` 中填写 `DEEPSEEK_API_KEY`。`TAVILY_API_KEY` 是可选的搜索配置；和风天气需同时填写 `WEATHER_API_KEY` 和控制台分配的 `WEATHER_API_HOST`（只填域名，不含 `https://`）。未配置时使用 wttr.in 或本地模拟数据兜底。真实密钥保存在本机 `.env` 中，该文件已被 Git 忽略。

直接依赖已固定为本次验证使用的版本；建议在独立虚拟环境中安装。`TRAVEL_MODEL` 可修改模型名；`TRAVEL_OFFLINE=1` 禁用工具的联网查询，但对话模型仍需密钥和网络。

## 运行

网页界面：

```powershell
python travel_web_server.py
```

打开 <http://127.0.0.1:8000>。

命令行：

```powershell
python travel_agent_v2.py
```

输入示例：`我想在6月去北京玩3天，预算3000元，喜欢历史和美食。`
输入 `exit` 或 `quit` 退出，输入 `reset` 开始新对话。网页也提供「新对话」按钮。

必填信息只从用户消息提取，支持多轮补充、中文月份/天数/金额，以及修改目的地或预算。信息不完整时直接追问，不调用模型或旅游工具。该提取器是规则实现，复杂表达可能需要进一步补充。

支持城市：北京、上海、西安、成都、重庆、广州、南京、杭州、苏州。目前景点和坐标数据只覆盖北京、上海、西安；其他城市的天气、餐饮和消费信息覆盖程度不同，缺失时会明确提示。

每条消息最多 4000 字，旅行天数为 1—30 天。单次工具调用限时 30 秒，每轮对话限时 180 秒，最多 6 轮工具查询。处理失败会提示并开始新对话；断线重连后也需重新提供旅行需求。

## 数据与算法

- 门票、食宿预算和景点数据属于本地模拟参考，不保证实时准确；舒适档超出原预算会明确标注。
- 行程按游览时长分配，所有入选景点只安排一次；没有数据的日程显示自由活动，超过 8 小时会提示调整。
- 路线使用确定性的最近邻算法，计入指定起点；距离是球面直线距离，不能替代道路导航。
- 目标月份不是当前月份时仅提供模拟气候参考，不将当前天气当作远期预报。当前月份的查询提供有日期的近期预报或当前天气，并注明范围。
- 天气成功响应缓存 10 分钟；网络搜索结果附来源链接。
- 和风天气使用专属 API Host、城市中心附近坐标和请求头认证，参考 [官方每日预报接口](https://dev.qweather.com/en/docs/api/weather/weather-daily-forecast/) 与 [API Host 配置](https://dev.qweather.com/en/docs/configuration/api-host/)。

## 评估

```powershell
python travel_evaluation.py
```

评估脚本调用真实模型和工具，需要配置密钥与网络，可能产生 API 费用。

无需密钥的离线回归测试（包括模拟模型、WebSocket 交互及本机 MCP 子进程）：

```powershell
python -m unittest discover -s tests -v
node tests/test_frontend.cjs
```

第二条需要 Node.js 18 或更高版本。GitHub Actions 配置会在 Python 3.11 / 3.12 上运行这些检查，不调用付费 API。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `travel_agent_v2.py` | 命令行对话入口 |
| `travel_mcp_server_v2.py` | MCP 旅游工具与数据服务 |
| `travel_web_server.py` | FastAPI / WebSocket 服务 |
| `travel_evaluation.py` | 场景评估脚本 |
| `travel_core.py` | CLI、网页与评估共享的异步 Agent 工作流 |
| `travel_request.py` | 用户需求提取与完整性检查 |
| `travel_config.py` | 项目配置与 MCP 进程参数 |
| `static/index.html` | 网页界面 |
| `tests/` | 无需密钥的回归测试 |

模型密钥和联网接口配置完成后才能验证真实模型的完整对话流程。网页服务默认只监听本机，同源 WebSocket 校验和消息校验已启用；公网部署仍需另行配置登录验证和代理。
