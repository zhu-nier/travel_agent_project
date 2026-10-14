# 旅游助手 V2

基于 LangChain、LangGraph 和 MCP 的中文旅游规划助手，支持命令行对话与 FastAPI / WebSocket 网页界面。

## 功能

- 收集目的地、月份、天数、预算后生成旅游方案。
- 查询天气与活动，规划预算档位、景点路线和雨天安排。
- 提供美食推荐、消费估算与行李清单。
- 天气和搜索接口结合本地模拟数据；结果可能使用兜底数据，请核对实际价格与营业信息。

## 安装

使用 Python 3.10 或更高版本，在项目目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

在 `.env` 中填写 `DEEPSEEK_API_KEY`。`TAVILY_API_KEY` 与 `WEATHER_API_KEY` 是可选的搜索、天气配置。真实密钥保存在本机 `.env` 中，该文件已被 Git 忽略。

依赖清单未锁定版本；建议在独立虚拟环境中安装。

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
输入 `exit` 或 `quit` 退出。

## 评估

```powershell
python travel_evaluation.py
```

评估脚本调用真实模型和工具，需要配置密钥与网络，可能产生 API 费用。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `travel_agent_v2.py` | 命令行对话入口 |
| `travel_mcp_server_v2.py` | MCP 旅游工具与数据服务 |
| `travel_web_server.py` | FastAPI / WebSocket 服务 |
| `travel_evaluation.py` | 场景评估脚本 |
| `static/index.html` | 网页界面 |

模型密钥和联网接口配置完成后才能验证完整对话流程。网页服务默认只监听本机。
