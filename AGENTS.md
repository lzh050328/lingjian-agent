# AGENTS.md - lingjian-agent (灵鉴智能体服务) 项目指南

## 技术栈

- **Python 3.13** + **FastAPI 0.135.2** + **Uvicorn 0.42.0**
- **LangChain 1.3.6** + **LangGraph 1.2.4** + **LangChain OpenAI 1.3.0**
- **Nacos** 服务注册与配置中心、**Redis** 会话管理、**MongoDB** 数据存储、**RabbitMQ** 消息队列
- **PyTorch 2.6.0** + **Transformers 5.4.0** + **XGBoost 3.2.0** + **Scikit-learn 1.8.0**
- **MCP (Model Context Protocol)** 集成高德地图服务
- 环境管理：Conda (`environment.yml`)

## 项目结构

```
lingjian-agent/
├── main.py                          # 项目入口，启动 FastAPI + Uvicorn 服务
├── application.yml                  # 应用配置文件（YAML 格式）
├── environment.yml                  # Conda 环境依赖配置
├── .env.example                     # 环境变量模板
│
├── lingjian_agent/                           # 智能体核心模块
│   ├── __init__.py
│   ├── BaseAgent.py                 # 智能体基础抽象类（SSE 流式输出、Redis 停止标记、LLM 初始化）
│   ├── Agents.py                    # 全局智能体注册表（agentId -> 实例映射）
│   │
│   ├── risk/                        # 风控智能体
│   │   ├── RiskControlAgent.py      # 登录风控智能体 (ID: 1001)
│   │   └── RefreshTokenRiskControlAgent.py  # Token 刷新风控智能体 (ID: 1002)
│   │
│   ├── carriage/                    # 运费智能体
│   │   └── ComputeCarriageAgent.py  # 运费计算智能体 (ID: 1003)
│   │
│   ├── categorie/                   # 品类预测智能体
│   │   ├── ParcelItemCheckAgent.py  # 包裹物品校验智能体 (ID: 1004)
│   │   ├── CategoryPredictor.py     # BERT 品类预测器
│   │   ├── predict.py               # 预测入口
│   │   ├── train_bert.py            # BERT 模型训练脚本
│   │   ├── data/                    # 训练数据
│   │   ├── pretrained_models/       # 预训练模型
│   │   └── saved_model/             # 保存的模型
│   │
│   ├── transport/                   # 运输智能体
│   │   ├── OrganTypeAgent.py        # 网点类型智能体 (ID: 1005)
│   │   └── CostFactorAgent.py       # 成本因子智能体 (ID: 1006)
│   │
│   ├── service_scope/               # 服务范围智能体
│   │   ├── ServiceScopeAgent.py     # 服务范围推荐智能体 (ID: 1007)
│   │   └── graph/                   # LangGraph 工作流
│   │       ├── RecommendationGraphBuilder.py  # 图构建器
│   │       ├── GraphState.py        # 图状态定义
│   │       ├── GraphRouters.py      # 图路由
│   │       ├── nodes/               # 图节点
│   │       └── clients/             # 外部客户端
│   │
│   ├── dispatch/                    # 调度智能体
│   │   ├── courier/                 # 快递员调度
│   │   │   ├── CourierDispatchAgent.py  # 快递员调度智能体 (ID: 1008)
│   │   │   ├── data/                # 数据
│   │   │   ├── model/               # 模型
│   │   │   └── tools/               # 工具
│   │   ├── transport_order/         # 运输订单调度
│   │   │   ├── TransportOrderDispatchAgent.py  # 运输订单调度智能体 (ID: 1009)
│   │   │   ├── data/                # 数据
│   │   │   └── model/               # 模型
│   │   └── transport_task/          # 运输任务调度
│   │       ├── TransportTaskDispatchAgent.py  # 运输任务调度智能体 (ID: 1010)
│   │       ├── consumer/            # 消息消费者
│   │       │   └── dispatch_train_consumer.py # 调度训练消息消费者
│   │       ├── model/               # 数据模型
│   │       │   └── dispatch_decision_record.py
│   │       ├── models/              # 机器学习模型文件
│   │       ├── repository/          # 数据仓储
│   │       ├── service/             # 业务服务
│   │       │   ├── DispatchTrainService.py
│   │       │   ├── ModelManager.py
│   │       │   └── PredictService.py
│   │       └── train/               # 训练相关
│   │
│   ├── mcp/                         # MCP 服务集成
│   │   └── MCPService.py            # MCP 服务管理器（高德地图）
│   │
│   └── prompts/                     # Prompt 配置
│       └── SystemPromptConfig.py    # 系统提示词配置
│
├── web/                             # Web 接口模块
│   ├── __init__.py
│   ├── WebApp.py                    # FastAPI 应用实例（生命周期管理、Nacos 注册、路由注册）
│   └── routers/
│       ├── HelloRouter.py           # 健康检查/测试路由
│       └── ChatRouter.py            # 智能体对话路由（流式/非流式/停止）
│
├── config/                          # 配置模块
│   ├── __init__.py
│   ├── ConfigManager.py             # 配置管理器（YAML 加载、环境变量替换）
│   ├── Logger.py                    # 日志配置
│   ├── NacosConfig.py               # Nacos 配置与客户端
│   └── RedisConfig.py               # Redis 配置与客户端
│
├── common/                          # 公共常量模块
│   ├── __init__.py
│   └── Common.py                    # 配置常量定义（Server、Nacos、AI、Prompt）
│
├── util/                            # 工具模块
│   ├── __init__.py
│   ├── HttpClientUtil.py            # HTTP 客户端工具
│   ├── JsonUtil.py                  # JSON 序列化工具
│   ├── NacosServiceSelector.py      # Nacos 服务选择器
│   ├── ProjectRoot.py               # 项目根路径工具
│   └── YamlLoader.py                # YAML 加载器
│
├── tools/                           # 业务工具模块
│   ├── __init__.py
│   ├── AreaTools.py                 # 区域工具
│   ├── CarriageTools.py             # 运费工具
│   ├── ServiceScopeTools.py         # 服务范围工具
│   └── TransportTools.py            # 运输工具
│
├── models/                          # 本地模型文件
│   └── bert-base-chinese/           # BERT 中文预训练模型
│
└── logs/                            # 日志目录
```

## 智能体列表

| Agent ID | 名称 | 类 | 说明 | 流式输出 |
|----------|------|-----|------|----------|
| 1001 | 风控智能体 | RiskControlAgent | 登录风险控制 | 否 |
| 1002 | Token 刷新风控智能体 | RefreshTokenRiskControlAgent | Token 刷新风险控制 | 否 |
| 1003 | 运费计算智能体 | ComputeCarriageAgent | 运费智能计算 | 否 |
| 1004 | 包裹物品校验智能体 | ParcelItemCheckAgent | 基于 BERT 的包裹品类预测 | 否 |
| 1005 | 网点类型智能体 | OrganTypeAgent | 网点类型识别 | 否 |
| 1006 | 成本因子智能体 | CostFactorAgent | 运输成本因子分析 | 否 |
| 1007 | 服务范围智能体 | ServiceScopeAgent | 服务范围推荐（LangGraph） | 否 |
| 1008 | 快递员调度智能体 | CourierDispatchAgent | 快递员任务调度 | 否 |
| 1009 | 运输订单调度智能体 | TransportOrderDispatchAgent | 运输订单智能调度 | 否 |
| 1010 | 运输任务调度智能体 | TransportTaskDispatchAgent | 运输任务智能调度 | 否 |

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/chat` | 非流式对话接口 |
| POST | `/chat/stream` | 流式对话接口（SSE） |
| POST | `/chat/stop?session_id=&agent_id=` | 停止指定会话 |
| GET/POST | `/hello` | 健康检查/测试接口 |

## 配置说明

### application.yml 主要配置项

| 配置项 | 说明 |
|--------|------|
| `server.port` | 服务端口（默认 28001） |
| `server.host` | 监听地址（默认 0.0.0.0） |
| `server.logger.level` | 日志级别 |
| `nacos.*` | Nacos 注册与配置中心 |
| `ai.openai.*` | OpenAI 兼容模型配置 |
| `ai.dispatch.*` | AI 调度模型配置（mock/real 模式） |
| `prompt.*` | 各智能体系统提示词配置 |
| `redis.*` | Redis 连接配置 |
| `rabbitmq.*` | RabbitMQ 消息队列配置 |
| `mongo.*` | MongoDB 数据库配置 |
| `cc.services.*` | 业务系统微服务名称映射 |
| `mcp-servers` | MCP 服务配置（高德地图） |

### 环境变量 (.env)

| 变量 | 说明 |
|------|------|
| `AMAP_MCP_URL` | 高德 MCP 服务地址 |
| `AI_OPENAI_MODEL` | AI 模型名称 |
| `AI_OPENAI_API_KEY` | AI API Key |
| `AI_OPENAI_BASE_URL` | AI Base URL |
| `AI_OPENAI_TEMPERATURE` | Temperature 参数 |
| `AI_OPENAI_TIMEOUT` | 请求超时时间 |

## 常用命令

```bash
# 激活 Conda 环境
conda env create -f environment.yml
conda activate cc_env_31313

# 启动服务
python main.py

# 安装/更新依赖
pip install -r requirements.txt  # 如果有 requirements.txt
```

## 架构说明

### BaseAgent 抽象类

所有智能体继承自 `BaseAgent`，提供以下能力：
- `execute()` - 流式执行接口，返回 `AsyncIterable[str]`（SSE 格式）
- `execute_one_shot()` - 非流式执行接口
- `session_detail()` - 获取会话详情
- `delete_session()` - 删除会话数据
- `stop()/reset_stop()/is_stop()` - Redis 停止标记管理
- `init_llm()` - 初始化 LLM（统一走 OpenAI Compatible 接口）

### SSE 流式输出

智能体通过 SSE (Server-Sent Events) 协议进行流式输出：
- 每条消息格式：`data: {"eventType": <类型>, "eventData": <数据>}\n\n`
- `eventType: 1002` 表示停止事件
- 前端可通过 `/chat/stop` 接口发送停止信号，通过 Redis 标记中断输出

### Nacos 服务注册

服务启动时自动向 Nacos 注册，关闭时自动注销：
- 服务名：`lingjian-agent`
- 心跳间隔：10 秒
- 配置命名空间：`<NACOS_CONFIG_NAMESPACE>`
- 注册命名空间：`<NACOS_DISCOVERY_NAMESPACE>`

### RabbitMQ 消息消费

`dispatch_train_consumer` 在启动时开始消费调度训练消息：
- Exchange: `cc.exchange.topic.aigc`
- Queue: `cc.queue.aigc.dispatch.train`
- RoutingKey: `CREATE` / `COMPLETE`

### MongoDB 存储

用于存储调度决策记录：
- Database: `cc_ai`
- Collections: `dispatch_decision_record`, `dispatch_decision_record_mock`

## AI 编码约束

- **environment.yml 修改需确认**：禁止随意修改环境依赖文件。如果确实需要新增/修改/删除依赖，必须经过人工确认后方可执行。
- **禁止随意删除代码**：不删除现有代码（除非有明确指令），优先在现有代码基础上做增量修改。
- **最小影响范围**：尽量以最小改动完成需求，不重构无关代码，不改动与任务无关的文件。
- **保留已有注释**：不对现有注释进行删改。如需补充注释，在已有注释基础上增量添加。
- **智能体注册**：新增智能体需要在 `lingjian_agent/Agents.py` 中注册，并在 `application.yml` 中配置对应的 prompt。
- **Agent ID 规范**：新增智能体 ID 需从 1011 开始递增，避免冲突。
