# lingjian-agent (灵鉴 · 智能体服务)

> 基于 Python + LangChain + LangGraph 的 AI 智能体服务，为物流平台提供智能决策与风控能力。

## 一、项目简介

**lingjian-agent** 是物流平台的 Python AI 服务层，作为 Java 微服务系统与 LLM 之间的桥梁，提供 10 个智能体的统一管理与调度。

**整体架构：** `Java 微服务 → Python AI 服务 → LLM / 机器学习模型`

**技术栈：** `Python 3.13` · `FastAPI` · `LangChain` · `LangGraph` · `XGBoost` · `BERT` · `Redis` · `MongoDB` · `RabbitMQ` · `Nacos` · `MCP`

### 核心特性
- 🤖 **10 个 AI 智能体**：覆盖风控、调度、识别、计算等核心业务场景
- 🔄 **SSE 流式输出**：支持 Server-Sent Events 实时推送，可中断控制
- 📊 **机器学习集成**：XGBoost 调度预测、BERT 物品分类
- 🗺️ **MCP 服务集成**：高德地图 MCP 服务接入
- 📈 **模型持续训练**：RabbitMQ 消息驱动，MongoDB 数据存储
- 🔌 **Nacos 服务注册**：自动注册发现，心跳健康检查

---

## 二、智能体列表

| Agent ID | 名称 | 类 | 说明 |
|----------|------|-----|------|
| 1001 | 风控智能体 | RiskControlAgent | 登录风险控制 |
| 1002 | Token 刷新风控智能体 | RefreshTokenRiskControlAgent | Token 刷新风险控制 |
| 1003 | 运费计算智能体 | ComputeCarriageAgent | 运费智能计算 |
| 1004 | 包裹物品校验智能体 | ParcelItemCheckAgent | 基于 BERT 的包裹品类预测 |
| 1005 | 网点类型智能体 | OrganTypeAgent | 网点类型识别 |
| 1006 | 成本因子智能体 | CostFactorAgent | 运输成本因子分析 |
| 1007 | 服务范围智能体 | ServiceScopeAgent | 服务范围推荐（LangGraph） |
| 1008 | 快递员调度智能体 | CourierDispatchAgent | 快递员任务调度 |
| 1009 | 运输订单调度智能体 | TransportOrderDispatchAgent | 运输订单智能调度 |
| 1010 | 运输任务调度智能体 | TransportTaskDispatchAgent | 运输任务智能调度 |

---

## 三、系统架构

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Java 微服务集群                            │
│              (chain-cloud-ms-aigc-service 等)                 │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP / Feign
┌──────────────────────────▼──────────────────────────────────┐
│              Python AI 服务 (lingjian-agent)        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                  FastAPI (28001)                       │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐ │  │
│  │  │风控智能体│  │调度智能体│  │识别智能体│  │计算智能体│ │  │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘ │  │
│  └───────────────────────────────────────────────────────┘  │
└──────┬──────────┬──────────┬──────────┬──────────┬──────────┘
       │          │          │          │          │
   ┌───▼───┐  ┌──▼───┐  ┌──▼───┐  ┌──▼───┐  ┌──▼──────┐
   │ Redis │  │Mongo │  │Rabbit│  │Nacos │  │  LLM    │
   │ 会话  │  │  DB  │  │  MQ  │  │注册  │  │ API     │
   │ 停止  │  │决策  │  │训练  │  │配置  │  │         │
   └───────┘  └──────┘  └──────┘  └──────┘  └─────────┘
```

### 3.2 模块结构

```
lingjian-agent/
├── main.py                          # 项目入口
├── application.yml                  # 应用配置
├── environment.yml                  # Conda 环境依赖
├── .env.example                     # 环境变量模板
│
├── lingjian_agent/                           # 智能体核心模块
│   ├── BaseAgent.py                 # 智能体基础抽象类
│   ├── Agents.py                    # 全局智能体注册表
│   ├── risk/                        # 风控智能体 (1001, 1002)
│   ├── carriage/                    # 运费智能体 (1003)
│   ├── categorie/                   # 品类预测智能体 (1004)
│   ├── transport/                   # 运输智能体 (1005, 1006)
│   ├── service_scope/               # 服务范围智能体 (1007)
│   ├── dispatch/                    # 调度智能体 (1008, 1009, 1010)
│   ├── mcp/                         # MCP 服务集成
│   └── prompts/                     # Prompt 配置
│
├── web/                             # Web 接口模块
│   ├── WebApp.py                    # FastAPI 应用实例
│   └── routers/
│       ├── HelloRouter.py           # 健康检查
│       └── ChatRouter.py            # 对话路由
│
├── config/                          # 配置模块
│   ├── ConfigManager.py             # 配置管理器
│   ├── Logger.py                    # 日志配置
│   ├── NacosConfig.py               # Nacos 配置
│   └── RedisConfig.py               # Redis 配置
│
├── common/                          # 公共常量模块
├── util/                            # 工具模块
└── tools/                           # 业务工具模块
```

---

## 四、API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/chat` | 非流式对话接口 |
| POST | `/chat/stream` | 流式对话接口（SSE） |
| POST | `/chat/stop?session_id=&agent_id=` | 停止指定会话 |
| GET/POST | `/hello` | 健康检查/测试接口 |

### /chat（非流式）

**请求参数：**

```json
{
    "question": "脖子上的装饰品",
    "sessionId": "11",
    "userToken": "xxxx",
    "agentId": 1004
}
```

**响应：**

```json
{
    "itemName": "脖子上的装饰品",
    "categoryId": 1014,
    "categoryName": "玩具礼品",
    "isProhibited": false,
    "reason": "属于饰品类，通常为项链等装饰品"
}
```

### SSE 流式输出格式

```
data: {"eventType": 1001, "eventData": "你好"}

data: {"eventType": 1001, "eventData": "！我是智链云途智能助手。"}

data: {"eventType": 1001, "eventData": "有什么我可以帮你的吗？"}

```

---

## 五、技术栈

| 分类 | 技术组件 |
|------|----------|
| **Web 框架** | FastAPI 0.135.2 · Uvicorn 0.42.0 |
| **AI 框架** | LangChain 1.3.6 · LangGraph 1.2.4 · LangChain OpenAI 1.3.0 |
| **机器学习** | PyTorch 2.6.0 · Transformers 5.4.0 · XGBoost 3.2.0 · Scikit-learn 1.8.0 |
| **中间件** | Nacos · Redis 7.4.0 · MongoDB · RabbitMQ (Pika) |
| **协议** | MCP (Model Context Protocol) · SSE (Server-Sent Events) |
| **环境** | Python 3.13 · Conda |

---

## 六、快速开始

### 6.1 环境要求

- Python 3.13
- Conda
- Redis
- MongoDB
- RabbitMQ
- Nacos

### 6.2 安装依赖

```bash
# 创建 Conda 环境
conda env create -f environment.yml

# 激活环境
conda activate cc_env_31313
```

### 6.3 准备模型文件

> 模型文件较大，未提交到 Git 仓库，请通过以下方式获取：

**方式一：从课前资料获取**
- `bert-base-chinese` 预训练模型已在课前资料中提供，解压至 `lingjian_agent/categorie/pretrained_models/` 目录

**方式二：自行训练**
- 执行训练脚本生成 `saved_model`：
  ```bash
  python lingjian_agent/categorie/train_bert.py
  ```
  训练完成后模型将保存至 `lingjian_agent/categorie/saved_model/` 目录

### 6.4 配置环境变量

复制 `.env.example` 为 `.env` 并修改配置：

```bash
cp .env.example .env
```

主要配置项：

| 变量 | 说明 |
|------|------|
| `AMAP_MCP_URL` | 高德 MCP 服务地址 |
| `AI_OPENAI_MODEL` | AI 模型名称 |
| `AI_OPENAI_API_KEY` | AI API Key |
| `AI_OPENAI_BASE_URL` | AI Base URL |
| `AI_OPENAI_TEMPERATURE` | Temperature 参数 |
| `AI_OPENAI_TIMEOUT` | 请求超时时间 |

### 6.5 启动服务

```bash
python main.py
```

服务启动后：
- Web 服务监听 `0.0.0.0:28001`
- 自动向 Nacos 注册服务
- 自动启动 RabbitMQ 消息消费者

---

## 七、配置说明

### application.yml 主要配置项

| 配置项 | 说明 |
|--------|------|
| `server.port` | 服务端口（默认 28001） |
| `server.host` | 监听地址（默认 0.0.0.0） |
| `nacos.*` | Nacos 注册与配置中心 |
| `ai.openai.*` | OpenAI 兼容模型配置 |
| `ai.dispatch.*` | AI 调度模型配置（mock/real 模式） |
| `prompt.*` | 各智能体系统提示词配置 |
| `redis.*` | Redis 连接配置 |
| `rabbitmq.*` | RabbitMQ 消息队列配置 |
| `mongo.*` | MongoDB 数据库配置 |
| `cc.services.*` | 业务系统微服务名称映射 |
| `mcp-servers` | MCP 服务配置（高德地图） |

---

## 八、架构说明

### 8.1 BaseAgent 抽象类

所有智能体继承自 `BaseAgent`，提供以下能力：
- `execute()` - 流式执行接口，返回 `AsyncIterable[str]`（SSE 格式）
- `execute_one_shot()` - 非流式执行接口
- `session_detail()` - 获取会话详情
- `delete_session()` - 删除会话数据
- `stop()/reset_stop()/is_stop()` - Redis 停止标记管理
- `init_llm()` - 初始化 LLM（统一走 OpenAI Compatible 接口）

### 8.2 SSE 流式输出

智能体通过 SSE (Server-Sent Events) 协议进行流式输出：
- 每条消息格式：`data: {"eventType": <类型>, "eventData": <数据>}\n\n`
- `eventType: 1002` 表示停止事件
- 前端可通过 `/chat/stop` 接口发送停止信号，通过 Redis 标记中断输出

### 8.3 Nacos 服务注册

服务启动时自动向 Nacos 注册，关闭时自动注销：
- 服务名：`lingjian-agent`
- 心跳间隔：10 秒
- 配置命名空间：`<NACOS_CONFIG_NAMESPACE>`
- 注册命名空间：`<NACOS_DISCOVERY_NAMESPACE>`

### 8.4 RabbitMQ 消息消费

`dispatch_train_consumer` 在启动时开始消费调度训练消息：
- Exchange: `cc.exchange.topic.aigc`
- Queue: `cc.queue.aigc.dispatch.train`
- RoutingKey: `CREATE` / `COMPLETE`

### 8.5 MongoDB 存储

用于存储调度决策记录：
- Database: `cc_ai`
- Collections: `dispatch_decision_record`, `dispatch_decision_record_mock`

---

## 九、智能体开发指南

### 9.1 新增智能体

1. 在 `lingjian_agent/` 目录下创建智能体模块
2. 继承 `BaseAgent` 抽象类，实现必要方法
3. 在 `lingjian_agent/Agents.py` 中注册智能体
4. 在 `application.yml` 中配置对应的 prompt

### 9.2 Agent ID 规范

新增智能体 ID 需从 **1011** 开始递增，避免冲突。

### 9.3 智能体模板

```python
from agent.BaseAgent import BaseAgent

class MyAgent(BaseAgent):
    def id(self) -> int:
        return 1011  # 从 1011 开始递增
    
    async def execute(self, question: str, session_id: str, user_token: str):
        # 流式执行逻辑
        yield make_sse_event(1001, "处理中...")
    
    async def execute_one_shot(self, question: str, session_id: str, user_token: str):
        # 非流式执行逻辑
        return {"content": "回复内容", "eventType": 1001}
    
    async def session_detail(self, user_id: int, session_id: str) -> list:
        # 获取会话详情
        pass
    
    async def delete_session(self, session_id: str):
        # 删除会话数据
        pass
```

---

## 十、注意事项

- **environment.yml 修改需确认**：禁止随意修改环境依赖文件，需人工确认
- **禁止随意删除代码**：优先在现有代码基础上做增量修改
- **最小影响范围**：尽量以最小改动完成需求，不重构无关代码
- **保留已有注释**：不对现有注释进行删改
- **Windows 兼容性**：项目入口使用 `SelectorEventLoop` 以兼容 psycopg 异步

---

## 十一、许可证

本项目仅供学习与研究使用。

---

**lingjian-agent** - 让物流更智能 🚚🤖
