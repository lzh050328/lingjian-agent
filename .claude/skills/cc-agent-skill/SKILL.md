---
name: cc-agent-skill
description: 基于 FastAPI + LangChain + LangGraph 的 Python AI 智能体服务框架，提供统一的智能体管理、多模型接入和 API 接口服务，支持流式响应和会话管理。
---

# CC Agent Server - Python AI Agent Framework

## 项目概述

这是一个基于Python的AI智能体服务框架，采用FastAPI + LangChain + LangGraph技术栈，支持多种AI模型接入，提供统一的智能体管理和API接口服务。

## 架构设计

### 1. 整体架构
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Layer     │    │   Agent Layer   │    │   Config Layer  │
│   - FastAPI     │◄──►│   - BaseAgent   │◄──►│   - Nacos       │
│   - SSE Stream  │    │   - Specialized  │    │   - Redis       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Routes    │    │   LLM Models    │    │   Data Sources  │
│   - ChatRouter  │    │   - OpenAI      │    │   - PostgreSQL  │
│   - HelloRouter │    │   - Custom APIs │    │   - MCP Tools   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 2. 目录结构规范

```
sl_agent_server/
├── main.py                    # 应用入口点
├── application.yml            # 配置文件
├── SKILL.md                   # 项目规范文档 (本文件)
├── lingjian_agent/                     # 智能体核心模块
│   ├── BaseAgent.py          # 基础智能体抽象类
│   ├── Agents.py             # 智能体注册中心
│   └── [domain]/             # 业务域智能体
│       ├── __init__.py
│       └── [AgentName].py
├── web/                      # Web服务层
│   ├── WebApp.py            # FastAPI应用配置
│   └── routers/             # API路由
├── config/                   # 配置管理
│   ├── ConfigManager.py     # 配置管理器
│   └── [ConfigType].py      # 特定配置
├── util/                     # 工具函数
└── tools/                    # 业务工具
```

## 编码规范

### 1. Agent开发规范

#### 基础Agent模板
```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from agent.BaseAgent import *
from agent.prompts import system_prompt_config

# 1. 定义响应数据类 (必需)
@dataclass
class [AgentName]Response:
    """[功能描述]响应结构"""
    field1: str
    field2: int
    # ... 其他字段

# 2. 定义枚举 (可选)
class [Status]Enum(Enum):
    """状态枚举"""
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"

# 3. 实现Agent类
class [AgentName](BaseAgent):
    def __init__(self, provider: str = "openai"):
        self.provider = provider
        self.llm: Optional[BaseChatModel] = None
        self.agent: Optional[CompiledStateGraph] = None

    async def init(self):
        """初始化Agent"""
        self.llm = super().init_llm(self.provider)
        # 初始化具体agent逻辑

    async def execute(self, question: str, session_id: str, user_token: str) -> AsyncIterable[str]:
        """流式执行方法"""
        # 实现流式响应逻辑
        yield make_sse_event(1001, "处理中...")
        # ... 业务逻辑
        yield make_sse_event(1001, result)

    async def execute_one_shot(self, question: str, session_id: str, user_token: str):
        """非流式执行方法"""
        # 实现一次性响应逻辑
        result = await self.agent.ainvoke({"messages": prompts})
        return parse_agent_response(result, [AgentName]Response)

    def id(self) -> int:
        """返回唯一Agent ID"""
        return 1001  # 唯一标识

    # 会话管理方法 (根据需求实现)
    async def session_detail(self, user_id: int, session_id: str) -> list:
        raise NotImplementedError("暂不提供实现.")

    async def delete_session(self, session_id: str):
        raise NotImplementedError("暂不提供实现.")

# 全局实例
global_agent_instance = [AgentName]()
```

#### Agent注册规范
在 `lingjian_agent/Agents.py` 中注册：
```python
from agent.risk.RiskControlAgent import risk_control_agent
from agent.carriage.ComputeCarriageAgent import compute_carriage_agent

AGENTS = {
    risk_control_agent.id(): risk_control_agent,
    compute_carriage_agent.id(): compute_carriage_agent,
    # ... 其他agents
}
```

### 2. API开发规范

#### Router模板
```python
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class RequestModel(BaseModel):
    """请求体模型"""
    field1: str
    field2: int = 0

@router.post("/endpoint")
async def endpoint_handler(req: RequestModel):
    """端点处理函数"""
    # 业务逻辑
    return {"status": "success", "data": result}
```

#### SSE流式响应规范
```python
from fastapi.responses import StreamingResponse
from agent.BaseAgent import make_sse_event

async def stream_handler() -> AsyncIterable[str]:
    """SSE流处理器"""
    yield make_sse_event(1001, "开始处理")
    # ... 处理逻辑
    yield make_sse_event(1001, result)
    yield make_sse_event(1002, None)  # 结束标记

def create_streaming_response(data: AsyncIterable[str]):
    """创建流式响应"""
    return StreamingResponse(
        data,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"}
    )
```

### 3. 配置管理规范

#### 配置文件结构 (application.yml)
```yaml
server:
  port: 28001
  host: 0.0.0.0

ai:
  openai:
    model: LongCat-Flash-Lite
    api-key: ${API_KEY}  # 环境变量支持
    base-url: https://api.example.com/
    temperature: 0.8

redis:
  host: localhost
  port: 6379
  password: password
```

#### 配置访问规范
```python
from config import config_manager

# 获取配置值
host = config_manager.get("server.host")
api_key = config_manager.get("ai.openai.api-key")
```

### 4. 工具类开发规范

#### 工具类模板
```python
class [ToolName]Utils:
    """[工具功能描述]"""

    @staticmethod
    def utility_method(param: str) -> dict:
        """静态工具方法"""
        # 实现工具逻辑
        return {"result": "success"}

    @classmethod
    async def async_utility(cls, param: str) -> dict:
        """异步工具方法"""
        # 实现异步逻辑
        return {"result": "success"}
```

## 技术栈规范

### 1. 核心依赖
- **Python**: 3.13+
- **FastAPI**: Web框架
- **LangChain/LangGraph**: AI智能体框架
- **Uvicorn**: ASGI服务器
- **Pydantic**: 数据验证
- **Redis**: 缓存和会话管理
- **Nacos**: 配置中心和服务发现

### 2. 代码质量要求

#### 类型注解
```
# ✅ 正确：完整的类型注解
async def process_data(data: dict[str, Any], session_id: str) -> AsyncIterable[str]:
    """处理数据并返回流"""
    for item in data:
        yield make_sse_event(1001, item)

# ❌ 错误：缺少类型注解
def process_data(data, session_id):
    return data
```

#### 错误处理
```
# ✅ 正确：结构化错误处理
try:
    result = await agent.execute(question, session_id, token)
except ConnectionError as e:
    logger.error(f"连接错误: {e}")
    yield make_sse_event(500, "服务暂时不可用")
except Exception as e:
    logger.exception(f"未知错误: {e}")
    yield make_sse_event(500, "处理请求时发生错误")
```

#### 日志规范
```
from config import logger

# ✅ 正确：结构化日志
logger.info(f"【模块名】操作描述: param1={value1}, param2={value2}")
logger.debug(f"【模块名】详细调试信息: {detailed_info}")
logger.error(f"【模块名】错误信息: {error_msg}")
```

## 部署和运维规范

### 1. 环境配置
```bash
# 环境变量配置
export LONGCAT_API_KEY="your-api-key"
export REDIS_PASSWORD="your-redis-password"

# 启动服务
python main.py
```

### 2. 监控指标
- **健康检查**: `/hello` 端点
- **性能指标**: 请求响应时间、并发连接数
- **业务指标**: Agent执行成功率、平均响应时间

### 3. 服务注册
```python
# Nacos服务注册
def register_service():
    """自动注册到Nacos服务发现"""
    client.add_naming_instance(
        service_name="sl-agent-server",
        ip="192.168.1.100",
        port=28001,
        heartbeat_interval=10
    )
```

## 扩展指南

### 1. 添加新Agent
1. 在 `lingjian_agent/[domain]/` 下创建新Agent文件
2. 继承 `BaseAgent` 并实现必需方法
3. 定义响应数据类
4. 在 `lingjian_agent/Agents.py` 中注册
5. 添加对应的API路由

### 2. 添加新API端点
1. 在 `web/routers/` 下创建或修改Router文件
2. 定义请求/响应模型
3. 实现业务逻辑
4. 在 `web/WebApp.py` 中注册路由

### 3. 添加配置项
1. 在 `application.yml` 中添加配置
2. 使用 `config_manager.get()` 访问
3. 支持环境变量替换

## 最佳实践

### 1. Agent设计原则
- **单一职责**: 每个Agent专注于一个业务领域
- **无状态设计**: 会话状态通过Redis管理
- **可测试性**: 提供mock和测试支持
- **可观测性**: 完整的日志和监控

### 2. 性能优化
- **异步处理**: 所有IO操作使用async/await
- **连接池**: 数据库和Redis连接复用
- **缓存策略**: 合理使用Redis缓存
- **流式响应**: 大数据量使用SSE流式传输

### 3. 安全考虑
- **输入验证**: 所有API参数严格验证
- **权限控制**: Token验证和权限检查
- **敏感信息**: API密钥使用环境变量
- **错误信息**: 避免泄露敏感信息

## 故障排查

### 常见问题
1. **Agent未注册**: 检查 `AGENTS` 字典是否包含新Agent
2. **配置加载失败**: 检查 `application.yml` 格式和环境变量
3. **SSE连接断开**: 检查Redis连接和超时设置
4. **模型调用失败**: 检查API密钥和网络连接

### 调试工具
```python
# 启用详细日志
logging.basicConfig(level=logging.DEBUG)

# 测试Agent独立运行
if __name__ == "__main__":
    agent = MyAgent()
    asyncio.run(agent.init())
    result = asyncio.run(agent.execute_one_shot("测试问题", "test-session", "test-token"))
    print(result)
```

---

**版本**: 1.0.0
**最后更新**: 2026-03-30
**适用框架**: Python + FastAPI + LangChain + LangGraph
