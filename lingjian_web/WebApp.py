from fastapi import FastAPI, Request
from starlette.responses import PlainTextResponse
from contextlib import asynccontextmanager

from lingjian_web.routers import *
from lingjian_agent import AGENTS
from lingjian_config import nacos_config, config_manager, logger
from lingjian_common import *
from lingjian_agent.dispatch.transport_task.consumer import dispatch_train_consumer

@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup()   # 启动时执行
    yield
    await shutdown()  # 关闭事件

# ========================= 创建 FastAPI 实例 =========================
app = FastAPI(
    title="SL Agent Web Server",
    description="智链云途智能体服务",
    lifespan=lifespan
)

# ========================= 启动事件 =========================
async def startup():
    # 初始化所有 Agent
    for agent in AGENTS.values():
        await agent.init()

    # 向nacos注册服务
    register_service()

    # 启动消息消费者
    dispatch_train_consumer.start()

# ========================= 关闭事件 =========================
async def shutdown():
    # 销毁所有 Agent
    for agent in AGENTS.values():
        await agent.destroy()

    # 从nacos注销服务
    deregister_service()

    # 停止消息消费者
    dispatch_train_consumer.stop()

# ========================= Nacos 注册与注销 =========================
def register_service():
    """
    注册智能体中心服务到 Nacos 服务发现
    """
    client = nacos_config.get_discovery_client()  # 获取 Nacos 注册客户端
    ip = nacos_config.get_discovery_ip()  # 服务 IP
    service_name = nacos_config.get_discovery_name()  # 服务名称
    port = int(config_manager.get(SERVER_PORT))  # 服务端口
    group_name = nacos_config.get_discovery_group()  # 分组名称

    # 将实例注册到 Nacos
    result = client.add_naming_instance(
        service_name=service_name,
        ip=ip,
        port=port,
        group_name=group_name,
        heartbeat_interval=10  # 心跳间隔 10 秒
    )
    logger.info(f"✅ Registered {service_name} to Nacos: {result}")
    return result

def deregister_service():
    """
    注销智能体中心服务
    """
    ip = nacos_config.get_discovery_ip()
    service_name = nacos_config.get_discovery_name()
    port = int(config_manager.get(SERVER_PORT))

    # 从 Nacos 注销实例
    result = nacos_config.get_discovery_client().remove_naming_instance(
        service_name, ip, port
    )
    logger.info(f"🧹 Deregistered {service_name} from Nacos")
    return result

# ========================= 异常处理 =========================
def system_exception_handler(req: Request, exc: Exception):
    """
    全局异常处理函数，将异常转换为 500 响应
    """
    return PlainTextResponse(
        content=str(exc),
        status_code=500
    )


# 添加全局异常处理器
app.add_exception_handler(Exception, system_exception_handler)

# ========================= 路由注册 =========================
# 将各个子路由模块挂载到不同前缀下
app.include_router(hello_router, prefix="/hello", tags=["hello"])
app.include_router(chat_router, prefix="/chat", tags=["chat"])