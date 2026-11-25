from fastapi import APIRouter

# ========================= 创建路由对象 =========================
hello_router = APIRouter()

# ========================= 路由接口 =========================
@hello_router.get("/")
async def create_token():
    return "hello world!"
