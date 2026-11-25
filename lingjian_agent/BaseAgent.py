import json
from abc import ABC, abstractmethod
from typing import AsyncIterable, Any

from langchain_openai import ChatOpenAI

from lingjian_config import redis_config, config_manager
from lingjian_util import JsonUtil


class StrictChatOpenAI(ChatOpenAI):
    """兼容上游接口对非流式响应做双重 JSON 编码的情况"""

    use_max_tokens: bool = False

    def _create_chat_result(self, response, generation_info=None):
        # 处理上游返回字符串而非 dict 的情况
        if isinstance(response, str):
            response = json.loads(response)
        return super()._create_chat_result(response, generation_info)

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if self.use_max_tokens and "max_completion_tokens" in payload:
            # 网关兼容模式：langchain 会把 max_tokens 转成 max_completion_tokens，
            # 网关会再补发 max_tokens，导致上游参数冲突 400
            payload["max_tokens"] = payload.pop("max_completion_tokens")
        return payload


STOP_EVENT = {"eventType": 1002}  # SSE 停止事件结构

redis = redis_config.get_instance()
STOP_FLAG_TTL = 120  # 停止标记超时时间（秒）


def format_sse_data(data: dict) -> str:
    """将数据格式化为标准 SSE 消息格式：data: <内容>\\n\\n"""
    return f"data: {JsonUtil.to_str(data)}\n\n"


def make_sse_event(event_type: int, data: Any) -> str:
    """构建 SSE 事件字符串"""
    return format_sse_data({"eventType": event_type, "eventData": data})


def parse_agent_response(result: dict, response_class: type = None):
    """解析 Agent invoke 结果，支持 structured_response 和 messages 两种格式"""
    if "structured_response" in result:
        return result["structured_response"]
    else:
        messages = result.get("messages", [])
        if not messages:
            return None

        last_message = messages[-1]
        if not hasattr(last_message, "content"):
            return None

        content = last_message.content
        if not content:
            raise ValueError(
                "LLM 返回 content 为空，疑似 reasoning 耗尽 token 预算（max_tokens 不足）"
            )
        if response_class:
            return JsonUtil.to_obj(content, response_class)
        return content


class BaseAgent(ABC):
    """所有智能体的基础抽象类，提供统一接口和 Redis 停止标记管理"""

    @abstractmethod
    async def execute(self, question: str, session_id: str, user_token: str) -> AsyncIterable[str]:
        """流式执行，返回 AsyncIterable[str]（SSE 格式）"""
        pass

    @abstractmethod
    async def execute_one_shot(self, question: str, session_id: str, user_token: str):
        """非流式执行，返回完整响应"""
        pass

    @abstractmethod
    def id(self) -> int:
        """返回智能体唯一 ID"""
        pass

    @abstractmethod
    async def session_detail(self, user_id: int, session_id: str) -> list:
        """获取会话详情"""
        pass

    @abstractmethod
    async def delete_session(self, session_id: str):
        """删除会话"""
        pass

    def is_streaming(self) -> bool:
        """是否支持流式输出，默认 True"""
        return True

    async def init(self):
        """智能体初始化钩子"""
        pass

    async def destroy(self):
        """智能体销毁钩子"""
        pass

    def stop(self, session_id: str):
        """标记会话为停止状态"""
        redis.setex(self.get_flags_key(session_id), STOP_FLAG_TTL, "1")

    def get_flags_key(self, session_id: str):
        """构建 Redis 停止标记 key：CC_AGENT_STOP_FLAGS:<agent_id>:<session_id>"""
        return f"CC_AGENT_STOP_FLAGS:{self.id()}:{session_id}"

    def reset_stop(self, session_id: str):
        """清除停止标记"""
        redis.delete(self.get_flags_key(session_id))

    def is_stop(self, session_id: str):
        """检查会话是否已标记停止"""
        return redis.exists(self.get_flags_key(session_id))

    def init_llm(self, provider: str):
        """初始化 LLM（OpenAI Compatible 接口）"""
        cfg = config_manager
        prefix = f"ai.{provider}"

        model_name = cfg.get(f"{prefix}.model")
        api_key = cfg.get(f"{prefix}.api-key")
        base_url = cfg.get(f"{prefix}.base-url")
        temperature = float(cfg.get(f"{prefix}.temperature", 0.7))
        timeout = int(cfg.get(f"{prefix}.timeout", 60))
        max_tokens = int(cfg.get(f"{prefix}.max-tokens", 4096))
        use_max_tokens = str(cfg.get(f"{prefix}.use-max-tokens", False)).lower() == "true"

        return StrictChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            timeout=timeout,
            max_tokens=max_tokens,
            use_max_tokens=use_max_tokens
        )
