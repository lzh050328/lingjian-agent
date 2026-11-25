from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any

from ykzh_agent.service_scope.graph.domain.models import GraphState


class BaseGraphNode(ABC):
    """
    所有图节点的抽象基类。

    设计目的：
    1. 统一 run 方法签名
    2. 保持每个节点都是“可调用对象”
    3. 便于注入依赖并做单测
    """

    @abstractmethod
    async def run(self, state: GraphState) -> Dict[str, Any]:
        """
        节点执行入口。

        参数：
        - state: 当前图状态对象

        返回：
        - dict，表示对图状态的局部更新
        """
        raise NotImplementedError

    async def __call__(self, state: GraphState) -> Dict[str, Any]:
        """
        让节点对象本身可调用，便于直接传给 LangGraph add_node。
        """
        return await self.run(state)