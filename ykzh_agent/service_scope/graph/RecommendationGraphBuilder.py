from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from ykzh_agent.service_scope.graph.GraphRouters import *
from ykzh_agent.service_scope.graph.domain import station_score_calculator
from ykzh_agent.service_scope.graph.domain.models import GraphState
from ykzh_agent.service_scope.graph.nodes import AddressAnalyzerAgentNode
from ykzh_agent.service_scope.graph.nodes import CandidateStationFetcherNode
from ykzh_agent.service_scope.graph.nodes import ReasonGeneratorNode
from ykzh_agent.service_scope.graph.nodes import StationFactEnricherNode
from ykzh_agent.service_scope.graph.nodes import StationRecommenderNode


class RecommendationGraphBuilder:
    """
    推荐图构建器。

    职责：
    1. 初始化 LLM、Node
    2. 用 LangGraph 将所有节点串起来
    3. 返回 compile 后的 graph 对象

    设计思路：
    - 所有节点都采用 class 的方式组织
    - builder 负责依赖注入
    - 每个节点自己管理所需的工具和客户端
    """

    def __init__(self, llm: BaseChatModel) -> None:
        """
        初始化推荐图构建器。

        Args:
            llm: 语言模型实例，由外部ServiceScopeAgent提供
        """
        self.llm = llm

        # =========================
        # 1. 获取评分计算器实例
        # =========================
        self.score_calculator = station_score_calculator

        # =========================
        # 2. 初始化 Nodes
        # =========================
        # 注意：每个节点自己管理所需的工具和客户端，LLM由外部传入
        self.address_analyzer_node = AddressAnalyzerAgentNode(self.llm)
        self.candidate_station_fetcher_node = CandidateStationFetcherNode()
        self.station_fact_enricher_node = StationFactEnricherNode(self.llm)
        self.station_recommender_node = StationRecommenderNode(self.score_calculator)
        self.reason_generator_node = ReasonGeneratorNode(self.llm)

    def build(self):
        """
        构建并编译 LangGraph。

        节点流程：
        START
          -> address_analyzer
          -> (有坐标?) candidate_station_fetcher : END
          -> candidate_station_fetcher
          -> station_fact_enricher
          -> station_recommender
          -> (成功?) reason_generator : END
          -> END
        """
        builder = StateGraph(GraphState)

        # 添加所有节点
        builder.add_node("address_analyzer", self.address_analyzer_node)
        builder.add_node("candidate_station_fetcher", self.candidate_station_fetcher_node)
        builder.add_node("station_fact_enricher", self.station_fact_enricher_node)
        builder.add_node("station_recommender", self.station_recommender_node)
        builder.add_node("reason_generator", self.reason_generator_node)

        # 构建图流程
        builder.add_edge(START, "address_analyzer")

        # 地址分析后，如果有坐标则继续，否则直接结束
        builder.add_conditional_edges(
            "address_analyzer",
            route_after_address_analysis,
            {
                "candidate_station_fetcher": "candidate_station_fetcher",
                "end": END,
            },
        )

        # 线性流程：候选网点获取 -> 网点事实补全 -> 推荐决策
        builder.add_edge("candidate_station_fetcher", "station_fact_enricher")
        builder.add_edge("station_fact_enricher", "station_recommender")

        # 推荐后，如果成功则生成理由，否则直接结束
        builder.add_conditional_edges(
            "station_recommender",
            route_after_recommender,
            {
                "reason_generator": "reason_generator",
                "end": END,
            },
        )

        builder.add_edge("reason_generator", END)

        graph = builder.compile()

        # try:
        #     # 使用 Mermaid 生成图表并保存为文件
        #     mermaid_code = graph.get_graph().draw_mermaid_png()
        #     file_name = "recommendation_graph.png"
        #     with open(file_name, "wb") as f:
        #         f.write(mermaid_code)
        #
        #     print(f"工作流结构图已保存至 {file_name}")
        # except Exception as e:
        #     print(f"An error occurred: {e}")

        return graph