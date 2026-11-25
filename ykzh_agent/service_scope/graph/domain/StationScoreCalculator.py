from .enums import LoadStatus, SlaStatus, StationStatus
from .models import StationFact

# =========================
# 推荐逻辑配置
# =========================
low_confidence_nearby_top_n: int = 2
route_time_high_score_threshold: int = 10
route_time_medium_score_threshold: int = 20
route_time_low_score_threshold: int = 30

class StationScoreCalculator:
    """
    网点评分计算器。

    设计目标：
    1. 保持规则简单、稳定、可解释
    2. 避免把最终推荐完全交给 LLM
    3. 让推荐结果可以审计和复盘

    当前 V1 使用以下维度评分：
    - 是否命中服务范围
    - 是否边界区
    - 路网时间
    - 网点负载
    - 网点状态
    - 近时段 SLA
    """
    def calculate(self, fact: StationFact) -> int:
        score = 0

        # =========================
        # 1. 服务范围分
        # 命中 polygon 的网点优先
        # =========================
        if fact.in_polygon:
            score += 40
        else:
            score += 10

        # =========================
        # 2. 相邻网点扣分
        # 相邻网点不是最优推荐，所以不直接剔除，只做轻微扣分
        # =========================
        if fact.is_adjacent:
            score -= 2

        # =========================
        # 3. 路网时间分
        # 路网时长越短，越有可能更适合揽收
        # =========================
        if fact.route_time_min <= route_time_high_score_threshold:
            score += 30
        elif fact.route_time_min <= route_time_medium_score_threshold:
            score += 20
        elif fact.route_time_min <= route_time_low_score_threshold:
            score += 10

        # =========================
        # 4. 网点负载分
        # 当前负载越轻，越适合接单
        # =========================
        if fact.load_status == LoadStatus.LOW:
            score += 20
        elif fact.load_status == LoadStatus.NORMAL:
            score += 15
        elif fact.load_status == LoadStatus.HIGH:
            score += 5
        elif fact.load_status == LoadStatus.OVERLOAD:
            score -= 20

        # =========================
        # 5. 网点状态分
        # 停运网点理论上前面就应被剔除，这里再兜底一次
        # =========================
        if fact.station_status == StationStatus.NORMAL:
            score += 10
        elif fact.station_status == StationStatus.LIMITED:
            score += 0
        elif fact.station_status == StationStatus.STOPPED:
            score -= 100

        # =========================
        # 6. SLA 近时段表现
        # 这个字段 V1 可选，如果下游系统暂无能力，可给默认 MEDIUM
        # =========================
        if fact.sla_status == SlaStatus.GOOD:
            score += 10
        elif fact.sla_status == SlaStatus.MEDIUM:
            score += 5
        elif fact.sla_status == SlaStatus.POOR:
            score += 0

        return score

# 全局评分计算器实例
station_score_calculator = StationScoreCalculator()