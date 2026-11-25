import random
import pandas as pd
import time


def generate_sample(n=5000):
    """
    生成运输订单优先级训练数据集

    该函数模拟生成用于训练运输订单调度优先级模型的样本数据。
    根据业务规则，优先级主要由等待时间和订单规模决定。

    参数:
        n (int): 需要生成的样本数量，默认为5000

    返回:
        pd.DataFrame: 包含以下字段的数据框
            - total_weight: 订单总重量(kg)
            - total_volume: 订单总体积(m³)
            - wait_time: 等待时间(秒)
            - route_order_count: 当前线路的订单数量
            - label: 优先级标签(0:普通, 1:高优先级)
    """
    data = []
    now = int(time.time())

    for _ in range(n):
        # 生成订单基础信息
        total_weight = random.uniform(1, 50)  # 订单重量: 1-50kg
        total_volume = random.uniform(0.5, 10)  # 订单体积: 0.5-10m³

        # 创建时间（模拟过去1小时内产生的订单）
        created = now - random.randint(0, 3600)

        # 计算等待时间（当前时间 - 创建时间）
        wait_time = now - created

        # 当前线路订单数（模拟线路繁忙程度）
        route_order_count = random.randint(1, 50)

        # ====== 优先级标签规则 ======
        # 业务规则：
        # 1. 等待时间超过30分钟(1800秒)的订单自动标记为高优先级
        # 2. 重量超过30kg且当前线路订单数超过20个的订单标记为高优先级
        # 3. 其他情况标记为普通优先级
        label = 0
        if wait_time > 1800 or (total_weight > 30 and route_order_count > 20):
            label = 1

        # 构建样本数据
        data.append({
            "total_weight": total_weight,
            "total_volume": total_volume,
            "wait_time": wait_time,
            "route_order_count": route_order_count,
            "label": label
        })

    return pd.DataFrame(data)


if __name__ == "__main__":
    # 生成10000个训练样本
    df = generate_sample(10000)

    # 保存到CSV文件用于模型训练
    df.to_csv("train_data.csv", index=False)
    print("数据生成完成")