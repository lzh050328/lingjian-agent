"""
快递员调度数据生成器

该模块用于生成模拟的快递员配送数据，用于训练和测试调度算法。
生成的数据包含快递员的各种特征以及是否按时完成配送的标签。

主要功能：
- 生成包含快递员ID、任务数量、配送距离、空闲时间、时间段等特征的模拟数据
- 基于预设的权重模型计算配送成功率
- 将生成的数据保存为CSV文件供后续分析使用

"""

import pandas as pd
import numpy as np

# 设置随机种子以确保结果可重现
np.random.seed(42)

# 定义要生成的数据样本数量
data_size = 10000

# 初始化数据存储列表
data = []

# 生成模拟数据
for _ in range(data_size):
    # 生成快递员ID (1-49之间的随机整数)
    courier_id = np.random.randint(1, 50)

    # 生成当前任务数量 (0-19之间的随机整数)
    task_count = np.random.randint(0, 20)

    # 生成配送距离 (0.5-10公里之间的随机浮点数)
    distance = np.random.uniform(0.5, 10)  # 单位：km

    # 生成空闲时间 (0-120分钟之间的随机整数)
    idle_time = np.random.randint(0, 120)  # 单位：分钟

    # 生成当前小时 (0-23之间的随机整数)
    hour = np.random.randint(0, 24)

    # 基于业务逻辑构造评分模型
    # 各项特征的权重：
    # - 任务数量权重：0.4 (任务越多，配送压力越大)
    # - 配送距离权重：0.3 (距离越远，配送难度越大)
    # - 空闲时间权重：-0.2 (空闲时间越长，配送效率越高)
    # - 时间段权重：0.1 (11-13点为午餐高峰期，配送难度增加)
    score = (
        0.4 * task_count +
        0.3 * distance -
        0.2 * idle_time +
        0.1 * (1 if 11 <= hour <= 13 else 0)
    )

    # 使用sigmoid函数将评分转换为概率值
    # 减去5是为了调整概率分布的中心点
    prob = 1 / (1 + np.exp(score - 5))

    # 根据计算出的概率决定是否按时完成配送
    # 1表示按时完成，0表示未按时完成
    on_time = 1 if np.random.rand() < prob else 0

    # 将生成的样本数据添加到列表中
    data.append([
        courier_id, task_count, distance, idle_time, hour, on_time
    ])

# 将数据列表转换为Pandas DataFrame
# 定义各列的名称
columns = ["courier_id", "task_count", "distance", "idle_time", "hour", "on_time"]
df = pd.DataFrame(data, columns=columns)

# 将生成的数据保存到CSV文件中
# index=False表示不保存行索引
output_file = "courier_data.csv"
df.to_csv(output_file, index=False)

# 输出生成完成信息
print(f"数据生成完成，共生成 {data_size} 条记录，已保存到 {output_file}")
print(f"数据包含以下字段：{', '.join(columns)}")
print("\n字段说明：")
print("- courier_id: 快递员ID")
print("- task_count: 当前任务数量")
print("- distance: 配送距离（公里）")
print("- idle_time: 空闲时间（分钟）")
print("- hour: 当前小时（0-23）")
print("- on_time: 是否按时完成（1=是，0=否）")