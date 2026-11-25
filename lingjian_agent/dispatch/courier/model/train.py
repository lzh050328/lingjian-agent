"""
快递员调度预测模型训练

该模块用于训练一个XGBoost分类器，预测快递员是否能够按时完成配送任务。
模型基于快递员的任务数量、配送距离、空闲时间和当前时间段等特征进行预测。

主要功能：
- 加载预处理后的快递员数据
- 准备训练特征和标签
- 使用XGBoost算法训练分类模型
- 评估模型性能
- 保存训练好的模型供后续使用
"""

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# 1. 数据加载和预处理
# 读取之前生成的快递员数据CSV文件
# 数据路径：../data/courier_data.csv (相对路径)
df = pd.read_csv("../data/courier_data.csv")

# 准备特征矩阵X和标签向量y
# 特征包括：任务数量、配送距离、空闲时间、当前小时
X = df[["task_count", "distance", "idle_time", "hour"]]
# 标签：是否按时完成配送 (1=按时完成，0=未按时完成)
y = df["on_time"]

# 2. 数据分割
# 将数据集分割为训练集和测试集
# test_size=0.2 表示20%的数据用于测试，80%用于训练
# random_state=42 确保每次运行结果一致，便于调试和复现
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"训练集样本数: {len(X_train)}")
print(f"测试集样本数: {len(X_test)}")
print(f"特征维度: {X_train.shape[1]}")

# 3. 模型训练
# 使用XGBoost分类器进行训练
# 参数说明：
# - n_estimators=100: 使用100棵决策树
# - max_depth=5: 每棵树的最大深度为5
# - learning_rate=0.1: 学习率为0.1，控制每棵树的贡献权重
model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1
)

# 在训练集上训练模型
print("开始训练模型...")
model.fit(X_train, y_train)
print("模型训练完成")

# 4. 模型评估
# 在测试集上进行预测
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.4f}")

# 5. 模型保存
# 使用joblib保存训练好的模型
# 保存路径：当前目录下的xgb_model.pkl文件
model_filename = "xgb_model.pkl"
joblib.dump(model, model_filename)

print(f"模型已保存到: {model_filename}")
print("\n模型训练流程完成！")
print("\n使用说明：")
print("- 该模型可用于预测快递员配送的准时率")
print("- 输入特征：[task_count, distance, idle_time, hour]")
print("- 输出：0（未按时完成）或 1（按时完成）")
print("- 模型文件可通过 joblib.load() 加载使用")
