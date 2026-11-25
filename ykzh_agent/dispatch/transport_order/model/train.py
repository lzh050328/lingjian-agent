import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


# ====== 运输订单优先级预测模型训练 ======
# 该模块用于训练一个XGBoost分类器，预测运输订单的优先级
# 优先级用于指导调度系统决定订单处理的先后顺序

# 读取训练数据
df = pd.read_csv("../data/train_data.csv")

# 定义特征列
features = [
    "total_weight",        # 订单总重量(kg)
    "total_volume",        # 订单总体积(m³)
    "wait_time",          # 等待时间(秒)
    "route_order_count"   # 当前线路订单数量
]

# 准备训练数据
X = df[features]  # 特征矩阵
y = df["label"]   # 标签(0:普通优先级, 1:高优先级)

# 切分训练集和测试集
# test_size=0.2 表示20%的数据用于测试，80%用于训练
# random_state=42 确保每次运行结果一致
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 初始化XGBoost分类器
model = XGBClassifier(
    n_estimators=100,     # 树的数量
    max_depth=4,          # 树的最大深度
    learning_rate=0.1,    # 学习率
    eval_metric="logloss" # 评估指标：对数损失，让模型学会输出更靠谱的概率值
)

# 训练模型
print("开始训练模型...")
model.fit(X_train, y_train)

# 在测试集上评估模型性能
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# 保存训练好的模型到文件
# 模型文件将用于后续的预测服务
joblib.dump(model, "transport_order_model.pkl")

print("模型训练完成并保存")