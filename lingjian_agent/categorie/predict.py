# predict.py
# 作用：
# 1. 加载训练好的模型
# 2. 输入一句商品名
# 3. 输出预测类别和概率
# 4. 支持将“模型内部连续标签id”还原成“真实业务类目id”
#
# 运行方式：
# python predict.py

import json
from pathlib import Path

import torch
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# 1. 基本路径配置
# =========================
BASE_DIR = Path(__file__).resolve().parent

# 训练好的分类模型目录
MODEL_DIR = BASE_DIR / "saved_model"

# 类目文件（业务类目）
CATEGORY_FILE = BASE_DIR / "data" / "categories.json"

# 标签映射文件（训练时自动生成）
LABEL_MAP_FILE = BASE_DIR / "data" / "label_mapping.json"

# 商品名称通常较短，和训练时保持一致
MAX_LEN = 32


# =========================
# 2. 自动选择设备
# =========================
def get_device():
    """
    如果有 GPU，就使用 GPU
    如果没有 GPU，就自动使用 CPU
    """
    if torch.cuda.is_available():
        print("检测到 GPU，使用 CUDA 推理。")
        return torch.device("cuda")
    else:
        print("未检测到 GPU，使用 CPU 推理。")
        return torch.device("cpu")


# =========================
# 3. 检查文件是否存在
# =========================
def check_required_files():
    """
    启动前先检查关键文件是否存在
    """
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"训练好的模型目录不存在：{MODEL_DIR}")

    if not (MODEL_DIR / "config.json").exists():
        raise FileNotFoundError(f"模型目录中缺少 config.json：{MODEL_DIR}")

    if not CATEGORY_FILE.exists():
        raise FileNotFoundError(f"类目文件不存在：{CATEGORY_FILE}")

    if not LABEL_MAP_FILE.exists():
        raise FileNotFoundError(f"标签映射文件不存在：{LABEL_MAP_FILE}")


# =========================
# 4. 加载类目文件
# =========================
def load_categories():
    """
    读取 categories.json，并构建：
    原始业务类目id -> 类目名称

    例如：
    {
        1001: "冰箱",
        1002: "洗衣机",
        1148: "厨卫大电"
    }
    """
    with open(CATEGORY_FILE, "r", encoding="utf-8") as f:
        categories = json.load(f)

    category_map = {}
    for item in categories:
        category_map[int(item["id"])] = item["name"]

    return category_map


# =========================
# 5. 加载标签映射文件
# =========================
def load_label_mapping():
    """
    读取 label_mapping.json

    训练时保存的是：
    - label2id: 原始业务类目id -> 模型内部连续标签id
    - id2label: 模型内部连续标签id -> 原始业务类目id

    这里我们主要用 id2label：
    比如：
    147 -> 1148
    """
    with open(LABEL_MAP_FILE, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    # json 读出来 key 都是字符串，这里转成 int
    label2id = {int(k): int(v) for k, v in mapping["label2id"].items()}
    id2label = {int(k): int(v) for k, v in mapping["id2label"].items()}

    return label2id, id2label


# =========================
# 6. 定义响应模型
# =========================
class PredictResult(BaseModel):
    """
    商品类目识别预测结果
    """
    text: str = Field(description="输入的商品名称文本")
    train_label_id: int = Field(description="模型内部训练标签 ID")
    category_id: int = Field(description="真实业务类目 ID")
    category_name: str = Field(description="类目名称")
    confidence: float = Field(description="预测置信度")


# =========================
# 7. 预测函数
# =========================
def predict(text: str, model, tokenizer, id2label, category_map, device) -> PredictResult:
    """
    对单条文本进行预测

    参数：
    - text: 用户输入的商品名
    - model: 已加载的分类模型
    - tokenizer: 已加载的 tokenizer
    - id2label: 模型内部连续标签 id -> 原始业务类目 id
    - category_map: 原始业务类目 id -> 类目名称
    - device: cuda 或 cpu

    返回：
    PredictResult 对象，包含：
    - text: 输入文本
    - train_label_id: 模型训练标签 ID
    - category_id: 业务类目 ID
    - category_name: 业务类目名称
    - confidence: 置信度
    """
    model.eval()

    # tokenizer 把文本转成模型能读的输入
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    # 移动到设备
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # 推理阶段不需要计算梯度
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    logits = outputs.logits

    # softmax 把原始分数转换成概率
    probs = torch.softmax(logits, dim=1)

    # 模型预测出的“内部连续标签id”
    pred_train_label_id = torch.argmax(probs, dim=1).item()

    # 该标签对应的概率
    pred_prob = probs[0][pred_train_label_id].item()

    # 将模型内部连续标签id还原成“真实业务类目id”
    raw_category_id = id2label[pred_train_label_id]

    # 再根据业务类目id找到类目名称
    raw_category_name = category_map.get(raw_category_id, "未知类目")

    return PredictResult(
        text=text,
        train_label_id=pred_train_label_id,
        category_id=raw_category_id,
        category_name=raw_category_name,
        confidence=pred_prob
    )


# =========================
# 8. 主函数
# =========================
def main():
    # 启动前先检查文件
    check_required_files()

    device = get_device()

    # 读取类目和标签映射
    category_map = load_categories()
    label2id, id2label = load_label_mapping()

    print(f"类目数量：{len(category_map)}")
    print(f"标签映射数量：{len(id2label)}")

    # 加载训练好的 tokenizer 和 model
    print(f"正在加载模型目录：{MODEL_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True
    )

    model.to(device)

    print("模型加载完成。")
    print("输入商品名称后按回车即可预测，输入 q 退出。")

    while True:
        text = input("\n请输入商品名称：").strip()

        if text.lower() == "q":
            print("已退出。")
            break

        if not text:
            print("输入不能为空，请重新输入。")
            continue

        result = predict(
            text=text,
            model=model,
            tokenizer=tokenizer,
            id2label=id2label,
            category_map=category_map,
            device=device
        )

        print("\n预测结果：")
        print(f"输入文本：{result.text}")
        print(f"模型训练标签ID：{result.train_label_id}")
        print(f"业务类目ID：{result.category_id}")
        print(f"业务类目名称：{result.category_name}")
        print(f"置信度：{result.confidence:.4f}")


if __name__ == "__main__":
    main()