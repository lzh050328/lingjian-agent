# train_bert.py
# 教学版：使用 bert-base-chinese 做商品分类
#
# 这版功能：
# 1. 如果本地已经有训练好的模型，就直接加载，不再重复训练
# 2. 如果本地没有模型，才开始训练
# 3. 如果你想强制重新训练，也可以手动打开开关
# 4. 自动读取 categories.json，并建立“业务类目id -> 模型训练标签id”的映射
# 5. 自动把 train_data.jsonl 里的原始 label 转成连续标签
# 6. 保存 label_mapping.json，供预测脚本使用
#
# 运行方式：
# python train_bert.py

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# =========================
# 1. 基本配置
# =========================
BASE_DIR = Path(__file__).resolve().parent

# 训练数据文件：每行一个 json，例如：
# {"text": "洗碗机", "label": 1148}
TRAIN_FILE = BASE_DIR / "data" / "train_data.jsonl"

# 类目文件：例如：
# [{"id": 1001, "name": "冰箱"}, {"id": 1148, "name": "厨卫大电"}]
CATEGORY_FILE = BASE_DIR / "data" / "categories.json"

# 标签映射文件：训练时自动生成，预测时要用
LABEL_MAP_FILE = BASE_DIR / "data" / "label_mapping.json"

# 已训练好的分类模型保存目录
MODEL_DIR = BASE_DIR / "saved_model"

# 本地预训练模型目录（你已下载到本地）
MODEL_NAME = BASE_DIR / "pretrained_models" / "bert-base-chinese"

# 商品名通常比较短，所以 32 基本够用
MAX_LEN = 32

# CPU 训练时批大小不要太大，否则会慢
BATCH_SIZE = 8

# 训练轮数
EPOCHS = 3

# 学习率
LR = 2e-5

# 随机种子
SEED = 42

# 是否强制重新训练
# False：如果本地已有模型，则直接加载
# True：即使本地已有模型，也重新训练
FORCE_RETRAIN = False


# =========================
# 2. 设置随机种子
# =========================
def set_seed(seed: int = 42):
    """
    固定随机种子，尽量保证每次训练结果更稳定、可复现
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# 3. 自动选择设备
# =========================
def get_device():
    """
    如果有 GPU，就使用 GPU
    如果没有 GPU，就自动使用 CPU
    """
    if torch.cuda.is_available():
        print("检测到 GPU，使用 CUDA 训练。")
        return torch.device("cuda")
    else:
        print("未检测到 GPU，使用 CPU 训练。")
        return torch.device("cpu")


# =========================
# 4. 判断本地模型是否存在
# =========================
def model_exists(model_dir: Path) -> bool:
    """
    Hugging Face 保存后的模型目录里，通常至少会有：
    - config.json
    - model.safetensors 或 pytorch_model.bin
    - tokenizer 相关文件

    这里做一个简单判断：
    只要 config.json 存在，就认为这是一个可加载的模型目录
    """
    return (model_dir / "config.json").exists()


# =========================
# 5. 读取 jsonl 文件
# =========================
def load_jsonl(file_path: Path):
    """
    读取 jsonl 文件
    每一行是一个 json 对象
    """
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# =========================
# 6. 读取类目文件
# =========================
def load_categories(category_file: Path):
    """
    读取 categories.json

    例如：
    [
      {"id": 1001, "name": "冰箱"},
      {"id": 1002, "name": "洗衣机"}
    ]
    """
    with open(category_file, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# 7. 构建标签映射
# =========================
def build_label_mapping(categories):
    """
    根据 categories.json 构建映射：

    业务类目 id        -> 模型训练标签 id
    例如：
    1001             -> 0
    1002             -> 1
    1148             -> 147

    以及反向映射：
    模型训练标签 id    -> 业务类目 id
    例如：
    0                -> 1001
    1                -> 1002
    147              -> 1148

    为什么要这样做？
    因为模型训练时，CrossEntropyLoss 要求 label 必须是 0 ~ num_labels-1 的连续整数。
    """
    raw_ids = [item["id"] for item in categories]

    label2id = {raw_id: idx for idx, raw_id in enumerate(raw_ids)}
    id2label = {idx: raw_id for raw_id, idx in label2id.items()}

    return label2id, id2label


# =========================
# 8. 编码训练数据中的标签
# =========================
def encode_train_data(data, label2id):
    """
    将 train_data.jsonl 中的原始业务类目 id，转换为模型训练所需的连续标签

    输入样例：
    {"text": "洗碗机", "label": 1148}

    输出样例：
    {"text": "洗碗机", "label": 147}
    """
    new_data = []

    for item in data:
        raw_label = item["label"]

        if raw_label not in label2id:
            raise ValueError(
                f"训练数据中的 label={raw_label} 不在 categories.json 中，请检查数据。"
            )

        new_item = item.copy()
        new_item["label"] = label2id[raw_label]
        new_data.append(new_item)

    return new_data


# =========================
# 9. 保存标签映射文件
# =========================
def save_label_mapping(label2id, id2label, save_path: Path):
    """
    保存标签映射，供 predict.py 使用

    注意：
    json 的 key 只能是字符串，所以这里保存时会转成字符串
    预测脚本读取后再转回 int 即可
    """
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "label2id": {str(k): v for k, v in label2id.items()},
                "id2label": {str(k): v for k, v in id2label.items()}
            },
            f,
            ensure_ascii=False,
            indent=2
        )


# =========================
# 10. 自定义数据集
# =========================
class GoodsDataset(Dataset):
    """
    自定义数据集：
    把每条样本中的文本 text 和标签 label 转成模型可以训练的格式
    """
    def __init__(self, data, tokenizer, max_len=32):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        label = item["label"]

        # tokenizer 把文本转成模型输入格式
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "text": text,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


# =========================
# 11. 训练一个 epoch
# =========================
def train_one_epoch(model, dataloader, optimizer, device):
    """
    训练一轮
    """
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


# =========================
# 12. 验证一个 epoch
# =========================
def eval_one_epoch(model, dataloader, device):
    """
    验证一轮
    """
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


# =========================
# 13. 加载已有模型
# =========================
def load_local_model(model_dir: Path, device):
    """
    从本地目录加载已经训练好的模型和 tokenizer
    """
    print(f"发现本地已有模型，直接加载：{model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), local_files_only=True)
    model.to(device)

    return tokenizer, model


# =========================
# 14. 新建模型
# =========================
def build_new_model(num_labels: int, device):
    """
    从本地 bert-base-chinese 初始化一个待训练模型
    """

    print("准备加载本地预训练模型目录：", MODEL_NAME)
    print("目录是否存在：", MODEL_NAME.exists())

    if not MODEL_NAME.exists():
        raise FileNotFoundError(f"本地模型目录不存在：{MODEL_NAME}")

    # 额外检查几个关键文件，方便排错
    for file_name in ["config.json", "vocab.txt"]:
        file_path = MODEL_NAME / file_name
        print(f"{file_name} 是否存在：{file_path.exists()}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_NAME),
        local_files_only=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_NAME),
        num_labels=num_labels,
        local_files_only=True
    )

    model.to(device)
    return tokenizer, model


# =========================
# 15. 训练并保存模型
# =========================
def train_and_save_model(model, tokenizer, train_loader, val_loader, device):
    """
    训练模型，并保存验证集效果最好的版本
    """
    optimizer = AdamW(model.parameters(), lr=LR)
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        print(f"\n===== 第 {epoch + 1}/{EPOCHS} 轮训练 =====")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, device)

        print(f"训练集 loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        print(f"验证集 loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        # 保存验证集上效果最好的模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            MODEL_DIR.mkdir(exist_ok=True)

            # save_pretrained 是 Hugging Face 的标准保存方式
            # 保存后可用 from_pretrained(MODEL_DIR) 再加载
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)

            print(f"发现更优模型，已保存到：{MODEL_DIR}")

    print("\n训练完成。")
    print(f"最佳验证集准确率：{best_val_acc:.4f}")


# =========================
# 16. 主函数
# =========================
def main():
    set_seed(SEED)
    device = get_device()

    # ---------- 1）读取训练数据 ----------
    raw_data = load_jsonl(TRAIN_FILE)
    print(f"总样本数：{len(raw_data)}")

    # ---------- 2）读取类目文件 ----------
    categories = load_categories(CATEGORY_FILE)
    print(f"类目文件中的类目数：{len(categories)}")

    # ---------- 3）构建标签映射 ----------
    label2id, id2label = build_label_mapping(categories)

    # 保存映射文件，后续 predict.py 会使用
    save_label_mapping(label2id, id2label, LABEL_MAP_FILE)
    print(f"标签映射已保存到：{LABEL_MAP_FILE}")

    # ---------- 4）把训练数据中的原始 label 转成连续标签 ----------
    data = encode_train_data(raw_data, label2id)

    labels = [x["label"] for x in data]
    num_labels = len(set(labels))

    print(f"类别数：{num_labels}")
    print(f"训练标签范围：{min(labels)} ~ {max(labels)}")

    # 情况 1：如果本地模型已存在，并且没有要求强制重训
    # 那么就直接加载模型，不再训练
    if model_exists(MODEL_DIR) and not FORCE_RETRAIN:
        tokenizer, model = load_local_model(MODEL_DIR, device)
        print("已跳过训练，后续可直接用于预测。")
        return

    # 情况 2：如果没有本地模型，或者要求强制重训
    # 则继续走训练流程
    train_data, val_data = train_test_split(
        data,
        test_size=0.2,
        random_state=SEED,
        stratify=labels
    )

    print(f"训练集：{len(train_data)}")
    print(f"验证集：{len(val_data)}")

    tokenizer, model = build_new_model(num_labels, device)

    train_dataset = GoodsDataset(train_data, tokenizer, max_len=MAX_LEN)
    val_dataset = GoodsDataset(val_data, tokenizer, max_len=MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    train_and_save_model(model, tokenizer, train_loader, val_loader, device)


if __name__ == "__main__":
    main()