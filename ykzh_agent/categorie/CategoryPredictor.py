# 作用：
# 1. 加载训练好的模型
# 2. 输入一句商品名
# 3. 输出预测类别和概率
# 4. 支持将"模型内部连续标签id"还原成"真实业务类目id"

import json
from pathlib import Path

import torch
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ykzh_config import logger

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
# 2. 定义响应模型
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
# 3. 类目预测器类
# =========================
class CategoryPredictor:
    """
    商品类目预测器

    该类封装了BERT模型的加载、文本预测等功能，
    提供简洁的API接口用于商品类目预测
    """

    def __init__(self, model_dir: str = None, category_file: str = None, label_map_file: str = None):
        """
        初始化预测器

        Args:
            model_dir: 模型目录路径，默认为 saved_model
            category_file: 类目文件路径，默认为 data/categories.json
            label_map_file: 标签映射文件路径，默认为 data/label_mapping.json
        """
        # 允许自定义路径
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self.category_file = Path(category_file) if category_file else CATEGORY_FILE
        self.label_map_file = Path(label_map_file) if label_map_file else LABEL_MAP_FILE

        # 设备选择
        self.device = self._get_device()

        # 检查必要文件
        self._check_required_files()

        # 加载类目映射和标签映射
        self.category_map = self._load_categories()
        self.label2id, self.id2label = self._load_label_mapping()

        # 加载模型和分词器
        self.tokenizer, self.model = self._load_model()

        logger.info(f"类目预测器初始化完成，支持 {len(self.category_map)} 个类目")

    def _get_device(self):
        """
        自动选择设备（GPU或CPU）

        Returns:
            torch.device: 选择的设备
        """
        if torch.cuda.is_available():
            logger.info("检测到 GPU，使用 CUDA 推理。")
            return torch.device("cuda")
        else:
            logger.info("未检测到 GPU，使用 CPU 推理。")
            return torch.device("cpu")

    def _check_required_files(self):
        """
        检查必要文件是否存在

        Raises:
            FileNotFoundError: 当必要文件不存在时抛出异常
        """
        if not self.model_dir.exists():
            raise FileNotFoundError(f"训练好的模型目录不存在：{self.model_dir}")

        if not (self.model_dir / "config.json").exists():
            raise FileNotFoundError(f"模型目录中缺少 config.json：{self.model_dir}")

        if not self.category_file.exists():
            raise FileNotFoundError(f"类目文件不存在：{self.category_file}")

        if not self.label_map_file.exists():
            raise FileNotFoundError(f"标签映射文件不存在：{self.label_map_file}")

    def _load_categories(self):
        """
        读取 categories.json，构建原始业务类目id -> 类目名称的映射

        Returns:
            dict: 类目映射字典 {category_id: category_name}

        Example:
            {
                1001: "冰箱",
                1002: "洗衣机",
                1148: "厨卫大电"
            }
        """
        with open(self.category_file, "r", encoding="utf-8") as f:
            categories = json.load(f)

        category_map = {}
        for item in categories:
            category_map[int(item["id"])] = item["name"]

        return category_map

    def _load_label_mapping(self):
        """
        读取 label_mapping.json

        训练时保存的是：
        - label2id: 原始业务类目id -> 模型内部连续标签id
        - id2label: 模型内部连续标签id -> 原始业务类目id

        Returns:
            tuple: (label2id, id2label) 两个映射字典
        """
        with open(self.label_map_file, "r", encoding="utf-8") as f:
            mapping = json.load(f)

        # json 读出来 key 都是字符串，这里转成 int
        label2id = {int(k): int(v) for k, v in mapping["label2id"].items()}
        id2label = {int(k): int(v) for k, v in mapping["id2label"].items()}

        return label2id, id2label

    def _load_model(self):
        """
        加载训练好的 tokenizer 和 model

        Returns:
            tuple: (tokenizer, model)
        """
        logger.info(f"正在加载模型目录：{self.model_dir}")

        tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir),
            local_files_only=True
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_dir),
            local_files_only=True
        )

        model.to(self.device)
        logger.info("模型加载完成。")

        return tokenizer, model

    def predict(self, text: str) -> PredictResult:
        """
        对单条文本进行预测

        Args:
            text: 用户输入的商品名

        Returns:
            PredictResult: 预测结果对象，包含：
                - text: 输入文本
                - train_label_id: 模型训练标签 ID
                - category_id: 业务类目 ID
                - category_name: 业务类目名称
                - confidence: 置信度
        """
        # 输入验证
        if not text or not text.strip():
            raise ValueError("输入文本不能为空")

        self.model.eval()

        # tokenizer 把文本转成模型能读的输入
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        # 移动到设备
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # 推理阶段不需要计算梯度
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        logits = outputs.logits

        # softmax 把原始分数转换成概率
        probs = torch.softmax(logits, dim=1)

        # 模型预测出的"内部连续标签id"
        pred_train_label_id = torch.argmax(probs, dim=1).item()

        # 该标签对应的概率
        pred_prob = probs[0][pred_train_label_id].item()

        # 将模型内部连续标签id还原成"真实业务类目id"
        raw_category_id = self.id2label[pred_train_label_id]

        # 再根据业务类目id找到类目名称
        raw_category_name = self.category_map.get(raw_category_id, "未知类目")

        return PredictResult(
            text=text,
            train_label_id=pred_train_label_id,
            category_id=raw_category_id,
            category_name=raw_category_name,
            confidence=pred_prob
        )

    def get_category_info(self, category_id: int) -> dict:
        """
        根据类目ID获取类目信息

        Args:
            category_id: 业务类目ID

        Returns:
            dict: 类目信息，包含id和name
        """
        return {
            "id": category_id,
            "name": self.category_map.get(category_id, "未知类目")
        }

    def get_all_categories(self) -> dict:
        """
        获取所有类目信息

        Returns:
            dict: 所有类目映射 {category_id: category_name}
        """
        return self.category_map.copy()


# 创建全局预测器实例
category_predictor = CategoryPredictor()
