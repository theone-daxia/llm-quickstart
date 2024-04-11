from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from transformers import Trainer, TrainingArguments
from peft import TaskType, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from typing import List, Dict
import torch

model_name_or_path = "/data/ai/models/llm/chatglm3-6b-b098244"
data_path = "/data/ai/dataset/adgen"
eval_data_path = None  # 验证数据路径，如果没有则设置为None
seed = 8  # 随机种子
max_input_length = 512  # 输入的最大长度
max_output_length = 1536  # 输出的最大长度
lora_rank = 4  # LoRA 秩
lora_alpha = 32  # LoRA alpha 值
lora_dropout = 0.05  # LoRA Dropout 率
resume_from_checkpoint = None  # 从指定路径的 checkpoint 恢复训练
prompt_text = ""  # 所有数据前的指令文本
compute_dtype = "fp32"  # 计算数据类型（fp32, fp16, bf16）


def tokenize_func(example, tokenizer, ignore_label_id=-100):
    """
    对单个数据样本进行tokenize处理。

    参数：
    example(dict): 单条数据样本。
    tokenizer(transformers.PreTrainedTokenizer): 用于 tokenize 文本的 tokenizer。
    ignore_label_id(int, optional): 在 label 中用于填充的忽略ID，默认为 -100。

    返回：
    dict: 包含 tokenized_input_ids 和 labels 的字典，用于模型训练。
    """

    # 构建问题文本
    question = prompt_text + example["content"]
    if example.get("input", None) and example["input"].strip():
        question += f"\n{example['input']}"

    # 构建答案文本
    answer = example["summary"]

    # 对问题和答案文本进行 tokenize 处理
    q_ids = tokenizer.encode(text=question, add_special_tokens=False)
    a_ids = tokenizer.encode(text=answer, add_special_tokens=False)

    # tokenize 后的长度超过限制，则进行截断
    if len(q_ids) > max_input_length - 2:
        q_ids = q_ids[:max_input_length - 2]  # 保留空间给 gmask 和 bos 标记
    if len(a_ids) > max_output_length - 1:
        a_ids = a_ids[:max_output_length - 1]  # 保留空间给 eos 标记

    # 构建模型的输入格式
    input_ids = tokenizer.build_inputs_with_special_tokens(q_ids, a_ids)
    question_length = len(q_ids) + 2  # 加上 gmask 和 bos 标记

    # 构建标签，对于问题部分的输入使用 ignore_label_id 填充
    labels = [ignore_label_id] * question_length + input_ids[question_length:]

    return {"input_ids": input_ids, "labels": labels}


tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, revision='b098244')
dataset = load_dataset(data_path)
column_names = dataset["train"].column_names
tokenized_dataset = dataset["train"].map(
    lambda example: tokenize_func(example, tokenizer),
    batched=False,  # 注意这里，True时：examples 是个 Dict[str, List]；False时：example 是个 Dict[str, Any]
    remove_columns=column_names,
)

# shuffle 会将数据集的索引列表打乱，以创建一个新的索引映射。
# 一旦数据集具有索引映射，速度可能会变慢 10 倍。因为需要额外的步骤来使用索引映射获取要读取的行索引，并且最重要的是，不再连续地读取数据块。
# 要恢复速度，需要再次使用 Dataset.flatten_indices() 将整个数据集重新写入磁盘，从而删除索引映射。
tokenized_dataset = tokenized_dataset.shuffle(seed=seed)
tokenized_dataset = tokenized_dataset.flatten_indices()


# 定义 DataCollatorForChatGLM 类，批量处理数据
class DataCollatorForChatGLM:
    """
    用于批量处理数据的 DataCollator，尤其是在使用 ChatGLM 模型时。

    该类负责将多个数据样本（tokenized input）合并为一个批量，并在必要时进行填充（padding）。

    属性：
    pad_token_id(int): 用于填充的 token ID。
    max_length(int): 单个批量数据的最大长度限制。
    ignore_label_id(int): 在 label 中用于填充的忽略 ID。
    """

    def __init__(self, pad_token_id: int, max_length: int = 2048, ignore_label_id: int = -100):
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.ignore_label_id = ignore_label_id

    def __call__(self, batch_data: List[Dict[str, List]]) -> Dict[str, torch.Tensor]:
        """
        批量处理数据。

        参数：
        batch_data(List[Dict[str, List]]): 包含多个数据样本的字典列表。

        返回：
        Dict[str, torch.Tensor]: 包含处理后的批量数据的字典。
        """

        # 计算批量中每个样本的长度
        len_list = [len(d["input_ids"]) for d in batch_data]
        batch_max_len = max(len_list)  # 找到最长的样本长度

        input_ids, labels = [], []
        for len_of_d, d in sorted(zip(len_list, batch_data), key=lambda x: -x[0]):
            pad_len = batch_max_len - len_of_d  # 计算需要填充的长度
            # 添加填充，并确保数据长度不超过最大长度限制
            ids = d["input_ids"] + [self.pad_token_id] * pad_len
            label = d["labels"] + [self.ignore_label_id] * pad_len
            if batch_max_len > self.max_length:
                ids = ids[:self.max_length]
                label = label[:self.max_length]
            input_ids.append(torch.LongTensor(ids))
            labels.append(torch.LongTensor(label))

        # 将处理后的数据堆叠成一个 tensor
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)

        return {"input_ids": input_ids, "labels": labels}


# 准备数据整理器
data_collator = DataCollatorForChatGLM(pad_token_id=tokenizer.pad_token_id)

# 训练模型
# 使用 nf4 量化数据类型加载模型，开启双量化配置，以 bf16 混合精度训练，预估显存占用接近 4GB
_compute_dtype_map = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16
}

# QLoRA 量化配置
q_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=_compute_dtype_map["bf16"]
)

# 加载模型
# revision="b098244"，版本对应的 ChatGLM3-6B 设置 use_reentrant=False
# 最新版本 use_reentrant 被设置为 True，会增加不必要的显存开销
model = AutoModel.from_pretrained(
    model_name_or_path,
    quantization_config=q_config,
    device_map="auto",
    trust_remote_code=True,
    revision="b098244"
)

# 获取当前模型占用的 GPU 显存（差值为预留给 PyTorch 的显存）
memory_footprint_bytes = model.get_memory_footprint()
memory_footprint_mib = memory_footprint_bytes / (1024 ** 2)  # 转换为 MiB

print(f"{memory_footprint_mib:.2f}MiB")

# 预处理量化后的模型，使其可以支持低精度微调训练
kbit_model = prepare_model_for_kbit_training(model)

# peft 适配模块设置
# 在 PEFT 库的 constants.py 文件中定义了不同的 PEFT 方法，在各类大模型上的微调适配模块。
# 通常名称相同的模型架构也类似，应用微调方法时的适配器设置也几乎一致。
target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING["chatglm"]
print("target_modules:", target_modules)

# LoRA 适配器配置
lora_config = LoraConfig(
    target_modules=target_modules,
    r=lora_rank,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias="none",
    inference_mode=False,
    task_type=TaskType.CAUSAL_LM
)
qlora_model = get_peft_model(kbit_model, lora_config)
qlora_model.print_trainable_parameters()

# 训练超参数配置
training_args = TrainingArguments(
    output_dir=f"{model_name_or_path}-qlora-adgen",
    per_device_train_batch_size=16,  # 每个设备的训练批量大小
    gradient_accumulation_steps=4,  # 梯度累积步数
    # per_device_eval_batch_size=8,  # 每个设备的评估批量大小
    learning_rate=1e-3,  # 学习率
    num_train_epochs=1,  # 训练轮数
    lr_scheduler_type="linear",  # 学习率调度器类型
    warmup_ratio=0.1,  # 预热比例
    logging_steps=10,  # 日志记录步数
    save_strategy="steps",  # 模型保存策略
    save_steps=100,  # 模型保存步数
    # evaluation_strategy="steps",  # 评估策略
    # eval_steps=500,  # 评估步数
    optim="adamw_torch",  # 优化器类型
    fp16=True,  # 是否使用混合精度训练
)
trainer = Trainer(
    model=qlora_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)
trainer.train(resume_from_checkpoint=True)

trainer.model.save_pretrained(f"{model_name_or_path}-qlora-adgen")
