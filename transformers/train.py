from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import TrainerCallback
import numpy as np
import evaluate
import re
import logging


# 配置日志格式
logging.basicConfig(
    filename='/data/ai/projects/try/train.log',  # 指定日志文件路径
    filemode='a',  # 追加模式，即在已有日志文件的末尾追加新的日志
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    level=logging.INFO  # 日志级别
)

model = AutoModelForSequenceClassification.from_pretrained("/data/ai/models/llm/bert-base-cased", num_labels=5)
tokenizer = AutoTokenizer.from_pretrained("/data/ai/models/llm/bert-base-cased")
dataset = load_dataset("/data/ai/dataset/yelp_review_full")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42)
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42)

print("map done")

# 评估指标
metric = evaluate.load("accuracy")

print("load metric done")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# 训练过程指标监控
# logging_steps 默认值为500，根据我们的训练数据和步长，将其设置为100
model_dir = "/data/ai/models/bert-base-cased-finetune-yelp"
training_args = TrainingArguments(output_dir=model_dir,
                                  evaluation_strategy="epoch",
                                  per_device_train_batch_size=32,
                                  num_train_epochs=3,
                                  logging_steps=500)

# 完整的超参数配置
logging.info("training_args: %s", training_args)


class LogCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        values = {"Epoch": int(state.epoch), "Training Loss": "No log", "Validation Loss": "No log"}
        for log in reversed(state.log_history):
            if "loss" in log:
                values["Training Loss"] = log["loss"]
                break

        metric_key_prefix = "eval"
        for k in metrics:
            if k.endswith("_loss"):
                metric_key_prefix = re.sub(r"\_loss$", "", k)
        _ = metrics.pop("total_flos", None)
        _ = metrics.pop("epoch", None)
        _ = metrics.pop(f"{metric_key_prefix}_runtime", None)
        _ = metrics.pop(f"{metric_key_prefix}_samples_per_second", None)
        _ = metrics.pop(f"{metric_key_prefix}_steps_per_second", None)
        _ = metrics.pop(f"{metric_key_prefix}_jit_compilation_time", None)
        for k, v in metrics.items():
            if k == f"{metric_key_prefix}_loss":
                values["Validation Loss"] = v
            else:
                splits = k.split("_")
                name = " ".join([part.capitalize() for part in splits[1:]])
                values[name] = v

        logging.info(values)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[LogCallback],
)

trainer.train()

small_test_dataset = tokenized_datasets["test"].shuffle(seed=64).select(range(100))
trainer.evaluate(small_test_dataset)

trainer.save_model(model_dir)  # 保存模型
trainer.save_state()  # 保存训练状态

