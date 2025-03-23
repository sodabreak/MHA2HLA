import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset, DatasetDict, load_from_disk
from collections import defaultdict
from modeling_llama import LlamaForCausalLM ,LlamaConfig # ✅ 从自定义 LLaMA 结构导入
import torch
from transformers import EarlyStoppingCallback
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# ✅ 1. **解析命令行参数**
parser = argparse.ArgumentParser(description="Load and fine-tune LLaMA model.")
parser.add_argument("--model_dir", type=str, required=True, help="Path to the model directory.")
parser.add_argument("--model_name", type=str, default="AICrossSim/clm-60m", help="Hugging Face model name.")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and evaluation.")
parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ 2. **定义路径**
model_dir = args.model_dir
model_name = args.model_name
dataset_path = os.path.join(model_dir, "wikitext-2-v1")
model_path = os.path.join(model_dir, "model.safetensors")
tokenizer_path = os.path.join(model_dir, "tokenizer.json")

# ✅ 3. **确保模型权重存在**
print(f"🔄 下载模型和 tokenizer：{model_name}")

config = LlamaConfig.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name, config=config).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# ✅ 5. **确保 Tokenizer 有 `pad_token`**
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

os.makedirs(model_dir, exist_ok=True)
model.save_pretrained(model_dir, safe_serialization=True)
tokenizer.save_pretrained(model_dir)


# ✅ 6. **检查数据集是否存在**
expected_splits = ["train", "validation", "test"]
if all(os.path.exists(os.path.join(dataset_path, split)) for split in expected_splits):
    print(f"📂 发现本地数据集 {dataset_path}，正在加载...")
    dataset = DatasetDict({
        "train": load_from_disk(os.path.join(dataset_path, "train")),
        "validation": load_from_disk(os.path.join(dataset_path, "validation")),
        "test": load_from_disk(os.path.join(dataset_path, "test"))
    })
else:
    print("🔍 数据集不存在，正在从 Hugging Face 下载 `wikitext-2-v1`...")
    dataset = load_dataset("wikitext", "wikitext-2-v1")
    dataset.save_to_disk(dataset_path)
    print(f"✅ 数据集已下载并保存至 {dataset_path}")

print(f"📂 数据集加载成功：{dataset}")

# ✅ 7. **Tokenization**
def tokenize_function(examples):
    tokens = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# ✅ 8. **训练参数**
training_args = TrainingArguments(
    output_dir=model_dir,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir=os.path.join(model_dir, "logs"),
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    fp16=torch.cuda.is_available(),
    push_to_hub=False,
    report_to="none",
    logging_first_step=True,
    load_best_model_at_end=True,            # 👉 加载验证集最优模型
    metric_for_best_model="eval_loss",      # 👉 以 eval_loss 为判断标准
    greater_is_better=False,                # 👉 eval_loss 越低越好
    save_total_limit=3,                     # 👉 只保留一个最佳模型
)

# ✅ 9. **训练模型**
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

train_result = trainer.train()

log_history = trainer.state.log_history

# 计算每个 epoch 的平均 Train Loss
epoch_loss = defaultdict(list)
epoch_eval_loss = {}

for entry in log_history:
    if "loss" in entry and "epoch" in entry:
        epoch_loss[entry["epoch"]].append(entry["loss"])
    if "eval_loss" in entry and "epoch" in entry:
        epoch_eval_loss[entry["epoch"]] = entry["eval_loss"]  # 只取最后一次评估 loss

# 计算每个 epoch 的平均 loss
epochs = sorted(epoch_loss.keys())
train_loss_per_epoch = [sum(losses) / len(losses) for losses in epoch_loss.values()]

# 获取验证集 loss
eval_epochs = sorted(epoch_eval_loss.keys())
eval_loss_per_epoch = [epoch_eval_loss[epoch] for epoch in eval_epochs]

# 画图
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss_per_epoch, label="Train Loss", linestyle="--", marker="o")
plt.plot(eval_epochs, eval_loss_per_epoch, label="Validation Loss", linestyle="-", marker="s")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curve")
plt.legend()
plt.grid()
plt.savefig(os.path.join(model_dir, "loss_curve_epoch.png"))
plt.show()

# ✅ 11. **保存模型**
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
print(f"✅ 训练完成！模型已保存至: {model_dir}")

# ✅ 12. **测试集推理**
print("🔍 开始在 `test` 数据集上进行推理...")
test_results = trainer.predict(tokenized_datasets["test"])

# 解析 `test` 结果
test_preds = torch.argmax(torch.tensor(test_results.predictions), dim=-1)
test_texts = [tokenizer.decode(pred, skip_special_tokens=True) for pred in test_preds]

# ✅ 13. **保存 `test` 输入和输出到 Excel**
test_df = pd.DataFrame({
    "input_text": dataset["test"]["text"],
    "predicted_text": test_texts
})

excel_path = os.path.join(model_dir, "test_results.xlsx")
test_df.to_excel(excel_path, index=False, encoding="utf-8")
print(f"✅ 测试集结果已保存至 {excel_path}")

# ✅ 14. **尝试评估**
try:
    from lm_eval.evaluator import simple_evaluate
    results = simple_evaluate(
        model=model,
        tasks=["wikitext"],
        batch_size=args.batch_size,
    )
    print("📊 Evaluation Results:", results)
except Exception as e:
    print(f"❌ 评估失败: {e}")

print("🎉 训练和测试完成！🚀")

