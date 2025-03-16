import os
import torch
import argparse
from transformers import LlamaForCausalLM, AutoTokenizer
from datasets import load_dataset, DatasetDict , load_from_disk
from transformers import TrainingArguments, Trainer

# ✅ 1. **解析命令行参数**
parser = argparse.ArgumentParser(description="Load and fine-tune LLaMA model.")
parser.add_argument("--model_dir", type=str, required=True, help="Path to the model directory.")
parser.add_argument("--model_name", type=str, default="AICrossSim/clm-60m", help="Hugging Face model name.")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ 2. **定义路径**
model_dir = args.model_dir
model_name = args.model_name
dataset_path = os.path.join(model_dir, "wikitext-2-v1")  # ✅ 数据集目录
model_path = os.path.join(model_dir, "model.safetensors")
tokenizer_path = os.path.join(model_dir, "tokenizer.json")

# ✅ 3. **确保模型权重存在**
if not os.path.exists(model_path):
    print(f"🔍 Model not found locally in {model_dir}. Downloading from Hugging Face...")
    model = LlamaForCausalLM.from_pretrained(model_name).to(device)
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir, safe_serialization=True)
else:
    print(f"✅ Model found locally in {model_dir}. Loading...")
    model = LlamaForCausalLM.from_pretrained(model_dir).to(device)

# ✅ 4. **确保 Tokenizer 存在**
if not os.path.exists(tokenizer_path):
    print(f"🔍 Tokenizer not found locally in {model_dir}. Downloading from Hugging Face...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(model_dir)
else:
    print(f"✅ Tokenizer found locally in {model_dir}. Loading...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

# ✅ 5. **确保 Tokenizer 有 `pad_token`**
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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

    # ✅ **将数据集保存到本地**
    dataset.save_to_disk(dataset_path)
    print(f"✅ 数据集已下载并保存至 {dataset_path}")

print(f"📂 数据集加载成功：{dataset}")

print("🚀 Model, Tokenizer, and Dataset are ready!")


# **查看数据结构**
print("样本数据:", dataset["train"][:5])  # 打印前 5 条

# ✅ 3. **Tokenization**
def tokenize_function(examples):
    tokens = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()  # 添加 `labels`
    return tokens


tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# ✅ 4. **训练参数**
training_args = TrainingArguments(
    output_dir=model_dir,
    per_device_train_batch_size=4,  # 根据显存调整
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir="E:/MHA2HLA/logs",
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    fp16=torch.cuda.is_available(),
    push_to_hub=False,
    report_to="none",
)

# ✅ 5. **训练模型**
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)

trainer.train()

# ✅ 6. **保存模型**
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
print("✅ 模型已保存至:", model_dir)


try:
    from lm_eval.evaluator import simple_evaluate

    results = simple_evaluate(
        model=model,
        tasks=["wikitext"],
        batch_size=4,
    )
    print("📊 Evaluation Results:", results)
except Exception as e:
    print(f"❌ 评估失败: {e}")


print("🎉 训练完成！🚀")