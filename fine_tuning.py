import os
import torch
from transformers import LlamaForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import google.protobuf
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Device Count: {torch.cuda.device_count()}")
print(f"Current CUDA Device: {torch.cuda.current_device()}" if torch.cuda.is_available() else "No GPU detected")
print(f"GPU Name: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "No GPU detected")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

# ✅ 0. **避免 protobuf 依赖问题**
#os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# ✅ 1. **确保模型权重和 tokenizer 存在**
model_dir = "E:/MHA2HLA/clm-60m-finetuned"
model_name = "AICrossSim/clm-60m"

if not os.path.exists(model_dir):
    print("🔍 Model not found locally. Downloading from Hugging Face...")
    model = LlamaForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
else:
    print("✅ Model found locally. Loading...")
    model = LlamaForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

# **确保 Tokenizer 有 `pad_token`**
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ✅ 2. **加载 Parquet 数据**
dataset_path = "E:/MHA2HLA/clm-60m-finetuned/wikitext-2-v1"

dataset = load_dataset("parquet", data_files={
    "train": f"{dataset_path}/train-00000-of-00001.parquet",
    "validation": f"{dataset_path}/validation-00000-of-00001.parquet",
    "test": f"{dataset_path}/test-00000-of-00001.parquet",
})

print("📂 数据集信息:", dataset)

# **查看数据结构**
print("🔎 样本数据:", dataset["train"][0])

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
'''
# ✅ 7. **评估**
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
'''
# ✅ 8. **上传 Hugging Face（可选）**
upload_to_hf = False  # 如果需要上传，改成 True
if upload_to_hf:
    os.system(f"huggingface-cli upload {model_dir} --repo AICrossSim/clm-60m-finetuned")

print("🎉 训练完成！🚀")
