import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset, DatasetDict, load_from_disk
from collections import defaultdict
from modeling_llama import LlamaForCausalLM ,LlamaConfig  # ✅ Import from custom LLaMA implementation
import torch
from transformers import EarlyStoppingCallback

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# 1. Parse command-line arguments
parser = argparse.ArgumentParser(description="Load and fine-tune LLaMA model.")
parser.add_argument("--model_dir", type=str, required=True, help="Path to the model directory.")
parser.add_argument("--model_name", type=str, default="AICrossSim/clm-60m", help="Hugging Face model name.")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and evaluation.")
parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Define paths
model_dir = args.model_dir
model_name = args.model_name
dataset_path = os.path.join(model_dir, "wikitext-2-v1")
model_path = os.path.join(model_dir, "model.safetensors")
tokenizer_path = os.path.join(model_dir, "tokenizer.json")

# 3. Ensure model weights are available
print(f"Downloading model and tokenizer: {model_name}")

config = LlamaConfig.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name, config=config).to(device)
tokenizer_name = "HuggingFaceTB/cosmo2-tokenizer"

# Load tokenizer with error handling
def load_tokenizer_safe(tokenizer_name, cache_dir=None):
    try:
        return AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
    except json.JSONDecodeError as e:
        print(f"Tokenizer JSON loading failed: {e}")
        local_name = tokenizer_name.replace("/", "--")
        local_cache_path = os.path.expanduser(f"{cache_dir or '~/.cache/huggingface/hub'}/models--{local_name}")
        print(f"Deleting corrupted cache: {local_cache_path}")
        shutil.rmtree(local_cache_path, ignore_errors=True)
        print("Retrying to download tokenizer...")
        return AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)

tokenizer = load_tokenizer_safe(tokenizer_name)

# 5. Ensure tokenizer has a pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

os.makedirs(model_dir, exist_ok=True)
model.save_pretrained(model_dir, safe_serialization=True)
tokenizer.save_pretrained(model_dir)

# 6. Load or download dataset
expected_splits = ["train", "validation", "test"]
if all(os.path.exists(os.path.join(dataset_path, split)) for split in expected_splits):
    print(f"Found local dataset {dataset_path}, loading...")
    dataset = DatasetDict({
        "train": load_from_disk(os.path.join(dataset_path, "../train")),
        "validation": load_from_disk(os.path.join(dataset_path, "validation")),
        "test": load_from_disk(os.path.join(dataset_path, "test"))
    })
else:
    print("Dataset not found, downloading `wikitext-2-v1` from Hugging Face...")
    dataset = load_dataset("wikitext", "wikitext-2-v1")
    dataset.save_to_disk(dataset_path)
    print(f"Dataset downloaded and saved to {dataset_path}")

print(f"Dataset loaded successfully: {dataset}")

# 7. Tokenize

def tokenize_function(examples):
    tokens = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512, return_attention_mask=True)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 8. Training arguments
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
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=3,
)

# 9. Train
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

# Compute average training loss per epoch
epoch_loss = defaultdict(list)
epoch_eval_loss = {}

for entry in log_history:
    if "loss" in entry and "epoch" in entry:
        epoch_loss[entry["epoch"]].append(entry["loss"])
    if "eval_loss" in entry and "epoch" in entry:
        epoch_eval_loss[entry["epoch"]] = entry["eval_loss"]

epochs = sorted(epoch_loss.keys())
train_loss_per_epoch = [sum(losses) / len(losses) for losses in epoch_loss.values()]

eval_epochs = sorted(epoch_eval_loss.keys())
eval_loss_per_epoch = [epoch_eval_loss[epoch] for epoch in eval_epochs]

# Plot
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

# 11. Save final model
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
print(f"Training complete. Model saved to: {model_dir}")

# 12. Inference on test set
print("Running inference on test dataset...")
test_results = trainer.predict(tokenized_datasets["test"])

test_preds = torch.argmax(torch.tensor(test_results.predictions), dim=-1)
test_texts = [tokenizer.decode(pred, skip_special_tokens=True) for pred in test_preds]

# 13. Save test inputs and predictions to Excel
test_df = pd.DataFrame({
    "input_text": dataset["test"]["text"],
    "predicted_text": test_texts
})

excel_path = os.path.join(model_dir, "test_results.xlsx")
test_df.to_excel(excel_path, index=False, encoding="utf-8")
print(f"Test results saved to {excel_path}")

# 14. Evaluation with lm_eval
try:
    from lm_eval.evaluator import simple_evaluate
    results = simple_evaluate(
        model=model,
        tasks=["wikitext"],
        batch_size=args.batch_size,
    )
    print("Evaluation Results:", results)
except Exception as e:
    print(f"Evaluation failed: {e}")

print("Training and inference complete.")
