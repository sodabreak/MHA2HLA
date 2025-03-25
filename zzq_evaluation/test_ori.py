import torch
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import LlamaForCausalLM, LlamaConfig
model_name="AICrossSim/clm-60m"
def compute_perplexity(model, tokenizer, dataset, desc="Perplexity Evaluation"):
    model.eval()
    model.cuda()

    losses = []
    total_tokens = 0
    used_examples = 0
    for example in tqdm(dataset, desc=desc):
        text = example["text"].strip()
        if len(text.split()) < 5:
            continue
        used_examples += 1
        split_point = len(text) // 2
        context = text[:split_point].strip()
        continuation = text[split_point:].strip()
        full = context + continuation

        inputs = tokenizer(full, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs.input_ids.to(model.device)
        labels = input_ids.clone()

        ctx_ids = tokenizer(context, add_special_tokens=False).input_ids
        ctx_len = len(ctx_ids)
        labels[:, :ctx_len] = -100  # 只评估 continuation 的 loss

        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss.item()
            num_tokens = (labels != -100).sum().item()
            losses.append(loss * num_tokens)
            total_tokens += num_tokens

    print(f"🔢 实际使用的样本数: {used_examples}")
    print(f"📏 Total tokens evaluated: {total_tokens}")
    ppl = np.exp(np.sum(losses) / total_tokens)
    print(f"\n📏 Perplexity: {ppl:.2f}")
    return ppl

# 加载模型和 tokenizer
config = LlamaConfig.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name, config=config)
tokenizer_name = "HuggingFaceTB/cosmo2-tokenizer"

# 加载 tokenizer（仍使用你的）
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载 config，并用随机初始化模型（不加载权重）
model = model.to("cuda").eval()


# 加载评估集（静态）
dataset = load_dataset("wikitext", "wikitext-2-v1", split="test")

# ✅ 计算 perplexity
compute_perplexity(model, tokenizer, dataset)
