import torch
from transformers import AutoTokenizer
from modeling_llama import LlamaForCausalLM, LlamaConfig


def check_tokenizer_model_consistency(model_dir, tokenizer_path=None, test_prompt="The book is about"):
    # 加载 tokenizer
    tokenizer_path = tokenizer_path or model_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 检查特殊 token
    print("\n🔍 Special Tokens 检查：")
    print("PAD token:", tokenizer.pad_token)
    print("UNK token:", tokenizer.unk_token)
    print("EOS token:", tokenizer.eos_token)
    print("BOS token:", tokenizer.bos_token)

    # 检查 vocab size
    print("\n🔢 Vocab & Embedding 尺寸检查：")
    print("Tokenizer vocab size:", len(tokenizer))

    config = LlamaConfig.from_pretrained(model_dir)
    model = LlamaForCausalLM.from_pretrained(model_dir, config=config).eval()

    embedding_size = model.get_input_embeddings().weight.shape[0]
    print("Model embedding size:", embedding_size)

    if len(tokenizer) != embedding_size:
        print("❌ ❗ vocab size ≠ embedding size，模型和 tokenizer 不匹配！")
    else:
        print("✅ ✔️ vocab size 与 embedding size 匹配")

    # 分词测试
    print("\n🧪 分词测试：")
    tokens = tokenizer.tokenize(test_prompt)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    decoded = tokenizer.decode(tokenizer.encode(test_prompt))

    print("原始文本:", test_prompt)
    print("Tokenizer tokens:", tokens)
    print("Token IDs:", ids)
    print("Decoded back:", decoded)

    # 模型生成测试
    print("\n🚀 模型生成检查：")
    inputs = tokenizer(test_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    with torch.no_grad():
        output = model.generate(input_ids=input_ids, max_new_tokens=30)

    gen_ids = output[0].tolist()
    gen_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print("🧠 生成 Token IDs:", gen_ids)
    print("📝 生成文本:", gen_text)

    # UNK 检查
    unk_count = sum(1 for t in tokenizer.convert_ids_to_tokens(gen_ids) if t == tokenizer.unk_token)
    if unk_count > 0:
        print(f"\n⚠️ 检测到生成中包含 {unk_count} 个 <unk> token，可能是词表太小或不匹配")
    else:
        print("\n✅ 未检测到 <unk> token，生成正常")


# 示例使用
if __name__ == "__main__":
    model_dir = r"E:\MHA2HLA\zzq_evaluation\best_epoch_7"
    tokenizer_path = "HuggingFaceTB/cosmo2-tokenizer"  # 或直接用 model_dir
    check_tokenizer_model_consistency(model_dir, tokenizer_path)
