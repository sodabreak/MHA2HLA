import torch
import time
import torch.profiler
from modeling_llama_HLA import LlamaForCausalLM ,LlamaConfig # 确保你有 modeling_llama.py 文件
model_name='AICrossSim/clm-60m'
# 选择设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 LLaMA 配置
config = LlamaConfig.from_pretrained(model_name)  # ✅ 这行不是必须的

# ✅ 正确加载预训练模型
model = LlamaForCausalLM.from_pretrained(model_name).to(device)

model.train()  # 启用训练模式（Dropout / 反向传播）

# 生成测试输入
batch_size = 1
seq_length = 128  # 例如：128 个 token
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length)).to(device)
attention_mask = torch.ones(batch_size, seq_length).to(device)
labels = torch.randint(0, config.vocab_size, (batch_size, seq_length)).to(device)  # 生成随机标签

# **测量各个环节耗时**
with torch.no_grad():  # 先测 Forward
    torch.cuda.synchronize()
    start_time = time.time()

    # **1. 词嵌入（Embedding）**
    embed_start = time.time()
    inputs_embeds = model.model.embed_tokens(input_ids)
    torch.cuda.synchronize()
    embed_time = time.time() - embed_start

    # **2. 旋转位置编码（RoPE）**
    rope_start = time.time()
    position_ids = torch.arange(seq_length, device=device).unsqueeze(0)
    position_embeddings = model.model.rotary_emb(inputs_embeds, position_ids)
    torch.cuda.synchronize()
    rope_time = time.time() - rope_start

    # **3. Transformer 层前向传播**
    transformer_start = time.time()
    hidden_states = inputs_embeds
    layer_times = []

    for layer in model.model.layers:
        layer_start = time.time()
        hidden_states = layer(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings
        )[0]
        torch.cuda.synchronize()
        layer_time = time.time() - layer_start
        layer_times.append(layer_time)

    torch.cuda.synchronize()
    transformer_time = time.time() - transformer_start

    # **4. RMSNorm（归一化）**
    norm_start = time.time()
    hidden_states = model.model.norm(hidden_states)
    torch.cuda.synchronize()
    norm_time = time.time() - norm_start

    # **5. 输出投影（LM Head）**
    lm_head_start = time.time()
    logits = model.lm_head(hidden_states)
    torch.cuda.synchronize()
    lm_head_time = time.time() - lm_head_start

    # **总前向传播时间**
    total_forward_time = time.time() - start_time

# **测量反向传播**
loss_fn = torch.nn.CrossEntropyLoss()
logits = logits.view(-1, config.vocab_size)  # 变形以适配 loss
labels = labels.view(-1)  # 变形为 1D
loss = loss_fn(logits, labels)

# **6. 计算梯度（Backward Pass）**
torch.cuda.synchronize()
backward_start = time.time()
loss.backward()
torch.cuda.synchronize()
backward_time = time.time() - backward_start

# **打印耗时**
print("\n📊 **前向传播（Forward Pass）各子环节耗时分析** 📊")
print(f"🕒 词嵌入（Embedding）：{embed_time:.4f} s")
print(f"🕒 旋转位置编码（RoPE）：{rope_time:.4f} s")
for i, t in enumerate(layer_times):
    print(f"🕒 Transformer 层 {i}（Attention + MLP）：{t:.4f} s")
print(f"🕒 Transformer 层总耗时：{transformer_time:.4f} s")
print(f"🕒 归一化（RMSNorm）：{norm_time:.4f} s")
print(f"🕒 输出投影（LM Head）：{lm_head_time:.4f} s")
print(f"🕒 **总前向传播时间**：{total_forward_time:.4f} s")

print("\n📊 **反向传播（Backward Pass）耗时分析** 📊")
print(f"🕒 反向传播（Backward Pass）：{backward_time:.4f} s")
print(f"🕒 **完整训练步（Forward + Backward）总时间**：{total_forward_time + backward_time:.4f} s")

# **使用 PyTorch Profiler 进行详细分析**
print("\n📊 详细性能分析（torch.profiler）")
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True
) as prof:
    loss = model(input_ids, attention_mask=attention_mask, labels=labels).loss
    loss.backward()

# 打印性能分析结果
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
