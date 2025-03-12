from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import torch 

device = "cuda" if torch.cuda.is_available() else "cpu"
# model name
model_name = "AICrossSim/clm-60m"
tokenizer_name = "HuggingFaceTB/cosmo2-tokenizer"

# 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForCausalLM.from_pretrained(model_name,  ignore_mismatched_sizes=True, torch_dtype=torch.float16).to(device)

# text_generator = pipeline("text-generation", model="AICrossSim/clm-200m")


# result = text_generator("The future of AI is", max_length=50, device=device)
# print(result[0]["generated_text"])

# inputs = tokenizer("AI will transform the world", return_tensors="pt").to(device)
# output = model.generate(**inputs, max_length=50)

# generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
# print(generated_text)

input_text = "The future of AI is"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

output = model.generate(input_ids, max_length=50, do_sample=True, temperature=0.7, top_k=50, eos_token_id=tokenizer.eos_token_id)

decoded_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(decoded_text)
