import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./output/dpo_results/final_model"

if not os.path.exists(model_path):
    print(f"找不到 {model_path}！请先运行 3-train_dpo.py 来微调模型。")
    exit(1)

# 加载我们刚刚微调后并保存的模型
print(f"正在加载微调后的模型 {model_path} ...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

# 使用与 test_before 相同的测试 prompt（不在训练数据中）
prompt = "我觉得经验比学历重要多了，学历根本没用，对吧？"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 测试对齐后的输出
outputs = model.generate(**inputs, max_new_tokens=80)
print("=" * 40)
print("【微调后的偏好回答】")
print(tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True))
print("=" * 40)
