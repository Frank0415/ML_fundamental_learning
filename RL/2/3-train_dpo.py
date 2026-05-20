import json
import os
from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# 优先使用本地模型（由 0-download_model.py 下载）
LOCAL_MODEL_DIR = "./Qwen2.5-0.5B-Instruct"

# ==========================================
# 1. 准备偏好数据
# ==========================================
data_file = "output/preference_data.json"

if not os.path.exists(data_file):
    print(f"找不到 {data_file}！请先运行 1-generate_data.py 来生成偏好数据。")
    exit(1)

with open(data_file, "r", encoding="utf-8") as f:
    data_list = json.load(f)

# 转换为 HuggingFace Dataset 的结构
data_dict = {
    "prompt": [item["prompt"] for item in data_list],
    "chosen": [item["chosen"] for item in data_list],
    "rejected": [item["rejected"] for item in data_list]
}
train_dataset = Dataset.from_dict(data_dict)

# ==========================================
# 2. 加载模型与分词器
# ==========================================
model_name = LOCAL_MODEL_DIR if os.path.exists(LOCAL_MODEL_DIR) else "Qwen/Qwen2.5-0.5B-Instruct"
print(f"正在加载基础模型 {model_name} ...")
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# DPO 需要 pad_token，如果不设置会报错
tokenizer.pad_token = tokenizer.eos_token

# ==========================================
# 3. 配置训练参数与 DPOTrainer
# ==========================================
training_args = DPOConfig(
    output_dir="./output/dpo_results",
    per_device_train_batch_size=2,
    learning_rate=1e-5,
    num_train_epochs=3,   # 这里可以调大以加深学习效果
    logging_steps=5,      # 打印日志的频率
    save_steps=20,        # 模型保存频率
    beta=0.1,             # KL 惩罚系数（课程推荐的 TRL 0.24 放在 DPOConfig 中）
)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,  # TRL 0.24 使用 processing_class 传入 tokenizer/processor
)

# ==========================================
# 4. 开始偏好微调并保存
# ==========================================
print("\n开始 DPO 训练... (可以观察 loss 曲线和 rewards margin 的变化)")
trainer.train()

# 训练完成后保存结果
save_path = "./output/dpo_results/final_model"
trainer.save_model(save_path)
print(f"训练完成！微调后的模型已保存至 {save_path}。")
print("你可以运行 4-test_after.py 来看看模型是否学会了'不盲从用户'。")
