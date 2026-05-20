"""
第2章 步骤0：从 ModelScope 下载模型
====================================

在运行任何实验之前，先将 Qwen2.5-0.5B-Instruct 下载到本地。
后续脚本会优先从本地加载模型，避免每次都从网络下载。

使用方法：
    pip install modelscope
    python 0-download_model.py
"""

import os
from modelscope import snapshot_download

# 模型保存目录
LOCAL_MODEL_DIR = "./Qwen2.5-0.5B-Instruct"

# ModelScope 上的模型 ID
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def download_model():
    if os.path.exists(LOCAL_MODEL_DIR) and os.path.exists(
        os.path.join(LOCAL_MODEL_DIR, "config.json")
    ):
        print(f"模型已存在于 {LOCAL_MODEL_DIR}，跳过下载。")
        print(f"如需重新下载，请删除 {LOCAL_MODEL_DIR} 目录后重试。")
        return LOCAL_MODEL_DIR

    print(f"正在从 ModelScope 下载模型 {MODEL_ID} ...")
    print("模型约 1GB，请耐心等待。")
    # 用 local_dir 而不是 cache_dir：cache_dir 会按 repo 路径再嵌套
    # 一层（./Qwen2.5-0.5B-Instruct/Qwen/Qwen2___5-0___5B-Instruct/），
    # 导致后续脚本从 LOCAL_MODEL_DIR 加载时找不到 config.json。
    model_dir = snapshot_download(
        MODEL_ID,
        local_dir=LOCAL_MODEL_DIR,
    )
    print(f"模型下载完成，保存至：{model_dir}")
    return model_dir


if __name__ == "__main__":
    download_model()
