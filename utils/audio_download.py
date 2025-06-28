# audio_download.py
# 下载 TTS 中文语音模型到指定目录：/root/autodl-tmp/model/audio

import os
from TTS.utils.manage import ModelManager

# 切换工作目录
model_dir = "/root/autodl-tmp/model/audio"
os.makedirs(model_dir, exist_ok=True)
os.chdir(model_dir)

# 下载模型
manager = ModelManager()
model_path = manager.download_model("tts_models/zh-CN/baker/tacotron2-DDC-GST")

print("✅ 模型已下载至：", model_path)
print("📁 当前所在目录：", os.getcwd())
