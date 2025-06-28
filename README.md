# Museum Multi-modal QA Demo

## 📌 项目简介
- 本项目用于模拟博物馆场景下的展品图像理解与多轮问答交互，并支持语音播报功能。
- 流程：图像上传 ➜ CLIP+FAISS 做语义检索 ➜ Qwen2.5-VL 生成描述/问答 ➜ Edge-TTS 实现语音回答。

## ⚙️ 技术栈
- Python, CLIP, FAISS, Qwen2.5-VL, Gradio (可选), Edge-TTS

## 🏃‍♀️ 快速开始
1. 安装依赖：
   ```bash
   pip install -r requirements.txt
