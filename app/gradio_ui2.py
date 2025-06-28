# gradio_ui.py（Day7：使用缓存模型播报）
import gradio as gr
import os
import json
from PIL import Image

# ===== 修复 PyTorch 权重加载限制 =====
import torch
try:
    from TTS.utils.radam import RAdam  # 正确引入 RAdam 类
    torch.serialization.add_safe_globals({"TTS.utils.radam.RAdam": RAdam})
    print("[INFO] 已允许 RAdam 加载")
except Exception as e:
    print("[WARN] 无法设置 safe_globals：", e)

# ===== 使用缓存模型加载 TTS =====
try:
    from TTS.api import TTS
    tts = TTS(model_name="tts_models/zh-CN/baker/tacotron2-DDC-GST", progress_bar=False, gpu=False)
    print("[INFO] 成功加载缓存模型：tts_models/zh-CN/baker/tacotron2-DDC-GST")
except Exception as e:
    print("[ERROR] TTS 模型加载失败：", e)
    tts = None

# ===== 数据路径 =====
qa_path = "/root/autodl-tmp/Multimodal_heritage/data/qa_results.json"
image_dir = "/root/autodl-tmp/Multimodal_heritage/data/images"
audio_output_path = "/root/autodl-tmp/Multimodal_heritage/data/description_audio.wav"

# ===== 加载问答结果 =====
with open(qa_path, encoding="utf-8") as f:
    qa_data = json.load(f)

image_dict = {entry["image_id"]: entry for entry in qa_data}

# ===== 合成语音函数 =====
def speak_text(text):
    if not tts:
        print("[WARN] TTS 模型未加载，跳过语音合成。")
        return None
    try:
        tts.tts_to_file(text=text, file_path=audio_output_path)
        print(f"[INFO] 语音文件生成成功：{audio_output_path}")
        return audio_output_path
    except Exception as e:
        print("[ERROR] 合成失败：", e)
        return None

# ===== 主展示函数 =====
def show_info(image_id):
    entry = image_dict[image_id]
    img_path = os.path.join(image_dir, image_id)
    image = Image.open(img_path).convert("RGB")

    description = entry.get("description", "")
    qa_text = "\n\n".join([f"Q: {q['question']}\nA: {q['answer']}" for q in entry.get("qa_pairs", [])])
    knowledge = entry.get("knowledge", "暂无匹配段落")

    audio = speak_text(description)

    return image, description, qa_text, knowledge, audio

# ===== Gradio UI 界面构建 =====
dropdown = gr.Dropdown(choices=list(image_dict.keys()), label="选择图片")
image_display = gr.Image(type="pil", label="图像展示")
description_box = gr.Textbox(label="图像描述")
qa_box = gr.Textbox(label="问答内容")
knowledge_box = gr.Textbox(label="📖 匹配知识段落")
audio_box = gr.Audio(label="🔊 图像说明语音播放", type="filepath")

interface = gr.Interface(
    fn=show_info,
    inputs=dropdown,
    outputs=[image_display, description_box, qa_box, knowledge_box, audio_box],
    title="多模态文化遗产展示系统",
    description="选择图像后，将展示其描述、问答结果、匹配知识段落和语音播报"
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860, share=False)
