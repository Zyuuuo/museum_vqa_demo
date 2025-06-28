# gradio_ui.py（Day7：集成 edge-tts 播报）
import gradio as gr
import os
import json
from PIL import Image
import asyncio
import edge_tts

# ===== 数据路径 =====
qa_path = "/root/autodl-tmp/Multimodal_heritage/data/qa_results.json"
image_dir = "/root/autodl-tmp/Multimodal_heritage/data/images"
audio_output_path = "/root/autodl-tmp/Multimodal_heritage/data/description_audio.mp3"

# ===== 加载问答结果 =====
with open(qa_path, encoding="utf-8") as f:
    qa_data = json.load(f)

image_dict = {entry["image_id"]: entry for entry in qa_data}

# ===== 语音合成函数（edge-tts 异步封装） =====
async def speak_async(text, voice="zh-CN-XiaoxiaoNeural", filename=audio_output_path):
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(filename)
        print(f"[INFO] 语音文件生成成功：{filename}")
    except Exception as e:
        print("[ERROR] 合成失败：", e)

# 同步封装，用于主线程调用
def speak_text(text):
    try:
        asyncio.run(speak_async(text))
        return audio_output_path
    except Exception as e:
        print("[WARN] TTS 播报失败：", e)
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
