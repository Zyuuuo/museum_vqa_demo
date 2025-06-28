# app/gradio_ui.py
import os
import json
from PIL import Image
import gradio as gr

QA_JSON_PATH = "/root/autodl-tmp/Multimodal_heritage/data/qa_results.json"
IMAGE_DIR = "/root/autodl-tmp/Multimodal_heritage/data/images"

# 加载 JSON
with open(QA_JSON_PATH, encoding="utf-8") as f:
    qa_data = json.load(f)

# 构建索引：文件名 -> 内容
data_dict = {item["image_id"]: item for item in qa_data}

# 可选项（图像文件名）
choices = list(data_dict.keys())

def display_artifact(image_id):
    entry = data_dict[image_id]
    image_path = os.path.join(IMAGE_DIR, image_id)
    description = entry["description"]
    qa_text = "\n\n".join([
        f"Q: {qa['question']}\nA: {qa['answer']}" for qa in entry["qa_pairs"]
    ])
    return Image.open(image_path), description, qa_text

with gr.Blocks(title="Cultural Heritage Artifact Viewer") as demo:
    gr.Markdown("# 🏺 Cultural Heritage Artifact Viewer")

    with gr.Row():
        with gr.Column(scale=1):
            image_selector = gr.Dropdown(choices=choices, label="📂 选择一个文物图像")
            image_display = gr.Image(label="图像展示", interactive=False)

        with gr.Column(scale=2):
            description_box = gr.Textbox(label="📝 描述", lines=3)
            qa_box = gr.Textbox(label="❓ 问答内容", lines=8)

    image_selector.change(fn=display_artifact, inputs=image_selector,
                          outputs=[image_display, description_box, qa_box])

    # 默认显示第一项
    demo.load(fn=display_artifact, inputs=image_selector,
              outputs=[image_display, description_box, qa_box],
              queue=True)

# 启动界面
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
