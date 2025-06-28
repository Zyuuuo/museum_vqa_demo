# qwen_vl_batch_qa.py（稳定版：提升生成长度 max_new_tokens=256）
import os
import json
from tqdm import tqdm
from PIL import Image

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ===== Step 1: 模型与处理器加载（使用 float16 精度） =====
model_path = "/root/autodl-tmp/Hugging-Face/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
).eval()

processor = AutoProcessor.from_pretrained(model_path)

# ===== Step 2: 问题模板（可扩展） =====
questions = [
    "请描述这张图像中最主要的内容。",
    "这件文物是什么材质的？",
    "这件文物可能来自哪个地区？",
    "它可能属于哪个历史时期？"
]

# ===== Step 3: 单张图像问答函数 =====
def qa_for_image(image: Image.Image, question_list):
    qa_result = []
    for question in question_list:
        torch.cuda.empty_cache()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=256)  # 提升生成长度

        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, generated_ids)]
        output = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        qa_result.append((question, output))

        del inputs, generated_ids, trimmed
        torch.cuda.empty_cache()

    return qa_result

# ===== Step 4: 图像目录批处理函数 =====
def run_batch_qa(image_dir, output_json):
    all_results = []

    for fname in tqdm(os.listdir(image_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(image_dir, fname)
        image = Image.open(img_path).convert("RGB").resize((512, 512))
        qa_pairs = qa_for_image(image, questions)

        result = {
            "image_id": fname,
            "title": qa_pairs[0][1][:20],
            "description": qa_pairs[0][1],
            "qa_pairs": [
                {"question": q, "answer": a} for q, a in qa_pairs[1:]
            ]
        }
        all_results.append(result)

        del image, qa_pairs
        torch.cuda.empty_cache()

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 共处理图像 {len(all_results)} 张，结果已保存至：{output_json}")

# ===== Step 5: 主函数入口 =====
if __name__ == "__main__":
    run_batch_qa(
        image_dir="/root/autodl-tmp/Multimodal_heritage/data/images",
        output_json="/root/autodl-tmp/Multimodal_heritage/data/qa_results.json"
    )
