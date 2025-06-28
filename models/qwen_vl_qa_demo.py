import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 1. 加载模型和处理器
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto", device_map="auto").eval()
processor = AutoProcessor.from_pretrained(model_id)

# 2. 加载本地图像（替换为你的图像路径）
local_image_path = "/root/autodl-tmp/Multimodal_heritage/data/images/1.jpg"
image = Image.open(local_image_path).convert("RGB")

# 3. 定义图像问答函数
def qwen_vl_ask(image: Image.Image, question: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

# 4. 提问示例
questions = [
    "这尊雕像是什么材质的？",
    "这件文物可能来自哪个地区？",
    "它可能属于哪个历史时期？",
]

print("😄Qwen-VL 本地图像中文问答结果：")
for q in questions:
    answer = qwen_vl_ask(image, q)
    print(f"Q: {q}\nA: {answer}\n")
