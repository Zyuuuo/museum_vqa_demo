# Day 2: 图像描述 + 图像问答 - 使用 BLIP-2
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Step 1: 加载模型和处理器
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "/root/autodl-tmp/Hugging-Face/hub/models--Salesforce--blip2-opt-2.7b/snapshots/59a1ef6c1e5117b3f65523d1c6066825bcf315e3"

print("[INFO] Loading BLIP-2 model...")
processor = Blip2Processor.from_pretrained(model_id)
model = Blip2ForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
).eval()

# Step 2: 加载图像
image_path = "/root/autodl-tmp/Multimodal_heritage/data/images/1.jpg"
image = Image.open(image_path).convert("RGB")

# Step 3: 生成图像描述
def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=50)
    return processor.decode(out[0], skip_special_tokens=True)

# Step 4: 图像问答函数
def answer_question(image, question):
    inputs = processor(images=image, text=question, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=30)
    return processor.decode(out[0], skip_special_tokens=True)

# Step 5: 运行
caption = generate_caption(image)
print(f"\n🖼 图像描述:\n{caption}\n")

questions = [
    "What is the object made of?",
    "Where might this object be from?",
    "What is the age of this artifact?"
]

print("🎯 图像问答:")
for q in questions:
    a = answer_question(image, q)
    print(f"Q: {q}\nA: {a}\n")
