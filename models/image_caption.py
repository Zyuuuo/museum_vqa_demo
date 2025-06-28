# Day 1: 项目初始化 + 图像数据选型 + CLIP 图像概念匹配 Demo

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import requests
import matplotlib.pyplot as plt

# 1. 载入预训练 CLIP 模型
model_path = '/root/autodl-tmp/Hugging-Face/hub/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
model = CLIPModel.from_pretrained(model_path)
processor = CLIPProcessor.from_pretrained(model_path)

# 2. 示例：博物馆展品图像（可替换为任意公开文化遗产图像）
image_path = '/root/autodl-tmp/Multimodal_heritage/data/images/1.jpg'
image = Image.open(image_path).convert("RGB")

candidate_texts = [
    "an ancient Egyptian statue",
    "a Pharaoh sculpture",
    "a bust of Akhenaten",
    "a Roman emperor bust",
    "a Mesopotamian stone carving",
    "a sandstone religious figure",
    "a medieval European knight statue"
]


# 4. 模型处理输入
inputs = processor(text=candidate_texts, images=image, return_tensors="pt", padding=True)

# 5. 推理得到图像与文本的相似度分布
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits_per_image.softmax(dim=1)[0]  # softmax 概率

# 6. 输出排序结果
print("\n[CLIP 图像语义匹配结果]")
ranked = sorted(zip(candidate_texts, logits), key=lambda x: -x[1])
for text, score in ranked:
    print(f"{text:<30}  ->  score: {score.item():.4f}")

# 7. 展示图像
plt.imshow(image)
plt.axis("off")
plt.title("Museum Artifact Image")
plt.show()

"""
✅ 今日目标：
- 成功运行 CLIP 模型对博物馆图像进行多标签语义分类（图→类属）
- 为后续模块（BLIP caption / VQA）提供语义初筛能力

📌 明日任务预告（Day 2）：使用 BLIP-2 自动生成图像描述 + 图像问答系统框架
"""
