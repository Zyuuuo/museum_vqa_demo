import requests
import os
from tqdm import tqdm

# 参数设置
SAVE_DIR = "/root/autodl-tmp/Multimodal_heritage/data/images"
NUM_IMAGES = 20  # 可自行调整为100或更多

# 创建目标文件夹
os.makedirs(SAVE_DIR, exist_ok=True)

# 获取对象 ID 列表
ids_url = "https://collectionapi.metmuseum.org/public/collection/v1/objects"
print("[INFO] 获取 The Met 所有开放图像对象 ID...")
all_ids = requests.get(ids_url).json()["objectIDs"]

print(f"[INFO] 总共可用对象数: {len(all_ids)}，开始下载前 {NUM_IMAGES} 张有图图像...")

count = 0
for object_id in tqdm(all_ids):
    if count >= NUM_IMAGES:
        break
    obj_url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"
    obj_data = requests.get(obj_url).json()
    image_url = obj_data.get("primaryImage")
    if image_url:
        try:
            img_data = requests.get(image_url).content
            with open(f"{SAVE_DIR}/{object_id}.jpg", "wb") as f:
                f.write(img_data)
            count += 1
        except Exception as e:
            print(f"[WARN] 图像下载失败: {object_id}, 错误: {e}")

print(f"\n✅ 共下载 {count} 张图像，保存于：{SAVE_DIR}")
