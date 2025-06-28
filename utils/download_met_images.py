# utils/download_met_images.py
import requests
import os
from tqdm import tqdm

def download_met_images(save_dir="/root/autodl-tmp/Multimodal_heritage/data/images", num_images=20):
    """
    下载大都会博物馆公开的前 num_images 张图像，保存到 save_dir。
    """
    os.makedirs(save_dir, exist_ok=True)

    ids_url = "https://collectionapi.metmuseum.org/public/collection/v1/objects"
    print("[INFO] 获取 The Met 所有开放图像对象 ID...")
    all_ids = requests.get(ids_url).json()["objectIDs"]

    print(f"[INFO] 总共可用对象数: {len(all_ids)}，开始下载前 {num_images} 张有图图像...")
    count = 0

    for object_id in tqdm(all_ids):
        if count >= num_images:
            break
        obj_url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{object_id}"
        obj_data = requests.get(obj_url).json()
        image_url = obj_data.get("primaryImage")

        if image_url:
            try:
                img_data = requests.get(image_url).content
                with open(f"{save_dir}/{object_id}.jpg", "wb") as f:
                    f.write(img_data)
                count += 1
            except Exception as e:
                print(f"[WARN] 图像下载失败: {object_id}, 错误: {e}")

    print(f"\n✅ 共下载 {count} 张图像，保存于：{save_dir}")

if __name__ == "__main__":
    download_met_images()
