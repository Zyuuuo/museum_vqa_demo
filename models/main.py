# main.py
from utils.download_met_images import download_met_images
from models.qwen_vl_batch_qa import run_batch_qa

if __name__ == "__main__":
    print("\n🖼️ Step 1: 下载 The Met 公共图像数据...")
    download_met_images(num_images=20)

    print("\n🧠 Step 2: 使用 Qwen-VL 生成图像问答内容...")
    run_batch_qa(
        image_dir="/root/autodl-tmp/Multimodal_heritage/data/images",
        output_json="/root/autodl-tmp/Multimodal_heritage/data/qa_results.json"
    )

    print("\n✅ 所有流程执行完成！")
