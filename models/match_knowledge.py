# match_knowledge.py
# 将 description 与 knowledge.txt 匹配，并写入知识字段
import torch
import os
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

# 路径配置
qa_path = "/root/autodl-tmp/Multimodal_heritage/data/qa_results.json"
knowledge_path = "/root/autodl-tmp/Multimodal_heritage/data/knowledge.txt"

# 加载知识段落
with open(knowledge_path, encoding="utf-8") as f:
    knowledge_paragraphs = [p.strip() for p in f.read().split("\n") if p.strip()]

# 加载 QA JSON
with open(qa_path, encoding="utf-8") as f:
    qa_data = json.load(f)

# 向量模型
model_id = "/root/autodl-tmp/Hugging-Face/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/86741b4e3f5cb7765a600d3a3d55a0f6a6cb443d"
model = SentenceTransformer(model_id)
paragraph_embeddings = model.encode(knowledge_paragraphs, convert_to_tensor=True)

# 匹配函数
def find_best_knowledge(text, top_k=1):
    query_embedding = model.encode(text, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, paragraph_embeddings)[0]
    top = torch.topk(cos_scores, k=top_k)
    return knowledge_paragraphs[top.indices[0]] if top_k == 1 else [knowledge_paragraphs[i] for i in top.indices]

# 添加字段
for entry in tqdm(qa_data):
    desc = entry.get("description", "")
    entry["knowledge"] = find_best_knowledge(desc)

# 保存新 JSON
with open(qa_path, "w", encoding="utf-8") as f:
    json.dump(qa_data, f, ensure_ascii=False, indent=2)

print("✅ 已将匹配知识段落写入每条记录的 'knowledge' 字段。")
