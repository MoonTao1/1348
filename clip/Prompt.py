import json
import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# 设置本地模型路径（软链接目录）
model_path = os.path.expanduser("~/.cache/huggingface/hub/models--Salesforce--blip-image-captioning-base")

# 加载 processor 和模型
processor = BlipProcessor.from_pretrained(model_path, local_files_only=True)
model = BlipForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# 数据路径
root = "/data9102/workspace/mwt/dataset/night/trafficframe/"
index_file = root + "test.json"

# 读取图像路径（每行是一个 JSON 字符串，如 "01/000006.jpg"）
train_imgs = [json.loads(line) for line in open(index_file)]

# 初始化结果字典
captions_dict = {}

for i, img_name in enumerate(train_imgs):
    img_path = os.path.join(root, img_name)
    try:
        image = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"[跳过] 无法打开图像 {img_path}，错误：{e}")
        captions_dict[img_name] = "[INVALID]"
        continue

    # 图像生成描述
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs)

    caption = processor.decode(out[0], skip_special_tokens=True)
    captions_dict[img_name] = caption
    print(f"[{i+1}/{len(train_imgs)}] {img_name} -> {caption}")

# 保存为 JSON 文件
output_json_path = os.path.join(root, "test_captions.json")
with open(output_json_path, "w") as f:
    json.dump(captions_dict, f, indent=2)

print(f"\n✅ 所有描述已保存到 JSON 文件: {output_json_path}")
