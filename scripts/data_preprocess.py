import os
from sklearn.datasets import fetch_lfw_people
import numpy as np
import joblib
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from facenet_pytorch import training
from collections import defaultdict
import shutil

def process_raw_images(data_dir):
    """process raw images from LFW dataset"""
    if os.path.exists(data_dir) and len(os.listdir(data_dir)) > 0:
        print("✅ raw images already exist, skip LFW processing step")
        return True

    print("📥 download and process LFW dataset...")
    lfw_people = fetch_lfw_people(color=True, resize=1.0, download_if_missing=True)
    print(f"✅ Dataset loaded. Total samples: {len(lfw_people.images)}")
    print(f"📐 Image shape: {lfw_people.images[0].shape}")
    print(f"🧑 Unique identities: {len(lfw_people.target_names)}")

    os.makedirs(data_dir, exist_ok=True)
    # 保存预处理后的数据
    for i in range(len(lfw_people.images)):
        person_name = lfw_people.target_names[lfw_people.target[i]]
        person_dir = os.path.join(data_dir, person_name.replace(" ", "_"))
        os.makedirs(person_dir, exist_ok=True)
        img = Image.fromarray((lfw_people.images[i] * 255).astype(np.uint8))
        img.save(os.path.join(person_dir, f'{i}.jpg'))
        if i % 100 == 0:
            print(f"\r💫 Processed: {i}/{len(lfw_people.images)} images", end="")
    
    print(f"\n✅ All raw images have been processed and saved to {data_dir}")
    return True

def process_face_detection(data_dir):
    """process face detection and alignment using MTCNN"""
    cropped_dir = data_dir + '_cropped'
    if os.path.exists(cropped_dir) and len(os.listdir(cropped_dir)) > 0:
        print("✅ face detection and alignment already done, skip MTCNN processing step")
        return True
    remove_low_sample_classes(data_dir)
    print("\n🔍 start face detection and alignment...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(
        image_size=160,  # 输出人脸图片的大小，这里是 160x160 像素
        margin=0, # 人脸周围额外留白的边距，0表示不留白
        min_face_size=20,  # 可以检测的最小人脸尺寸，小于这个尺寸的人脸会被忽略
        thresholds=[0.6, 0.7, 0.7],  # 阈值越高，检测越严格，误检率越低，但可能会漏检
        factor=0.709,   # 图像金字塔的缩放因子，用于多尺度人脸检测
        post_process=True,  # 是否进行后处理，如非极大值抑制
        device=device   # 运行模型的设备（CPU/GPU/MPS）
    )

    dataset = datasets.ImageFolder(data_dir)
    
    dataset.samples = [
        (p, p.replace(data_dir, cropped_dir))
        for p, _ in dataset.samples
    ]
    
    loader = DataLoader(
        dataset,
        num_workers=0 if os.name == 'nt' else 8,
        #num_workers=0,
        batch_size=1,
        collate_fn=training.collate_pil
    )

    for i, (x, y) in enumerate(loader):
        os.makedirs(os.path.dirname(y[0]), exist_ok=True)
        mtcnn(x, save_path=y)
        print(f'\r💫 Processing faces: {i + 1}/{len(loader)}', end='')

    print(f"\n✅ All faces have been detected and aligned, saved to {cropped_dir}")
    del mtcnn
    return True

def remove_low_sample_classes(data_dir, min_images=20):
    print(f"\n🔍 Filtering classes with < {min_images} images...")
    class_image_counts = defaultdict(int)

    # 统计每个类的图片数
    for cls_name in os.listdir(data_dir):
        cls_path = os.path.join(data_dir, cls_name)
        if os.path.isdir(cls_path):
            count = len([f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
            class_image_counts[cls_name] = count

    removed = 0
    for cls_name, count in class_image_counts.items():
        if count < min_images:
            cls_path = os.path.join(data_dir, cls_name)
            shutil.rmtree(cls_path)
            removed += 1
            print(f"🗑️ Removed '{cls_name}' with only {count} images")

    print(f"✅ Finished filtering. Removed {removed} low-sample classes.")

def main():
    data_dir = '/home/egatech2017/airflow/data/test_images'
    
    # 第一阶段：处理原始图片
    if not process_raw_images(data_dir):
        print("❌ process raw images failed")
        return

    # 第二阶段：人脸检测和对齐
    if not process_face_detection(data_dir):
        print("❌ process face detection failed")
        return

    print("✨ all steps completed!")

if __name__ == "__main__":
    main()