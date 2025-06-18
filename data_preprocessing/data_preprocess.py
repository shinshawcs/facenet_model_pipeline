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
from google.cloud import storage


def process_raw_images(data_dir):
    """download and process LFW dataset"""
    if os.path.exists(data_dir) and len(os.listdir(data_dir)) > 0:
        print("✅ raw images already exist, skip LFW processing step")
        return True

    print("📥 download and process LFW dataset...")
    lfw_people = fetch_lfw_people(color=True, resize=1.0, download_if_missing=True,data_home='/mnt/data/scikit_learn_data')
    print(f"✅ Dataset loaded. Total samples: {len(lfw_people.images)}")
    print(f"📐 Image shape: {lfw_people.images[0].shape}")
    print(f"🧑 Unique identities: {len(lfw_people.target_names)}")

    os.makedirs(data_dir, exist_ok=True)
    # save preprocessed data
    for i in range(len(lfw_people.images)):
        person_name = lfw_people.target_names[lfw_people.target[i]]
        person_dir = os.path.join(data_dir, person_name.replace(" ", "_"))
        os.makedirs(person_dir, exist_ok=True)
        #image [0,1] float
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
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(
        image_size=160,  # output face image size, 160x160 pixels
        margin=0, # margin around face, 0 means no margin
        min_face_size=20,  # minimum face size to detect, faces smaller than this will be ignored 20*20 pixels ignore
        thresholds=[0.6, 0.7, 0.7],  # thresholds for detection, p-net(bounding-box,landmarks),r-net(),o-net(eye,nose,mouth),
        factor=0.709,   # scaling factor for image pyramid, used for multi-scale face detection
        post_process=True,  # whether to perform post-processing, such as non-maximum suppression
        device='cpu'  # device to run the model (CPU/GPU/MPS)
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
        batch_size=1,# batch size
        collate_fn=training.collate_pil # collate function for batch processing
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

    # count images for each class
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

BUCKET_NAME = "my-facenet-bucket"
storage_client = storage.Client()

def upload_to_gcs(local_folder):
    bucket = storage_client.bucket(BUCKET_NAME)
    for root, _, files in os.walk(local_folder):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_folder)
            blob = bucket.blob(f"preprocessed_data/{relative_path}")
            blob.upload_from_filename(local_path)
            print(f"✅ Uploaded {relative_path} to GCS bucket.")
            
def main():
    airflow_shared_data = os.environ.get('AIRFLOW_SHARED_DATA')
    if not airflow_shared_data:
        print("❌ Environment variable AIRFLOW_SHARED_DATA not set.")
        return
    data_dir = f'{airflow_shared_data}/test_images'
    cropped_data_dir = f'{data_dir}_cropped'
    
    # first stage: process raw images
    if not process_raw_images(data_dir):
        print("❌ process raw images failed")
        return

    # second stage: face detection and alignment
    if not process_face_detection(data_dir):
        print("❌ process face detection failed")
        return
    
    #third stage: upload to GCS
    upload_to_gcs(cropped_data_dir);
    print("✨ all steps completed!")

if __name__ == "__main__":
    main()