import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import argparse
import os
import subprocess
from pathlib import Path
from google.cloud import storage
from datetime import datetime

GCS_BUCKET = os.environ.get('GCS_BUCKET', 'my-facenet-bucket')

def get_next_version(bucket_name, model_prefix):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=model_prefix)
    version_set = set()
    for blob in blobs:
        rel_path = blob.name[len(model_prefix):]  
        if "/" in rel_path:
            version_str = rel_path.split("/")[0]
            if version_str.isdigit():
                version_set.add(int(version_str))
    if version_set:
        next_version = max(version_set) + 1
    else:
        next_version = 1
    return str(next_version)

def upload_new_version_to_gcs(local_plan_path, bucket_name, model_prefix):
    next_version = get_next_version(bucket_name, model_prefix)
    gcs_path = f"{model_prefix}{next_version}/model.plan"
    upload_to_gcs(local_plan_path, gcs_path, bucket_name)
    print(f"✅ Uploaded to {gcs_path}")
    
def upload_to_gcs(local_path, gcs_path, bucket_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    print(f"✅ Uploaded {local_path} to gs://{bucket_name}/{gcs_path}")
    
def build_trt_engine():  
    base_dir = Path(os.environ.get('MODEL_BASE_PATH', '/opt/airflow/shared/models'))
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No run directories under {base_dir}")
    latest_subdir = max(subdirs, key=lambda d: d.stat().st_mtime)
    onnx_path = f'{latest_subdir}/facenet_model_qat_fp32.onnx'
    plan_path = f"{latest_subdir}/model.plan"
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"❌ ONNX file not found at: {onnx_path}")
    
    trtexec_cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        "--minShapes=input:1x3x160x160",
        "--optShapes=input:8x3x160x160",
        "--maxShapes=input:32x3x160x160",
        "--explicitBatch",
        "--fp16",
        f"--saveEngine={plan_path}"
    ]

    print(f"🚀 Running TensorRT engine build:\n{' '.join(trtexec_cmd)}")
    timestamp = datetime.now().strftime("%Y%m%d")
    gcs_model_dir = f"models/{timestamp}"
    model_prefix = "triton/models/facenet_tensorrt/"
    try:
        subprocess.run(trtexec_cmd, check=True)
        print(f"✅ TensorRT engine saved to {plan_path}")
        upload_to_gcs(plan_path, f"{gcs_model_dir}/model.plan", GCS_BUCKET)
        upload_new_version_to_gcs(plan_path, GCS_BUCKET, model_prefix)
        print(f"✅ TensorRT model) saved to {gcs_model_dir}")
    except FileNotFoundError:
        print("❌ Cannot find trtexec command")
    except subprocess.CalledProcessError as e:
        print(f"❌ TensorRT export failed: {e}")


if __name__ == "__main__":
    build_trt_engine()