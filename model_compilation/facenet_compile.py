from re import T
import torch
import os
from facenet_pytorch import InceptionResnetV1
import subprocess
from torch.ao.quantization import quantize_fx, QConfigMapping
from google.cloud import storage
from datetime import datetime
from pathlib import Path

torch.backends.quantized.engine = 'fbgemm'
GCS_BUCKET = os.environ.get('GCS_BUCKET', 'my-facenet-bucket')
MODEL_BASE_PATH = '/opt/airflow/shared/models'
os.makedirs(MODEL_BASE_PATH, exist_ok=True)

def remove_fake_quant(model):
    modules_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, torch.quantization.FakeQuantize):
            print(f"Removing FakeQuantize layer: {name}")
            modules_to_replace.append(name)
    for name in modules_to_replace:
        parent_name = ".".join(name.split(".")[:-1])
        parent_module = model.get_submodule(parent_name) if parent_name else model
        setattr(parent_module, name.split(".")[-1], torch.nn.Identity())
    return model

def upload_to_gcs(local_path, gcs_path, bucket_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    print(f"✅ Uploaded {local_path} to gs://{bucket_name}/{gcs_path}")
    
def export_model_to_onnx_and_trt(model, dummy_input, prefix_path, use_int8=False):
    timestamp = datetime.now().strftime("%Y%m%d")
    gcs_model_dir = f"models/{timestamp}"
    suffix = "qat_fp32" if use_int8 else "fp32"
    onnx_path = f"{prefix_path}/facenet_model_{suffix}.onnx"
    suffix_trt = "qat_int8" if use_int8 else "fp16"
    trt_engine_path = f"{prefix_path}/facenet_model_{suffix_trt}.trt"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=13
    )
    print(f"✅ ONNX model ({suffix}) saved to {onnx_path}")
    upload_to_gcs(onnx_path, f"{gcs_model_dir}/facenet_model_{suffix}.onnx", GCS_BUCKET)
    print(f"✅ ONNX model ({suffix}) saved to {gcs_model_dir}")
    trtexec_cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        "--minShapes=input:1x3x160x160",
        "--optShapes=input:8x3x160x160",
        "--maxShapes=input:32x3x160x160",
        "--explicitBatch",
        "--int8" if use_int8 else "--fp16",
        f"--saveEngine={trt_engine_path}"
    ]
    print(f"✅ TensorRT model ({suffix_trt}) saved to {gcs_model_dir}")
    try:
        subprocess.run(trtexec_cmd, check=True)
        print(f"✅ TensorRT engine ({suffix_trt}) saved to {trt_engine_path}")
        upload_to_gcs(trt_engine_path, f"{gcs_model_dir}/facenet_model_{suffix_trt}.trt", GCS_BUCKET)
        print(f"✅ TensorRT model ({suffix_trt}) saved to {gcs_model_dir}")
    except FileNotFoundError:
        print("❌ Cannot find trtexec command")
    except subprocess.CalledProcessError as e:
        print(f"❌ TensorRT export failed ({suffix_trt}): {e}")
        
def main():    
    base_dir = Path(os.environ.get('MODEL_BASE_PATH', '/opt/airflow/shared/models'))
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No run directories under {base_dir}")
    latest_subdir = max(subdirs, key=lambda d: d.stat().st_mtime)
    dummy_input = torch.randn(1, 3, 160, 160)
    # set up plain fp32 model
    plain_fp32_path = f'{latest_subdir}/facenet_model_plain_fp32.pth'
    if not os.path.exists(plain_fp32_path):
        print(f"❌ Model not found at {plain_fp32_path}")
        return
    checkpoint_fp32 = torch.load(plain_fp32_path, map_location="cpu")
    weight = checkpoint_fp32['model_state_dict']['conv2d_1a.conv.weight']
    if weight.size(1) == 4:  
        print("⚠️ Removing Alpha channel from pretrained weights.")
        checkpoint_fp32['model_state_dict']['conv2d_1a.conv.weight'] = weight[:, :3, :, :]
    model_fp32 = InceptionResnetV1(classify=True, pretrained='vggface2', num_classes=len(checkpoint_fp32['classes']))
    model_fp32.load_state_dict(checkpoint_fp32['model_state_dict'], strict=False)
    model_fp32.eval()
    export_model_to_onnx_and_trt(model_fp32, dummy_input, latest_subdir, use_int8=False)
    #  set up qat fp32 model
    qat_fp32_path = f'{latest_subdir}/facenet_model_qat_fp32.pth'
    if not os.path.exists(qat_fp32_path):
        print(f"❌ Model not found at {qat_fp32_path}")
        return
    checkpoint_qat_fp32 = torch.load(qat_fp32_path, map_location='cpu')
    weight = checkpoint_qat_fp32['model_state_dict']['conv2d_1a.conv.weight']
    if weight.size(1) == 4:  
        print("⚠️ Removing Alpha channel from pretrained weights.")
        checkpoint_qat_fp32['model_state_dict']['conv2d_1a.conv.weight'] = weight[:, :3, :, :]
    model_qat = InceptionResnetV1(classify=True, pretrained='vggface2', num_classes=len(checkpoint_qat_fp32['classes']))
    model_qat.load_state_dict(checkpoint_qat_fp32['model_state_dict'], strict=False)
    model_qat.eval()
    export_model_to_onnx_and_trt(model_qat, dummy_input, latest_subdir, use_int8=True) 
  
if __name__ == "__main__":
    main()