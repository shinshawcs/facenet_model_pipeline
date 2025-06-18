import torch
import torch.nn as nn
import os
from facenet_pytorch import InceptionResnetV1,fixed_image_standardization
from torch.quantization import prepare_qat, convert,fuse_modules,fuse_linear_bn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from pathlib import Path

torch.backends.quantized.engine = 'fbgemm'
def evaluate(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in dataloader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total

def load_test_loader(data_path, classes):
    transform = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    dataset = datasets.ImageFolder(data_path, transform=transform)
    test_inds = np.arange(len(dataset))
    np.random.seed(42)
    np.random.shuffle(test_inds)
    test_inds = test_inds[int(0.85 * len(test_inds)):]
    loader = DataLoader(
        dataset,
        batch_size=32,
        sampler=SubsetRandomSampler(test_inds),
        num_workers=0
    )
    return loader 
def fuse_all_conv_bn(model):
    for name, module in model.named_children():
        if hasattr(module, 'conv') and hasattr(module, 'bn'):
            print(f"Fusing {name}.conv and {name}.bn")
            fuse_modules(module, ['conv', 'bn'], inplace=True)        
        fuse_all_conv_bn(module)
        
def fuse_linear_bn(linear, bn):
    if not isinstance(linear, nn.Linear) or not isinstance(bn, nn.BatchNorm1d):
        raise TypeError("Expected Linear and BatchNorm1d modules.")
    device = linear.weight.device 
    W = linear.weight.to(device)
    b = linear.bias.to(device) if linear.bias is not None else torch.zeros(W.size(0)).to(device)
    mean = bn.running_mean.to(device)
    var_sqrt = torch.sqrt(bn.running_var.to(device) + bn.eps).to(device)
    gamma = bn.weight.to(device)
    beta = bn.bias.to(device)
    W_new = W * (gamma / var_sqrt).reshape([-1, 1])
    b_new = (b - mean) / var_sqrt * gamma + beta
    fused_linear = nn.Linear(W_new.size(1), W_new.size(0))
    fused_linear.weight.data.copy_(W_new)
    fused_linear.bias.data.copy_(b_new)
    fused_linear.to(device)  
    return fused_linear

def main():
    base_dir = Path(os.environ.get('MODEL_BASE_PATH', '/opt/airflow/shared/models'))
    pvc_data_path = os.environ.get('PVC_DATA_PATH', '/opt/airflow/shared/test_images_cropped')
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No run directories under {base_dir}")
    latest_subdir = max(subdirs, key=lambda d: d.stat().st_mtime)
    
    fp32_path = latest_subdir / 'facenet_model_plain_fp32.pth'
    qat_fp32_path = latest_subdir / 'facenet_model_qat_fp32.pth'
    
    # ------ FP32 baseline ------
    checkpoint_fp32 = torch.load(fp32_path, map_location='cpu')
    classes = checkpoint_fp32['classes']
    test_loader = load_test_loader(pvc_data_path, classes)
    model_fp32 = InceptionResnetV1(classify=True, pretrained='vggface2', num_classes=len(classes))
    model_fp32.load_state_dict(checkpoint_fp32['model_state_dict'], strict=False)
    model_fp32.eval()
    acc_fp32 = evaluate(model_fp32, test_loader)
    print(f"✅ FP32 Accuracy : {acc_fp32:.2f}%")
    # ------ QAT-FP32（未convert，optional）------
    checkpoint_qat_fp32 = torch.load(qat_fp32_path, map_location='cpu')
    classes = checkpoint_fp32['classes']
    model_qat_fp32 = InceptionResnetV1(classify=True, pretrained='vggface2', num_classes=len(classes))
    model_qat_fp32.eval()
    fuse_all_conv_bn(model_qat_fp32)
    model_qat_fp32.last_linear = fuse_linear_bn(model_qat_fp32.last_linear, model_qat_fp32.last_bn)
    weight = checkpoint_qat_fp32['model_state_dict']['conv2d_1a.conv.weight']
    if weight.size(1) == 4:
        print("⚠️ Removing Alpha channel from quantized weights.")
        checkpoint_qat_fp32['model_state_dict']['conv2d_1a.conv.weight'] = weight[:, :3, :, :]
    model_qat_fp32.load_state_dict(checkpoint_qat_fp32['model_state_dict'], strict=False)
    model_qat_fp32.eval()
    acc_qat_fp32 = evaluate(model_qat_fp32, test_loader)
    print(f"⚙️  QAT FP32 Accuracy : {acc_qat_fp32:.2f}%")
   
    print("\n🧪 Evaluation Summary:")
    print(f"FP32        : {acc_fp32:.2f}%")
    print(f"QAT FP32    : {acc_qat_fp32:.2f}%")
    print(f"Accuracy Drop (FP32 → QAT FP32): {acc_fp32 - acc_qat_fp32:.2f}%")

if __name__ == "__main__":
    main()