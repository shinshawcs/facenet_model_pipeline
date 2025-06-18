# scripts/train_facenet.py

import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1,training,fixed_image_standardization
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score
import os
import numpy as np
from torch.ao.quantization import QConfig, default_observer, default_weight_observer
import torch.optim as optim
from torch.quantization import prepare_qat, convert,fuse_modules,fuse_linear_bn
from torch.utils.data import DataLoader,SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import wandb 
from google.cloud import storage
from datetime import datetime


GCS_BUCKET = os.environ.get('GCS_BUCKET', 'my-facenet-bucket')
PVC_DATA_PATH = os.environ.get('PVC_DATA_PATH', '/opt/airflow/shared/test_images_cropped')
MODEL_BASE_PATH = '/opt/airflow/shared/models'
os.makedirs(MODEL_BASE_PATH, exist_ok=True)
    
def upload_to_gcs(local_path, bucket_name, gcs_path):
    if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
        print(f"✅ GCS credentials set to {os.environ['GOOGLE_APPLICATION_CREDENTIALS']}")
    else:
        print("⚠️ GOOGLE_APPLICATION_CREDENTIALS is not set!")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    print(f"✅ Uploaded {local_path} to gs://{bucket_name}/{gcs_path}")
    
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

def patch_block8_relu(model):
    for name, module in model.named_modules():
        if 'block8' in name and not hasattr(module, 'relu'):
            print(f"🛠️ Adding ReLU to {name}")
            module.relu = nn.ReLU() 
            
def find_latest_model(base_dir, pattern='facenet_model_plain_fp32.pth'):
    """Find the latest model in time-stamped subdirectories"""
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    subdirs.sort(reverse=True)
    for subdir in subdirs:
        candidate = os.path.join(subdir, pattern)
        if os.path.exists(candidate):
            return candidate
    return None

def train_plain_fp32_model(dataset,device,latest_model_path,model_dir,train_loader,val_loader,start_epoch=0,epochs=2):
    model_path = os.path.join(model_dir, 'facenet_model_plain_fp32.pth')
    model = InceptionResnetV1(classify=True, pretrained='vggface2', num_classes=len(dataset.classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [3, 4], gamma=0.1)
    metrics = {'fps': training.BatchTimer(), 'acc': training.accuracy}
    start_epoch = 0
    val_metrics = None
    if latest_model_path and os.path.exists(latest_model_path):
        print(f"📥 Loading plain FP32 checkpoint from {latest_model_path}")
        checkpoint = torch.load(latest_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        val_metrics = checkpoint.get('val_metrics', None)
        print(f"✅ Loaded plain model from epoch {start_epoch}")
    writer = SummaryWriter()
    writer.iteration, writer.interval = 0, 10
    for epoch in range(start_epoch, epochs):
         print(f'\nEpoch {epoch + 1}/{epochs} (Plain FP32)')
         model.train()
         training.pass_epoch(
             model, loss_fn, train_loader,
             optimizer, scheduler, batch_metrics=metrics, show_running=True, device=device,
             writer=writer
         )
         model.eval()
         val_metrics = training.pass_epoch(model, loss_fn, val_loader, batch_metrics=metrics, show_running=True, device=device, writer=writer)
    torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_metrics': val_metrics,
            'classes': dataset.classes
        }, model_path)
    writer.close()
    print(f"✅ Plain FP32 model (trained {epochs} epochs) saved to {model_path}")
    return model_path

def train_qat_model(dataset,device,latest_qat_fp32_ckpt_path,model_dir,train_loader,val_loader,plain_fp32_ckpt_path,start_epoch=0,epochs=2):
    model_path = os.path.join(model_dir, 'facenet_model_qat_fp32.pth')
    model = InceptionResnetV1(classify=True, pretrained='vggface2', num_classes=len(dataset.classes)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00005)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [3, 4], gamma=0.1)
    metrics = {'fps': training.BatchTimer(), 'acc': training.accuracy}
    val_metrics = None

    if not (plain_fp32_ckpt_path and os.path.exists(plain_fp32_ckpt_path)):
        raise RuntimeError("QAT need plain FP32 model，please train plain FP32 model！")
    fp32_ckpt = torch.load(plain_fp32_ckpt_path, map_location=device)
    model.load_state_dict(fp32_ckpt['model_state_dict'])
    
    model.eval()
    fuse_all_conv_bn(model)
    model.last_linear = fuse_linear_bn(model.last_linear, model.last_bn)
    torch.backends.quantized.engine = 'fbgemm'
    qconfig = QConfig(
        activation=default_observer.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
        weight=default_weight_observer.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine)
    )
    model.qconfig = qconfig
    model.train()
    model = prepare_qat(model)
    if latest_qat_fp32_ckpt_path and os.path.exists(latest_qat_fp32_ckpt_path):
        print(f"📥 Loading QAT FP32 checkpoint from {latest_qat_fp32_ckpt_path}")
        checkpoint = torch.load(latest_qat_fp32_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        val_metrics = checkpoint.get('val_metrics', None)
        print(f"✅ Loaded QAT model from epoch {start_epoch}")
    else:
        print("🔄 Preparing QAT model")
        start_epoch = 0
    writer = SummaryWriter()
    writer.iteration, writer.interval = 0, 10
    for epoch in range(start_epoch, epochs):
        print(f'\nEpoch {epoch + 1}/{epochs} (QAT FP32)')
        model.train()
        training.pass_epoch(model, loss_fn, train_loader, optimizer, scheduler, batch_metrics=metrics, show_running=True, device=device, writer=writer)
        model.eval()
        val_metrics = training.pass_epoch(model, loss_fn, val_loader, batch_metrics=metrics, show_running=True, device=device, writer=writer)
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_metrics': val_metrics,
        'classes': dataset.classes
    }, model_path)
    writer.close()
    print(f"✅ QAT FP32 model (trained {epochs} epochs) saved to {model_path}")
    return model, model_path, val_metrics

def save_qat_int8_model(model, dataset, model_dir, epochs, val_metrics):
    int8_path = os.path.join(model_dir, 'facenet_model_qat_int8.pth')
    quantized_model = convert(model.eval())
    torch.save({
        'epoch': epochs,
        'model_state_dict': quantized_model.state_dict(),
        'optimizer_state_dict': None,
        'val_metrics': val_metrics,
        'classes': dataset.classes
    }, int8_path)
    print(f"✅ QAT INT8 model saved to {int8_path}")
    return int8_path 

def main():
    # data preprocessing and loading
    transform = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization  # ImageNet normalization parameters
    ])

    dataset = datasets.ImageFolder(PVC_DATA_PATH, transform=transform)
    print(f"✅ load dataset completed, {len(dataset)} images")
    print(f"👥 {len(dataset.classes)} people")
    
    # split dataset into training set (70%), validation set (15%) and test set (15%)
    img_inds = np.arange(len(dataset))
    np.random.seed(42)  # set random seed for reproducibility
    np.random.shuffle(img_inds)
    
    train_size = int(0.7 * len(img_inds))
    val_size = int(0.15 * len(img_inds))
    
    train_inds = img_inds[:train_size]
    val_inds = img_inds[train_size:train_size + val_size]
    
    print(f"✅ train_inds: length {len(train_inds)} ({len(train_inds)/len(img_inds):.1%})")
    print(f"✅ val_inds: length {len(val_inds)} ({len(val_inds)/len(img_inds):.1%})")

    # set smaller batch size to reduce memory usage
    batch_size = 32
    train_loader = DataLoader(
        dataset,
        num_workers=4,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_inds),
        pin_memory=True 
    )
    val_loader = DataLoader(
        dataset,
        num_workers=4,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(val_inds),
        pin_memory=True 
    )

    # # initialize model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
        
    timestamp = datetime.now().strftime("%Y%m%d")
    model_dir = os.path.join(MODEL_BASE_PATH, timestamp)
    os.makedirs(model_dir, exist_ok=True)
    
    #train plain fp32 model
    latest_fp32_plain = find_latest_model(MODEL_BASE_PATH, pattern='facenet_model_plain_fp32.pth')
    plain_fp32_path = train_plain_fp32_model(dataset, device, latest_fp32_plain, model_dir,train_loader, val_loader, start_epoch=0, epochs=2)
    
    #train quantized model
    latest_qat_fp32 = find_latest_model(MODEL_BASE_PATH, pattern='facenet_model_qat_fp32.pth')
    qat_model, qat_fp32_path, val_metrics = train_qat_model(dataset, device, latest_qat_fp32, model_dir,train_loader, val_loader, plain_fp32_path,start_epoch=0, epochs=2)
    qat_int8_path = save_qat_int8_model(qat_model, dataset, model_dir, epochs=5, val_metrics=val_metrics)
     
    gcs_model_dir = f"models/{timestamp}" 
    upload_to_gcs(plain_fp32_path, GCS_BUCKET, f"{gcs_model_dir}/facenet_model_plain_fp32.pth")
    upload_to_gcs(qat_fp32_path, GCS_BUCKET, f"{gcs_model_dir}/facenet_model_qat_fp32.pth")
    upload_to_gcs(qat_int8_path, GCS_BUCKET, f"{gcs_model_dir}/facenet_model_qat_int8.pth")
      
if __name__ == "__main__":
    main()