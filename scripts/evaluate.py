import torch
import os
from facenet_pytorch import InceptionResnetV1
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from facenet_pytorch import fixed_image_standardization, training

torch.backends.quantized.engine = 'fbgemm'
def main():
    model_path = "/home/egatech2017/airflow/models/facenet_model_dynamic_quantized.pth"
    data_dir = "/home/egatech2017/airflow/data/test_images_cropped"

    checkpoint = torch.load(model_path, map_location='cpu')
    classes = checkpoint['classes']

    transform = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    dataset = datasets.ImageFolder(data_dir, transform=transform)

    test_inds = np.arange(len(dataset))
    np.random.seed(42)
    np.random.shuffle(test_inds)
    test_inds = test_inds[int(0.85 * len(test_inds)):]  # 后15%

    test_loader = DataLoader(
        dataset,
        batch_size=32,
        sampler=SubsetRandomSampler(test_inds),
        num_workers=0
    )

    model = InceptionResnetV1(
        classify=True,
        pretrained='vggface2',
        num_classes=len(classes)
    )
    model_quantized = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    model_quantized.load_state_dict(checkpoint['model_state_dict'])
    model_quantized.eval()

    print("🔍 Running evaluation...")
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model_quantized(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    print(f"✅ Test Accuracy: {accuracy:.4f}%")

if __name__ == "__main__":
    main()