# scripts/train_facenet.py

import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os
import numpy as np
from PIL import Image
from glob import glob
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from facenet_pytorch import training
from torch.utils.tensorboard import SummaryWriter
from facenet_pytorch import fixed_image_standardization
import multiprocessing as mp
def main():
    # load aligned face images processed by MTCNN
    data_dir = '/home/egatech2017/airflow/data/test_images_cropped'
    
    if not os.path.exists(data_dir):
        print(f"❌ processed face data set does not exist: {data_dir}")
        print("please run data_preprocess.py first")
        return

    # data preprocessing and loading
    transform = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization  # ImageNet normalization parameters
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
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
        num_workers=0,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_inds)
    )
    val_loader = DataLoader(
        dataset,
        num_workers=0,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(val_inds)
    )

    # # initialize model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'
    print(f"Using device: {device}")
    
    # check if the saved model exists
    save_dir = '/home/egatech2017/airflow/models'
    model_path = os.path.join(save_dir, 'facenet_model.pth')
    
    start_epoch = 0
    if os.path.exists(model_path):
        print(f"📥 Loading saved model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        resnet = InceptionResnetV1(
            classify=True,
            pretrained='vggface2',
            num_classes=len(dataset.classes)
        ).to(device)
        resnet.load_state_dict(checkpoint['model_state_dict'])
        #start_epoch = checkpoint['epoch']
        print(f"✅ Model loaded successfully (trained for {start_epoch} epochs)")
    else:
        # if no saved model, create new model
        print("🆕 Creating new model...")
        resnet = InceptionResnetV1(
            classify=True,
            pretrained='vggface2',
            num_classes=len(dataset.classes)
        ).to(device)

    # # set optimizer and learning rate scheduler
    optimizer = optim.Adam(resnet.parameters(), lr=0.0001)  # reduce learning rate
    if os.path.exists(model_path):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✅ Optimizer state loaded")

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [5, 10], gamma=0.1)
    epochs = 1  # total training epochs

    loss_fn = torch.nn.CrossEntropyLoss()
    metrics = {
        'fps': training.BatchTimer(),
        'acc': training.accuracy
    }

    writer = SummaryWriter()
    writer.iteration, writer.interval = 0, 10

    print('\n\nInitial')
    print('-' * 10)
    resnet.eval()
    training.pass_epoch(
        resnet, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

    # continue training from the last training end
    for epoch in range(start_epoch, epochs):
        print('\nEpoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        resnet.train()
        training.pass_epoch(
            resnet, loss_fn, train_loader, optimizer, scheduler,
            batch_metrics=metrics, show_running=True, device=device,
            writer=writer
        )

        resnet.eval()
        val_metrics = training.pass_epoch(
            resnet, loss_fn, val_loader,
            batch_metrics=metrics, show_running=True, device=device,
            writer=writer
        )

    writer.close()

    os.makedirs(save_dir, exist_ok=True)
    
    # save model (include architecture and weights)
    torch.save({
        'epoch': epochs,
        'model_state_dict': resnet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_metrics': val_metrics,
        'classes': dataset.classes
    }, model_path)
    
    print(f"\n✅ Model saved to {model_path}")
    
if __name__ == "__main__":
    main()