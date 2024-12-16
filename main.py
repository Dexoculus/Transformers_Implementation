import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import Compose, Resize, ToTensor
from torchsummary import summary

import base64
from PIL import Image
from IPython.display import Image, display
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from Transformers.Vision_Transformer import ViT
from Transformers.linformer import Linformer

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model = ViT().to(device)
summary(model, (1, 28, 28), device=device)
batch = 32

transform = transforms.Compose([
    transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 데이터셋의 평균 및 표준편차로 정규화
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    epoch_loss = running_loss / len(data_loader)
    epoch_accuracy = 100 * correct / total
    print(f'Training Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

def validate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    epoch_loss = running_loss / len(data_loader)
    epoch_accuracy = 100 * correct / total
    print(f'Validation Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

# 훈련 주기 실행
for epoch in range(10):  # 에폭 수는 필요에 따라 조정 가능
    print(f'Epoch {epoch+1}\n')
    train_epoch(model, train_loader, criterion, optimizer, device)
    validate(model, test_loader, criterion, device)