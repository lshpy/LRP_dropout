import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import ResNetWithHook
from strategy import apply_strategy
from utils.lrp import compute_lrp
from utils.gradcam import compute_gradcam
import csv
import os

# 실험 조건 리스트
units = ['pixel', 'patch', 'channel']
strategies = ['baseline', 'random', 'suppressive', 'gradcam_amp', 'hybrid_drop', 'hybrid_amp', 'mixed']
epochs = 5  # 빠른 실험용 (최종 논문은 20회 이상 권장)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터셋 준비
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# 결과 저장 파일
result_path = "results/experiment_results.csv"
os.makedirs("results", exist_ok=True)
with open(result_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["strategy", "unit", "test_accuracy"])

# 전체 실험 루프
for unit in units:
    for strategy in strategies:
        print(f"\n🚀 Running: {strategy.upper()} @ {unit.upper()}")

        model = ResNetWithHook().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        # 학습 루프
        for epoch in range(epochs):
            model.train()
            for x, y in trainloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()

                features = model.extract_features(x)

                if strategy != 'baseline':
                    R = compute_lrp(model, x, y)
                    A = compute_gradcam(model, x, y)
                    features = apply_strategy(features, R, A, strategy, unit)

                logits = model.classify_from_features(features)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

        # 평가
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(device), y.to(device)
                features = model.extract_features(x)
                if strategy != 'baseline':
                    R = compute_lrp(model, x, y)
                    A = compute_gradcam(model, x, y)
                    features = apply_strategy(features, R, A, strategy, unit)

                logits = model.classify_from_features(features)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = 100. * correct / total
        print(f"✅ Test Accuracy: {acc:.2f}%")

        # 기록 저장
        with open(result_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([strategy, unit, acc])
