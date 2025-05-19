import torch
from torchvision import datasets, transforms
from model import ResNetWithHook
from strategy import apply_strategy
from utils.lrp import compute_lrp
from utils.gradcam import compute_gradcam

# --- 하이퍼파라미터 ---
strategy_type = 'hybrid_drop'  # 예: 'random', 'suppressive', 'hybrid_amp'
unit = 'patch'                 # 'pixel', 'patch', 'channel'
epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 데이터셋 로딩 ---
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# --- 모델 정의 ---
model = ResNetWithHook().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --- 학습 루프 ---
for epoch in range(epochs):
    model.train()
    for x, y in trainloader:
        x, y = x.to(device), y.to(device)
        
        features = model.extract_features(x)  # 마지막 conv 출력
        
        R = compute_lrp(model, x, y)          # relevance score
        A = compute_gradcam(model, x, y)      # grad-cam score
        
        features = apply_strategy(features, R, A, strategy_type, unit)  # 전략 적용
        logits = model.classify_from_features(features)
        
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch + 1}/{epochs} complete.")
