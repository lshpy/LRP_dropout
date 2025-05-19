import torch
import torch.nn.functional as F

def compute_lrp(model, input_tensor, target):
    # 마지막 conv layer feature 기준 LRP 예시 (Captum 대체 가능)
    # 여기선 simple gradient * input 기반 유사 relevance 구현
    input_tensor.requires_grad = True
    output = model(input_tensor)
    one_hot = F.one_hot(target, num_classes=output.size(1)).float()
    output.backward(gradient=one_hot.to(output.device), retain_graph=True)

    relevance = input_tensor.grad * input_tensor  # relevance ≈ gradient * input
    relevance = relevance.abs().mean(dim=1, keepdim=True)  # 채널 평균
    return relevance.detach()
