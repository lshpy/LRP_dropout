import torch
import torch.nn.functional as F

def compute_gradcam(model, input_tensor, target_class=None):
    # 간단한 Grad-CAM 유사 계산 (마지막 conv feature에 대한 gradient)
    input_tensor.requires_grad = True
    features = model.extract_features(input_tensor)
    output = model.classify_from_features(features)

    if target_class is None:
        target_class = output.argmax(dim=1)

    one_hot = F.one_hot(target_class, num_classes=output.size(1)).float()
    output.backward(gradient=one_hot.to(output.device), retain_graph=True)

    grads = features.grad if features.grad is not None else torch.autograd.grad(
        outputs=output, inputs=features,
        grad_outputs=one_hot.to(output.device),
        create_graph=True, retain_graph=True)[0]

    weights = grads.mean(dim=(2, 3), keepdim=True)  # GAP
    cam = (weights * features).sum(dim=1, keepdim=True)  # shape: (B, 1, H, W)
    cam = F.relu(cam)
    cam = cam / (cam.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] + 1e-8)  # normalize
    return cam.detach()
