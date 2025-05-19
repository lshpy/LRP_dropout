import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_mask(image_tensor, mask_tensor, save_path, title="Dropout Mask"):
    img = image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    mask = mask_tensor.squeeze().cpu().numpy()

    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title("Original")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='hot', interpolation='nearest')
    plt.axis("off")
    plt.title(title)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
