import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from model.model_msca_pfr import MobileNetV2_MSCA_PFR   # <-- your unified model


def load_image(path, size=224):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    return img, tensor


def compute_saliency(model, img_tensor, target_class=None):
    img_tensor = img_tensor.clone().detach().requires_grad_(True)

    output = model(img_tensor)
    if target_class is None:
        target = output.max(1)[0]      
    else:
        target = output[:, target_class]

    model.zero_grad()
    target.backward()

    saliency = img_tensor.grad.abs().squeeze().cpu()
    saliency = saliency.max(dim=0)[0]       
    return saliency


def visualize_saliency(model_path, image_path,
                       output_path="saliency_map.png",
                       class_index=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = MobileNetV2_MSCA_PFR(num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load image
    original_img, img_tensor = load_image(image_path)
    img_tensor = img_tensor.to(device)

    # Compute saliency
    saliency = compute_saliency(model, img_tensor, target_class=class_index)
    saliency = saliency / saliency.max()   # normalize

    # Convert original img for overlay
    display_img = np.array(original_img.resize((224, 224))) / 255.0

    # Plot
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(display_img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(display_img, alpha=0.6)
    plt.imshow(saliency, cmap="hot", alpha=0.6)
    plt.title("Saliency Map")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saliency map saved to: {output_path}")


if __name__ == "__main__":
    model_path = "checkpoints/best_model.pth"
    image_path = "example.jpg"
    visualize_saliency(model_path, image_path, class_index=None)
