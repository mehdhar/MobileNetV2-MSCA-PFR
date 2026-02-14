import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from model.model_msca_pfr import MobileNetV2_MSCA_PFR   # your model


def load_image(img_path, size=224):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    return img, tensor


def visualize_gradcam(model_path, image_path, output_path="gradcam_result.png", class_index=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = MobileNetV2_MSCA_PFR(num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load image
    img_rgb, img_tensor = load_image(image_path)
    img_tensor = img_tensor.to(device)

    # Target layer: last block before pooling
    target_layer = model.msca_block  # adjust if needed

    # Initialize Grad-CAM
    cam = GradCAM(model=model, target_layers=[target_layer])

    # Define class target
    targets = [ClassifierOutputTarget(class_index)] if class_index is not None else None

    # Compute CAM
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]

    # Normalize image for overlay
    img_np = img_tensor.cpu()[0].permute(1, 2, 0).numpy()
    img_np = img_np - img_np.min()
    img_np = img_np / img_np.max()

    # Generate heatmap overlay
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    # Save
    plt.figure(figsize=(6, 6))
    plt.imshow(visualization)
    plt.axis("off")
    plt.title("Grad-CAM")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Grad-CAM saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    model_path = "checkpoints/best_model.pth"
    image_path = "example.jpg"
    visualize_gradcam(model_path, image_path, class_index=0)
