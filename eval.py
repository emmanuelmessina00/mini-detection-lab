import torch
import wandb
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F

def evaluate_detection_loss(model, dataloader, device):
    model.train()
    total_loss = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in labels]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

    return total_loss/len(dataloader)

@torch.no_grad()
def log_prediction_samples(model, dataset,device,num_samples=3,score_threshold=0.6):
    model.eval()
    class_names=['background','pedestrian']
    logged_images=[]

    for i in range(num_samples):
        img_tensor, _ = dataset[i]
        predictions=model([img_tensor.to(device)])[0]

        keep=predictions["scores"] >= score_threshold
        pred_boxes=predictions["boxes"][keep].cpu()
        pred_labels=predictions["labels"][keep].cpu()
        pred_scores=predictions["scores"][keep].cpu()

        img_clamped = torch.clamp(img_tensor, 0.0, 1.0)
        img_uint8 = (img_clamped * 255).to(torch.uint8)

        pred_labels_text = [
            f"{class_names[lbl]}: {score:.2f}"
            for lbl, score in zip(pred_labels, pred_scores)
        ]

        if len(pred_boxes) > 0:
            img_with_boxes = draw_bounding_boxes(
                img_uint8,
                pred_boxes,
                labels=pred_labels_text,
                colors="cyan",
                width=3
            )
        else:
            img_with_boxes = img_uint8

        pil_img = F.to_pil_image(img_with_boxes)
        logged_images.append(wandb.Image(pil_img, caption=f"Pedestrian Test {i}"))

    wandb.log({"Pedestrian Predictions": logged_images})
