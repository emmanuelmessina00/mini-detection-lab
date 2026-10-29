import torch
import wandb
from torch.utils.data import random_split, DataLoader

from data.download_data import download_data
from dataset.pedestrian_dataset import PennFudanDataset
from eval import log_prediction_samples, evaluate_detection_loss
from models.detector import get_pedestrian_detector
from train import train_one_epoch
from utils.stats import collate_fn
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Dispositivo attivo: {device}")

    download_data()

    full_dataset = PennFudanDataset()
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    wandb.init(
        project="Penn-Fudan-Recognition-Project",
        name="faster-rcnn-pedestrians",
        config={
            "architecture": "Faster-RCNN-ResNet50-FPN",
            "epochs": 4,
            "batch_size": 4,
            "lr": 0.005
        }
    )

    model = get_pedestrian_detector(num_classes=2)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    epochs = 4
    print("\n[*] Avvio training...")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        val_loss = evaluate_detection_loss(model, test_loader, device)

        print(f"Epoca [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

    # 6. Visualizzazione finale
    print("[*] Invio campioni visivi di test su WandB...")
    log_prediction_samples(model, test_dataset, device, num_samples=4, score_thresh=0.6)

    wandb.finish()
    print("[*] Sessione completata!")
if __name__ == "__main__":
    main()