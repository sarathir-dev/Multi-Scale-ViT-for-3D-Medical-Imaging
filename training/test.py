# Testing Script
import torch
from models.vit3d import SimpleViT
from datasets.organmnist3d import get_dataloader
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleViT(volume_size=(28, 28, 28),
                  patch_size=(7, 7, 7),
                  num_classes=11,
                  dim=512,
                  depth=4,
                  heads=8,
                  mlp_dim=1024,
                  channels=1).to(device)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

dataloaders = get_dataloader()
test_loader = dataloaders["test"]

preds, labels = [], []
with torch.no_grad():
    for batch in test_loader:
        images, targets = batch[0].to(device), batch[1].to(device)
        preds.extend(torch.argmax(model(images), dim=1).cpu().numpy())
        labels.extend(targets.cpu().numpy())

print(f"Test Accuracy: {accuracy_score(labels, preds):.4f}")
