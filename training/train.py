# Training script
import torch
from torch import nn
from torch.optim import Adam
from models.vit3d import SimpleViT
from datasets.organmnist3d import get_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleViT(
    volume_size=(28, 28, 28),
    patch_size=(7, 7, 7),
    num_classes=11,
    dim=512,
    depth=4,
    heads=8,
    mlp_dim=1024,
    channels=1
).to(device)

dataloaders = get_dataloader()
train_loader, val_loader = dataloaders["train"], dataloaders["val"]

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    model.train()
    for batch in train_loader:
        images, labels = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch: {epoch + 1}")

torch.save(model.state_dict(), "best_model.pth")
