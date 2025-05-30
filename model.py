from torch.utils.data import DataLoader
from dataset import LichessStream, collate
from pathlib import Path
import torch
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 75000
num_epochs = 8

learning_rate = 0.0001

# ----- Dataset ----------------------------------

# Get the batches of preprocessed data
batches = list(Path("data/split_data").glob("batch_*"))

# Create train/test split
split = int(0.8 * len(batches))

# train_stream = LichessStream(batches=batches[:split])
# valid_stream = LichessStream(batches=batches[split:])

train_stream = LichessStream(batches=batches[:1])
valid_stream = LichessStream(batches=batches[1:2])

train_loader = DataLoader(
    train_stream,
    batch_size=batch_size,
    # num_workers=3,
    collate_fn=collate,
    # prefetch_factor=4,
    pin_memory=True,
    drop_last=True,
)

valid_loader = DataLoader(
    valid_stream,
    batch_size=batch_size,
    # num_workers=1,
    collate_fn=collate,
    # prefetch_factor=4,
    pin_memory=True,
    drop_last=True,
)


# Need to come up with a unified "vocab"
idx_to_move = train_stream.idx_to_move


# ----- Model -----------------------------------
class ImitationModel(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(ImitationModel, self).__init__()

        self.seq = torch.nn.Sequential(
            torch.nn.Linear(in_features, 20_000),
            torch.nn.ReLU(),
            torch.nn.Linear(20_000, 10_000),
            torch.nn.ReLU(),
            torch.nn.Linear(10_000, 5_000),
            torch.nn.ReLU(),
            torch.nn.Linear(5_000, out_features)
        )

    def forward(self, x):
        x = self.seq(x)
        return x


model = ImitationModel(in_features=837, out_features=len(idx_to_move)).to(device)


# ----- Logging -----------------------------------
run = wandb.init(
    entity="chess-ai",
    project="Imitation-Model",
    config={
        "architecture": "MLP",
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": num_epochs,
        "num_params": sum(p.numel() for p in model.parameters()),
    },
)


# ----- Training -----------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_fn = torch.nn.CrossEntropyLoss()

scaler = torch.amp.GradScaler()

for epoch in range(num_epochs):

    # ---- TRAIN ----
    model.train()
    train_loss = train_correct = train_seen = 0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device, dtype=torch.float32), batch_y.to(device).long()

        with torch.amp.autocast(device_type=str(device)):
            preds = model(batch_x)
            loss  = loss_fn(preds, batch_y)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item() * batch_y.size(0)

        batch_correct = (preds.argmax(dim=1) == batch_y).sum().item()
        batch_seen = batch_y.size(0)

        train_correct += batch_correct
        train_seen += batch_seen

        run.log({"train/batch_loss": loss.item(), "train/batch_acc": batch_correct / batch_seen * 100.0})

    epoch_train_loss = train_loss / train_seen
    epoch_train_acc  = train_correct / train_seen * 100

    # ---- VAL ----
    model.eval()
    val_loss = val_correct = val_seen = 0
    with torch.no_grad(), torch.amp.autocast(device_type=str(device)):
        for batch_x, batch_y in valid_loader:
            batch_x, batch_y = batch_x.to(device, dtype=torch.float32), batch_y.to(device).long()
            preds = model(batch_x)
            loss  = loss_fn(preds, batch_y)

            val_loss   += loss.item() * batch_y.size(0)
            val_correct += (preds.argmax(1) == batch_y).sum().item()
            val_seen   += batch_y.size(0)

    epoch_val_loss = val_loss / val_seen
    epoch_val_acc  = val_correct / val_seen * 100

    run.log({
        "epoch": epoch,
        "train/loss": epoch_train_loss,
        "train/acc":  epoch_train_acc,
        "val/loss":   epoch_val_loss,
        "val/acc":    epoch_val_acc,
        "lr": optimizer.param_groups[0]["lr"],
    })


run.finish()
