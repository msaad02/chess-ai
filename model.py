from torch.utils.data import DataLoader
from dataset import LichessDataset
from pathlib import Path
import torch
import wandb
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 40000
num_epochs = 2

learning_rate = 0.0001

# ----- Dataset ----------------------------------

batches = list(Path("data/split_data").glob("batch_*"))[:20]
target_mappings = Path("data/split_data/distinct_moves.json")

split = int(0.8 * len(batches))

train_dataset = LichessDataset(
    batches=batches[:split], target_mappings=target_mappings, batch_size=batch_size
)
valid_dataset = LichessDataset(
    batches=batches[split:], target_mappings=target_mappings, batch_size=batch_size
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=None,
    num_workers=4,
    prefetch_factor=3,
    pin_memory=True,
)

valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=None,
    num_workers=2,
    prefetch_factor=1,
    pin_memory=True,
)

with open(target_mappings, "r") as f:
    idx_to_move = json.load(f)


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
            torch.nn.Linear(5_000, out_features),
        )

    def forward(self, x):
        x = self.seq(x)
        return x


model = ImitationModel(in_features=837, out_features=len(idx_to_move)).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_fn = torch.nn.CrossEntropyLoss()


# ---- Logging ---------------------------------
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


# ---- Training ---------------------------------

for epoch in range(num_epochs):
    model.train()

    report_every = 10
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        # accumulate
        running_loss += loss.item() * targets.size(0)  # sum, not mean
        running_correct += (outputs.argmax(1) == targets).sum().item()
        running_total += targets.size(0)

        if (batch_idx + 1) % report_every == 0:
            run.log(
                {
                    "train/loss": running_loss / running_total,
                    "train/acc%": running_correct / running_total * 100,
                }
            )
            running_loss = running_correct = running_total = 0

    # ---------- VALIDATION ----------
    model.eval()
    val_loss = val_correct = val_total = 0

    with torch.no_grad():
        for v_inputs, v_targets in valid_dataloader:
            v_inputs = v_inputs.to(device, non_blocking=True)
            v_targets = v_targets.to(device, non_blocking=True)

            preds = model(v_inputs)
            loss = loss_fn(preds, v_targets)

            val_loss += loss.item() * v_targets.size(0)
            val_correct += (preds.argmax(1) == v_targets).sum().item()
            val_total += v_targets.size(0)

    run.log(
        {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "val/loss": val_loss / val_total,
            "val/acc%": val_correct / val_total * 100,
        }
    )


torch.save(model.state_dict(), Path("models/v1"))


run.finish()
