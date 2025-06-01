"""
Trainigs imitation learning model
"""

import json
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import wandb
from dataset import LichessDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 40000
NUM_EPOCHS = 2

LEARNING_RATE = 0.0001

# ----- Dataset ----------------------------------

batches = list(Path("data/split_data").glob("batch_*"))[:20]
target_mappings = Path("data/split_data/distinct_moves.json")

split = int(0.8 * len(batches))

train_dataset = LichessDataset(
    batches=batches[:split], target_mappings=target_mappings, batch_size=BATCH_SIZE
)
valid_dataset = LichessDataset(
    batches=batches[split:], target_mappings=target_mappings, batch_size=BATCH_SIZE
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

with open(target_mappings, "r", encoding="utf-8") as f:
    idx_to_move = json.load(f)


# ----- Model -----------------------------------
class ImitationModel(torch.nn.Module):
    """Simple MLP for chess imitation model"""

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
        """Forward pass on model"""
        x = self.seq(x)
        return x


model = ImitationModel(in_features=837, out_features=len(idx_to_move)).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

loss_fn = torch.nn.CrossEntropyLoss()


# ---- Logging ---------------------------------
run = wandb.init(
    entity="chess-ai",
    project="Imitation-Model",
    config={
        "architecture": "MLP",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
        "num_params": sum(p.numel() for p in model.parameters()),
    },
)


# ---- Training ---------------------------------

for epoch in range(NUM_EPOCHS):
    model.train()

    REPORT_EVERY = 10
    RUNNING_LOSS = 0.0
    RUNNING_CORRECT = 0
    RUNNING_TOTAL = 0

    for batch_idx, (inputs, targets) in enumerate(train_dataloader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        # accumulate
        RUNNING_LOSS += loss.item() * targets.size(0)  # sum, not mean
        RUNNING_CORRECT += (outputs.argmax(1) == targets).sum().item()
        RUNNING_TOTAL += targets.size(0)

        if (batch_idx + 1) % REPORT_EVERY == 0:
            run.log(
                {
                    "train/loss": RUNNING_LOSS / RUNNING_TOTAL,
                    "train/acc%": RUNNING_CORRECT / RUNNING_TOTAL * 100,
                }
            )
            RUNNING_LOSS = RUNNING_CORRECT = RUNNING_TOTAL = 0

    # ---------- VALIDATION ----------
    model.eval()
    VAL_LOSS = VAL_CORRECT = VAL_TOTAL = 0

    with torch.no_grad():
        for v_inputs, v_targets in valid_dataloader:
            v_inputs = v_inputs.to(device, non_blocking=True)
            v_targets = v_targets.to(device, non_blocking=True)

            preds = model(v_inputs)
            loss = loss_fn(preds, v_targets)

            VAL_LOSS += loss.item() * v_targets.size(0)
            VAL_CORRECT += (preds.argmax(1) == v_targets).sum().item()
            VAL_TOTAL += v_targets.size(0)

    run.log(
        {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "val/loss": VAL_LOSS / VAL_TOTAL,
            "val/acc%": VAL_CORRECT / VAL_TOTAL * 100,
        }
    )


torch.save(model.state_dict(), Path("models/v1"))


run.finish()
