from torch.utils.data import DataLoader, random_split
from dataset import LichessDataset
import torch
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 2048
num_epochs = 256

learning_rate = 0.0001

# ----- Dataset ----------------------------------

dataset = LichessDataset("data/split_data")

train_data, test_data = random_split(dataset, [0.8, 0.2])

train_loader = DataLoader(train_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

idx_to_move = dataset.move_to_idx


# ----- Model -----------------------------------
class ImitationModel(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(ImitationModel, self).__init__()
        self.linear1 = torch.nn.Linear(in_features, 20_000)
        self.linear2 = torch.nn.Linear(20_000, 10_000)
        self.linear3 = torch.nn.Linear(10_000, 5_000)
        self.linear4 = torch.nn.Linear(5_000, out_features)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.activation(x)
        x = self.linear4(x)
        x = self.activation(x)

        return x


model = ImitationModel(
    in_features=837, 
    out_features=len(idx_to_move)
).to(device)


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

for epoch in range(num_epochs):

    model.train()
    total_loss = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        preds = model(batch_x)
        loss = loss_fn(preds, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss

    loss = total_loss / len(train_loader)

    run.log({"loss": loss})

run.finish()
