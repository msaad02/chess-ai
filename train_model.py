"""
Train an imitation learning model on a dataset of chess positions and moves.

This script loads training and validation data from pre-split .npz files,
initializes a multilayer perceptron (MLP) model for move prediction, and
trains it using cross-entropy loss and the Adam optimizer. Optionally logs
metrics to Weights & Biases.

Functions
---------
get_split_dataloader(split_data_dir, distinct_moves_path, batch_size, proportion_train)
    Returns DataLoaders for training and validation datasets.

training_loop(dataloader, model, optimizer, loss_fn, device, wandb_run)
    Runs a single training epoch and logs metrics to Weights & Biases.

validation_loop(dataloader, model, optimizer, loss_fn, device, wandb_run)
    Evaluates the model on validation data and logs metrics to Weights & Biases.

train_model(model_save_dir, split_data_dir, distinct_moves_path, batch_size,
            num_epochs, learning_rate, proportion_train, device, report_to_wandb)
    End-to-end training loop: loads data, initializes model, trains for multiple epochs,
    evaluates, logs metrics, and saves model artifacts.

Usage
-----
Run this script from the command line with configurable arguments:

    python train_model.py --model_version 1 --batch_size 40000 --num_epochs 10

By default, models are saved in `models/model_vX/`, and metrics are logged to Weights & Biases.
"""

import json
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import wandb
from dataset import LichessDataset
from model import ImitationModel


def get_split_dataloader(
    split_data_dir: Path,
    distinct_moves_path: Path,
    batch_size: int,
    proportion_train: float = 0.8,
    max_num_batches: int = -1,
) -> tuple[DataLoader, DataLoader]:
    """
    Returns an IterableDataset wrapped in DataLoaders for training and validation.

    Parameters
    ----------
    split_data_dir : Path
        Directory containing batch_XX.npz files.
    distinct_moves_path : Path
        Path to JSON file mapping indices to UCI moves.
    batch_size : int
        Number of samples to load per batch (used internally in dataset).
    proportion_train : float, optional
        Proportion of batches to use for training (default is 0.8).
    max_num_batches : int, optional
        If != -1, will filter to max_num_batches batches for training/validation.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        Tuple of (train_loader, validation_loader).
    """

    batches = list(split_data_dir.glob("batch_*"))

    if max_num_batches:
        batches = batches[:max_num_batches]

    split = int(proportion_train * len(batches))

    train_dataset = LichessDataset(
        batches=batches[:split],
        target_mappings=distinct_moves_path,
        batch_size=batch_size,
    )
    valid_dataset = LichessDataset(
        batches=batches[split:],
        target_mappings=distinct_moves_path,
        batch_size=batch_size,
    )

    # NOTE: batch_size=None since batching is manully implemented in the dataset
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

    return train_dataloader, valid_dataloader


def training_loop(
    dataloader: DataLoader,
    model: ImitationModel,
    optimizer: torch.optim.Adam,
    loss_fn: torch.nn.CrossEntropyLoss,
    device: str,
    wandb_run=None,
) -> tuple[ImitationModel, torch.optim.Adam]:
    """
    Runs a single training epoch for the given model and logs performance metrics.

    The function iterates over the training data, performs forward and backward passes,
    updates model weights, and optionally logs loss and accuracy metrics to Weights & Biases.

    Parameters
    ----------
    dataloader : DataLoader
        DataLoader providing batches of input tensors and corresponding targets.
    model : ImitationModel
        Model to be trained.
    optimizer : torch.optim.Adam
        Optimizer used to update the model parameters.
    loss_fn : torch.nn.CrossEntropyLoss
        Loss function to compute training loss.
    device : str
        Device on which to perform training (e.g., "cuda" or "cpu").
    wandb_run : wandb.wandb_run.Run
        An active Weights & Biases run object used for logging metrics. If None, logging is skipped.

    Returns
    -------
    tuple of (ImitationModel, torch.optim.Adam)
        The updated model and optimizer after the training loop.
    """

    model.train()

    report_every = 10
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        if wandb_run:
            running_loss += loss.item() * targets.size(0)  # sum, not mean
            running_correct += (outputs.argmax(1) == targets).sum().item()
            running_total += targets.size(0)

            if (batch_idx + 1) % report_every == 0:
                wandb_run.log(
                    {
                        "train/loss": running_loss / running_total,
                        "train/acc%": running_correct / running_total * 100,
                    }
                )
                running_loss = running_correct = running_total = 0

    return model, optimizer


def validation_loop(
    dataloader: DataLoader,
    model: ImitationModel,
    optimizer: torch.optim.Adam,
    loss_fn: torch.nn.CrossEntropyLoss,
    device: str,
    wandb_run=None,
) -> None:
    """
    Evaluates the model on the validation set and logs loss and accuracy.

    The function runs a full pass over the validation DataLoader without gradient computation.
    Loss and accuracy are accumulated across the entire dataset. Metrics are optionally logged
    to Weights & Biases.

    Parameters
    ----------
    dataloader : DataLoader
        DataLoader providing batches of input tensors and corresponding targets for validation.
    model : ImitationModel
        Model to be evaluated.
    optimizer : torch.optim.Adam
        Optimizer used during training; used here only for logging the learning rate.
    loss_fn : torch.nn.CrossEntropyLoss
        Loss function to compute validation loss.
    device : str
        Device on which to perform validation (e.g., "cuda" or "cpu").
    wandb_run :
        An active Weights & Biases run object used for logging metrics. If None, logging is skipped.

    Returns
    -------
    None
    """

    model.eval()
    val_loss = val_correct = val_total = 0

    with torch.no_grad():
        for v_inputs, v_targets in dataloader:
            v_inputs = v_inputs.to(device, non_blocking=True)
            v_targets = v_targets.to(device, non_blocking=True)

            preds = model(v_inputs)
            loss = loss_fn(preds, v_targets)

            val_loss += loss.item() * v_targets.size(0)
            val_correct += (preds.argmax(1) == v_targets).sum().item()
            val_total += v_targets.size(0)

    if wandb_run:
        wandb_run.log(
            {
                "lr": optimizer.param_groups[0]["lr"],
                "val/loss": val_loss / val_total,
                "val/acc%": val_correct / val_total * 100,
            }
        )


def train_model(
    model_save_dir: Path,
    split_data_dir: Path,
    distinct_moves_path: Path,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    proportion_train: float = 0.8,
    device: str = "cuda",
    report_to_wandb: bool = False,
    max_num_batches: int = -1,
):
    """
    Oversees model training for provided dataset.

    Parameters
    ----------
    model_save_dir: Path
        Directory to save model and idx to target mappings
    split_data_dir : Path
        Directory containing batch_XX.npz files.
    distinct_moves_path : Path
        Path to JSON file mapping indices to UCI moves.
    batch_size : int
        Batch size used for training.
    num_epochs : int
        Number of epochs to train the model for.
    learning_rate : float
        Learning rate to use for optimizer.
    proportion_train : float, optional
        Proportion of data used for training vs validation (default is 0.8).
    device : str, optional
        Device to run training on (e.g. 'cuda' or 'cpu').
    report_to_wandb : bool, optional
        Whether to report model training information to weight and biases website.
    max_num_batches : int, optional
        If != -1, will filter to max_num_batches batches for training/validation.
    """

    with open(distinct_moves_path, "r", encoding="utf-8") as f:
        idx_to_move = json.load(f)

    model = ImitationModel(in_features=837, out_features=len(idx_to_move))

    if device == "cuda":
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_fn = torch.nn.CrossEntropyLoss()

    train_dataloader, valid_dataloader = get_split_dataloader(
        split_data_dir=split_data_dir,
        distinct_moves_path=distinct_moves_path,
        batch_size=batch_size,
        proportion_train=proportion_train,
        max_num_batches=max_num_batches,
    )

    if report_to_wandb:
        wandb_run = wandb.init(
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
    else:
        wandb_run = None

    for epoch in range(num_epochs):

        # Perform training for this epoch
        model, optimizer = training_loop(
            dataloader=train_dataloader,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            wandb_run=wandb_run,
        )

        # Check validation performance for this epoch
        validation_loop(
            dataloader=valid_dataloader,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            wandb_run=wandb_run,
        )

        if wandb_run:
            wandb_run.log({"epoch": epoch})

    # Save the model and target map
    model_save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), model_save_dir / "model.pt")

    with open(model_save_dir / "target_mapping.json", "w", encoding="utf-8") as f:
        json.dumps(idx_to_move, indent=4)

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_save_dir", type=str, default="models/")
    parser.add_argument("--model_version", type=int)
    parser.add_argument("--split_data_dir", type=str, default="data/split_data")
    parser.add_argument(
        "--distinct_moves_path", type=str, default="data/split_data/distinct_moves.json"
    )
    parser.add_argument("--batch_size", type=int, default=40000)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--proportion_train", type=float, default=0.85)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--report_to_wandb", type=bool, default=True)
    parser.add_argument("--max_num_batches", type=int, default=None)
    args = parser.parse_args()

    train_model(
        model_save_dir=Path(args.model_save_dir) / f"model_v{args.model_version}",
        split_data_dir=Path(args.split_data_dir),
        distinct_moves_path=Path(args.distinct_moves_path),
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        proportion_train=args.proportion_train,
        device=args.device,
        max_num_batches=args.max_num_batches,
    )
