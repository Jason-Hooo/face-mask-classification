
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dataset import ImageDataset, train_transform, test_transform
from model import EfficientNet 
from pathlib import Path
import argparse

def accuracy_fn(y_pred, y_true):
    """
    Calculates the number of correct predictions.

    Args:
        y_pred (torch.Tensor): Model predictions.
        y_true (torch.Tensor): Ground truth labels.

    Returns:
        int: Number of correct predictions.
    """
    return (y_pred == y_true).sum().item()

def train_step(dataloader, model, cost_fn, optimizer, accuracy_fn, device, epoch_progress, epoch):
    """
    Executes a single training epoch.

    Args:
        dataloader (DataLoader): DataLoader for the training dataset.
        model (nn.Module): The neural network model to train.
        cost_fn (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimization algorithm.
        accuracy_fn (callable): Function to compute accuracy.
        device (str): Device to run computation on ('cpu' or 'cuda').
        epoch_progress (tqdm): Progress bar for the current epoch.
        epoch (int): Current epoch number.
    """
    train_cost = 0
    train_correct = 0
    model.train()
    for (x, y) in dataloader:
        x = x.to(device)
        y = y.to(device)

        y_pred = model(x)
        cost = cost_fn(y_pred, y)
        train_cost += cost.item() * len(x)
        train_correct += accuracy_fn(y_pred.argmax(dim=1), y)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

    total_samples = len(dataloader.dataset)
    train_cost /= total_samples
    train_acc = (train_correct / total_samples) * 100

    epoch_progress.write(f"Epoch {epoch}: Train Cost: {train_cost:.4f}, Train Acc: {train_acc:.2f}%")

def test_step(dataloader, model, cost_fn, accuracy_fn, device, epoch_progress, epoch):
    """
    Evaluates the model on the test dataset.

    Args:
        dataloader (DataLoader): DataLoader for the testing dataset.
        model (nn.Module): The neural network model to evaluate.
        cost_fn (nn.Module): Loss function.
        accuracy_fn (callable): Function to compute accuracy.
        device (str): Device to run computation on ('cpu' or 'cuda').
        epoch_progress (tqdm): Progress bar for the current epoch.
        epoch (int): Current epoch number.

    Returns:
        tuple: (test_cost, test_acc) representing the average loss and accuracy percentage.
    """
    test_cost = 0
    test_correct = 0
    model.eval()

    with torch.inference_mode():
        for (x, y) in dataloader:
            x = x.to(device)
            y = y.to(device)
            test_pred = model(x)
            test_cost += cost_fn(test_pred, y).item() * len(x)
            test_correct += accuracy_fn(test_pred.argmax(dim=1), y)

    total_samples = len(dataloader.dataset)
    test_cost /= total_samples
    test_acc = (test_correct / total_samples) * 100

    epoch_progress.write(f"Epoch {epoch}: Test Cost: {test_cost:.4f}, Test Acc: {test_acc:.2f}%")

    return test_cost, test_acc
    
    
def train_model(args):
    """
    Main pipeline for initializing data, model, and executing the training loop.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing hyperparameters.
    """
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.manual_seed_all(args.seed)
    else:
        device = "cpu"

    tqdm.write(f"Device: {device}")
    tqdm.write(f"Dataset: {args.dataset_root}")
    tqdm.write(f"Model: EfficientNet-{args.model_name}")
    tqdm.write(f"Batch Size: {args.batch_size}, Epochs: {args.epochs}, LR: {args.learning_rate}")
    tqdm.write("-" * 50)

    train_dataset = ImageDataset(args.dataset_root, train=True, transform=train_transform)
    test_dataset = ImageDataset(args.dataset_root, train=False, transform=test_transform)

    num_workers = min(4, os.cpu_count() // 2) if device == "cuda" else 0
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False
    )

    NUM_CLASSES = len(train_dataset.classes)

    model = EfficientNet(model_name=args.model_name, out_classes=NUM_CLASSES)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    best_test_acc = 0.0

    epoch_progress = tqdm(range(args.epochs), desc="Training", unit="epoch")

    for epoch in epoch_progress:
        epoch_progress.set_postfix_str(f"Best Acc: {best_test_acc:.2f}%")

        train_step(train_dataloader, model, loss_fn, optimizer, accuracy_fn, device, epoch_progress, epoch + 1)

        test_cost, test_acc = test_step(test_dataloader, model, loss_fn, accuracy_fn, device, epoch_progress, epoch + 1)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            epoch_progress.write(f"✓ Epoch {epoch + 1}: Better model found! Best Test Acc: {best_test_acc:.2f}%")

            save_path = Path(args.save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(model.state_dict(), save_path)

        epoch_progress.write("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet model for face mask classification")
    parser.add_argument("--dataset_root", type=str, default="Face_Mask_Dataset",
                        help="Path to dataset directory")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate for optimizer")
    parser.add_argument("--model_name", type=str, default="B1",
                        choices=["B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7"],
                        help="EfficientNet model variant")
    parser.add_argument("--save_path", type=str, default="weights/trained_model_parameters.pth",
                        help="Path to save trained model weights")
    parser.add_argument("--seed", type=int, default=87,
                        help="Random seed for reproducibility")

    args = parser.parse_args()
    train_model(args)
