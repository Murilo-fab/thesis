import os
import csv
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from power_allocation import sum_rate_loss

def nmse_loss(y_pred, y_true):
    y_pred_flat = y_pred.view(y_pred.size(0), -1)
    y_true_flat = y_true.view(y_true.size(0), -1)
    mse = torch.sum((y_true_flat - y_pred_flat)**2, dim=-1)
    normalization = torch.sum(y_true_flat**2, dim=-1)
    return mse / normalization

def train_lwm(model, train_loaders, val_loaders, optimizer, scheduler, epochs, device, save_dir, log_file):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # CSV log
    if not os.path.exists(log_file):
        with open(log_file, mode="w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Train NMSE", "Validation NMSE", "Learning Rate", "Best Model"])

    train_nmse_losses = []
    val_nmse_losses = []
    best_val_nmse = float('inf')

    for epoch in range(epochs):
        model.train()
        train_nmse = 0.0
        train_samples = 0

        print(f"\nEpoch {epoch+1} /{epochs} [Training]")
        for length, train_loader in train_loaders.items():
            print(f"Processing sequences of length {length}")
            with tqdm(train_loaders, desc=f"Length {length} [Training]", unit="batch") as t:
                for batch in t:
                    optimizer.zero_grad()

                    input_ids, masked_tokens, masked_pos = [b.to(device) for b in batch]

                    logits_lm, _, _ = model(input_ids, masked_pos)

                    loss = torch.sum(nmse_loss(masked_tokens, logits_lm))
                    loss.backward()
                    optimizer.setp()
                    scheduler.step()

                    train_nmse += loss.item()
                    train_samples  += input_ids.shape[0]

                    t.set_postfix({"nmse": train_nmse/train_samples, "lr": scheduler.get_last_lr()[0]})

        train_nmse /= max(train_samples, 1)
        train_nmse_losses.append(train_nmse)

        if epoch % 2 == 0:
            model.eval()
            val_nmse = 0.0
            val_samples = 0
            with torch.no_grad():
                print(f"\nEpoch {epoch+1}/{epochs} [Validation]")
                for length, val_loader in val_loaders.item():
                    print(f"Processing sequences of length {length}")
                    with tqdm(val_loader, desc=f"Length {length} [Validation]", unit="batch") as t:
                        for batch in t:
                            input_ids, masked_tokens, masked_pos = [b.to(device) for b in batch]

                            logits_lm, _, _ = model(input_ids, masked_pos)

                            loss = torch.sum(nmse_loss(logits_lm, masked_tokens))
                            val_nmse += loss.item()
                            val_samples += input_ids.shape[0]

                            t.set_postfix({"nmse": val_nmse/val_samples})

            val_nmse /= max(val_samples, 1)
            val_nmse_losses.append(val_nmse)
            
            is_best_model = False
            if val_nmse < best_val_nmse:
                best_val_nmse = val_nmse
                model_path = os.path.join(save_dir, f"lwm_epoch{epoch}_train{train_nmse:.4f}_val{val_nmse:.4f}.pth")
                torch.save(model.state_dict(), model_path)
                print(f"Model saved: {model_path}")
                is_best_model = True

        print(f"\tTrain NMSE: {train_nmse:.4f}")
        print(f"\tValidation NMSE: {val_nmse:.4f}")
        print(f"\tLearning Rate: {scheduler.get_last_lr()[0]:.6f}")

        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, train_nmse, val_nmse, scheduler.get_last_lr()[0], is_best_model])

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_nmse_losses) + 1), train_nmse_losses, label="Train NMSE")
        plt.plot(range(1, len(val_nmse_losses) + 1), val_nmse_losses, label="Validation NMSE")
        plt.xlabel("Epochs")
        plt.ylabel("NMSE")
        plt.title("Training and Validation NMSE Loss")
        plt.legend()
        plt.grid(True)
        plt.show()

    print("Training and validation complete.")
    return model

def train_downstream_model(model: nn.Module,
                           p_total: float,
                           noise_variance: float,
                           train_loader: torch.utils.data.DataLoader,
                           val_loader: torch.utils.data.DataLoader,
                           optimizer: torch.optim.Optimizer,
                           scheduler, 
                           epochs: int,
                           device: torch.device,
                           log_file: str = None,
                           save_dir: str = None):

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    if log_file:
        if not os.path.exists(log_file):
            with open(log_file, mode="w", newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Epoch", "Train Rate", "Validation Rate", "Learning Rate", "Best Model"])

    best_val_loss = float('inf')
    trange_epochs = trange(epochs)

    for epoch in trange_epochs:
        # Training
        model.train()
        train_loss = 0.0

        for batch_idx, (batch_channels,) in enumerate(train_loader):
            batch_channels = batch_channels.to(device)

            optimizer.zero_grad()
            p_pred = model(batch_channels, p_total)

            loss = sum_rate_loss(p_pred, batch_channels, noise_variance)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (batch_channels,) in enumerate(val_loader):
                batch_channels = batch_channels.to(device)

                p_pred = model(batch_channels, p_total)

                loss = sum_rate_loss(p_pred, batch_channels, noise_variance)
                val_loss += loss.item()

        
        val_loss /= len(val_loader)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        trange_epochs.set_postfix({
            "Train Loss": train_loss, 
            "Validation Loss": val_loss})
        
        is_best_model = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_dir:
                model_path = os.path.join(save_dir, f"downstream_epoch{epoch}_train{train_loss:.4f}_val{val_loss:.4f}.pth")
                torch.save(model.state_dict(), model_path)
                print(f"Model saved: {model_path}")
            is_best_model = True
        
        if log_file:
            with open(log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch+1, train_loss, val_loss, scheduler.get_last_lr()[0], is_best_model])
        
    return model
