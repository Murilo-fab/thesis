# General imports
import os
import csv
import math
from datetime import datetime
from tqdm import tqdm, trange

# Torch imports
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

# Matplotlib
import matplotlib.pyplot as plt


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
                           train_loader,
                           val_loader,
                           optimizer_config,
                           criterion,
                           epochs,
                           device,
                           results_folder):

    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    results_folder = f"{results_folder}/{time_now}"
    os.makedirs(results_folder, exist_ok=True)
    log_file = os.path.join(results_folder, "training_log.csv")

    with open(log_file, mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Rate", "Validation Rate", "Learning Rate", "Time"])
    
    model = model.to(device)

    optimizer = torch.optim.Adam(model.task_head.parameters(), lr=optimizer_config["task_head_lr"])

    best_val_loss = float('inf')
    trange_epochs = trange(epochs)

    for epoch in trange_epochs:
        # Training
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch_channels = batch[0].to(device)
            batch_tokens = batch[1].to(device)

            optimizer.zero_grad()
            pred = model(batch_tokens)

            loss = criterion(pred, batch_channels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch_channels = batch[0].to(device)
                batch_tokens = batch[1].to(device)

                pred = model(batch_tokens)

                loss = criterion(pred, batch_channels)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        time_now = datetime.now().strftime("%Y-%m-%d_%H-%M")
        
        current_lr = optimizer.param_groups[0]['lr']

        trange_epochs.set_postfix({
            "Train Loss": train_loss, 
            "Validation Loss": val_loss,
            "LR": current_lr})
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = model.state_dict()

        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, train_loss, val_loss, current_lr, time_now])
        
    model_path = os.path.join(results_folder, f"model_{time_now}.pth")
    torch.save(best_state_dict, model_path)

    return model

def cosine_with_warmup_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)

def finetune(model,
             train_loader,
             val_loader,
             fine_tune_layers,
             optimizer_config,
             criterion,
             epochs,
             warmup_epochs,
             device,
             results_folder):
    
    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M")
    results_folder = f"{results_folder}/{time_now}"
    os.makedirs(results_folder, exist_ok=True)
    log_file = os.path.join(results_folder, "training_log.csv")

    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Validation Loss", "Learning Rate - Task Head", "Learning Rate - Encoder", "Time"])
    
    model = model.to(device)
    # Set up optimizer
    optimizer = torch.optim.AdmaW([
        {
            "params": model.task_head.parameters(),
            "lr": optimizer_config["task_head_lr"]
        },
        {
            "params": model.encoder.parameters(),
            "lr": optimizer_config["encoder_lr"]
        }
    ])
    # Set up scheduler with linear warmup and cosine
    steps_per_epoch = len(train_loader)
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = epochs * steps_per_epoch

    scheduler = cosine_with_warmup_scheduler(
        optimizer,
        warmup_steps,
        total_steps
    )

    best_val_loss = float('inf')
    trange_epochs = trange(epochs)
    for epoch in trange_epochs:
        if epoch == warmup_epochs:
            model.fine_tune(fine_tune_layers)
        
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch_channels = batch[0].to(device)
            batch_tokens = batch[1].to(device)

            optimizer.zero_grad()
            pred = model(batch_tokens)

            loss = criterion(pred, batch_channels)
            loss.backward()

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch_channels = batch[0].to(device)
                batch_tokens = batch[1].to(device)

                pred = model(batch_tokens)

                loss = criterion(pred, batch_channels)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        time_now = datetime.now().strftime("%Y-%m-%d_%H-%M")

        head_lr = optimizer.param_groups[0]["lr"]
        encoder_lr = optimizer.param_groups[1]["lr"]

        trange_epochs.set_postfix({
            "Train Loss": train_loss, 
            "Validation Loss": val_loss,
            "Head LR": head_lr,
            "Encoder LR": encoder_lr})
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = model.state_dict()

        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, train_loss, val_loss, head_lr, encoder_lr, time_now])

    model_path = os.path.join(results_folder, f"model_{time_now}.pth")
    torch.save(best_state_dict, model_path)
    
    return model