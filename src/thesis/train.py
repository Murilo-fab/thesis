# General imports
import os
import math
import csv
import trackio
from datetime import datetime
from tqdm import tqdm, trange
from typing import Optional, Dict
from .utils import OptimizerConfigs, TrackioParams

# Torch imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

def nmse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Normalized Mean Squared Error (NMSE) between predictions and targets.
    
    The NMSE is calculated per sample as: ||y_true - y_pred||^2 / ||y_true||^2.

    Inputs:
    y_pred (torch.Tensor): The predicted tensor (Batch, ...).
    y_true (torch.Tensor): The ground truth tensor (Batch, ...).

    Outputs:
    torch.Tensor: A tensor of NMSE values with shape (Batch,).
    """
    # Flatten all dimensions except the batch dimension (dim 0)
    # This ensures we calculate energy over the entire sample, regardless of shape (Seq, Feat, etc.)
    y_pred_flat = y_pred.view(y_pred.size(0), -1)
    y_true_flat = y_true.view(y_true.size(0), -1)

    # Calculate MSE (numerator) and Energy (denominator)
    mse = torch.sum((y_true_flat - y_pred_flat)**2, dim=-1)
    normalization = torch.sum(y_true_flat**2, dim=-1)

    # Avoid division by zero
    normalization = torch.clamp(normalization, min=1e-8)

    return mse / normalization

def train_lwm(model: nn.Module,
              train_loaders: Dict[int, DataLoader],
              val_loaders: Dict[int, DataLoader],
              optimizer: torch.optim.Optimizer,
              scheduler: torch.optim.lr_scheduler.LRScheduler,
              epochs: int,
              device: torch.device,
              save_dir: str,
              trackio_params: Optional[TrackioParams] = None) -> nn.Module:
    """
    Trains a Large Wireless Model (LWM) with support for bucketed DataLoaders (by sequence length).

    Inputs:
    model (nn.Module): The LWM model to train.
    train_loaders (Dict[int, DataLoader]): Dictionary mapping seq_len -> DataLoader.
    val_loaders (Dict[int, DataLoader]): Dictionary mapping seq_len -> DataLoader.
    optimizer (torch.optim.Optimizer): The optimizer.
    scheduler (torch.optim.lr_scheduler.LRScheduler): Learning rate scheduler.
    epochs (int): Number of epochs to train.
    device (torch.device): Computation device.
    save_dir (str): Directory to save model checkpoints.
    trackio_params (TrackioParams, optional): Parameters for experiment tracking.

    Outputs:
    nn.Module: The trained model.
    """
    # 1. Setup Run Directory (standardized timestamp format)
    start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_dir = os.path.join(save_dir, start_time)
    os.makedirs(run_dir, exist_ok=True)

    # 2. Initialize Trackio
    if trackio_params:
        trackio.init(
            project=trackio_params["project"],
            name=trackio_params.get("name", f"LWM_Run_{start_time}"),
            group=trackio_params.get("group", ""),
            config=trackio_params.get("config", {}),
            embed=False
        )

    best_val_nmse = float('inf')

    # 3. Epoch Loop
    for epoch in range(epochs):

        # --- Training Phase ---
        model.train()
        train_nmse_accum = 0.0
        train_samples = 0

        print(f"\nEpoch {epoch+1} /{epochs} [Training]")

        # Iterate over bucketed loaders (Length -> Loader)
        for length, train_loader in train_loaders.items():
            # Use tqdm on the loader itself, not the dict
            with tqdm(train_loader, desc=f"Length {length} [Training]", unit="batch") as t:
                for batch in t:
                    # Move data to device
                    input_ids, masked_tokens, masked_pos = [b.to(device) for b in batch]

                    optimizer.zero_grad()

                    # Forward pass
                    logits_lm, _, _ = model(input_ids, masked_pos)

                    # Calculate loss (NMSE)
                    # Note: nmse_loss returns per-sample loss, so we sum it for the batch
                    loss = torch.sum(nmse_loss(masked_tokens, logits_lm))
                    
                    loss.backward()
                    optimizer.setp()
                    scheduler.step()

                    # Accumulate metrics
                    batch_size = input_ids.shape[0]
                    train_nmse_accum += loss.item() # loss is sum over batch
                    train_samples  += batch_size

                    # Update progress bar
                    current_lr = scheduler.get_last_lr()[0]
                    t.set_postfix({"NMSE": train_nmse_accum/train_samples, "LR": current_lr})

        # Calculate average training NMSE for the epoch
        avg_train_nmse = train_nmse_accum / max(train_samples, 1)

        # --- Validation Phase (Every 2 epochs) ---
        avg_val_nmse = None # Placeholder for logging

        if epoch % 2 == 0:
            model.eval()
            val_nmse_accum = 0.0
            val_samples = 0
            with torch.no_grad():

                print(f"\nEpoch {epoch+1}/{epochs} [Validation]")

                for length, val_loader in val_loaders.item():
                    with tqdm(val_loader, desc=f"Length {length} [Validation]", unit="batch") as t:
                        for batch in t:
                            input_ids, masked_tokens, masked_pos = [b.to(device) for b in batch]

                            logits_lm, _, _ = model(input_ids, masked_pos)

                            loss = torch.sum(nmse_loss(logits_lm, masked_tokens))
                            val_nmse_accum += loss.item()
                            val_samples += input_ids.shape[0]

                            t.set_postfix({"NMSE": val_nmse_accum/val_samples})

            avg_val_nmse = val_nmse_accum / max(val_samples, 1)
            
            if avg_val_nmse < best_val_nmse:
                best_val_nmse = avg_val_nmse
                model_name = f"lwm_best_epoch{epoch+1}_nmse{best_val_nmse:.4f}.pth"
                model_path = os.path.join(run_dir, model_name)
                torch.save(model.state_dict(), model_path)
                print(f"Model saved: {model_path}")

            print(f"\tTrain NMSE: {avg_train_nmse:.4f}")
            print(f"\tValidation NMSE: {avg_val_nmse:.4f}")
            print(f"\tLearning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # --- Logging to Trackio ---
        if trackio_params:
            # Prepare log dict
            log_data = {
                "Train NMSE": avg_train_nmse,
                "Learning Rate": scheduler.get_last_lr()[0]
            }
            # Only log Validation NMSE if we actually ran validation this epoch
            if avg_val_nmse is not None:
                log_data["Validation NMSE"] = avg_val_nmse
            
            trackio.log(log_data)
    
    # 4. Finish Run
    if trackio_params:
        trackio.finish()
        
    print("Training complete.")
    return model

def train_downstream_model(model: nn.Module,
                           train_loader: DataLoader,
                           val_loader: DataLoader,
                           optimizer_configs: OptimizerConfigs,
                           criterion: nn.Module,
                           epochs: int,
                           device: torch.device,
                           save_dir: str,
                           trackio_params: Optional[TrackioParams] = None) -> nn.Module:
    """
    Train the downstream model

    Inputs:
    model (nn.Module): The downstream model to be trained
    train_loader (DataLoader): Loader for training data
    val_loader (DataLoader): Loader for validation data
    optimizer_config (OptimizerConfigs): Configuration dict, e.g., {"task_head_lr": 0.001}
    criterion (nn.Module): Loss function
    epochs (int): Number of training epochs
    device (torch.device): Device to run training on (CPU/GPU)
    save_dir (str): The parent directory where results will be saved
    trackio_params (TrackioParams, optional): Parameters for experiment tracking

    Outputs:
    model (nn.Module): The downstream model trained
    """
    # 1. Get the start time and create the specific run directory
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_dir = os.path.join(save_dir, start_time)
    os.makedirs(run_dir, exist_ok=True)
    log_file = os.path.join(run_dir, "training_log.csv")

    # Initialize the CSV log
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Validation Loss", "Learning Rate"])
    
    # 2. If trackio parameters are provided, initiate the tracker
    if trackio_params:
        trackio.init(
            project=trackio_params["project"],
            name=trackio_params.get("name", f"Run_{start_time}"), # Default to timestamp if no name
            group=trackio_params.get("group", ""),
            config=trackio_params.get("config", {}),
            embed=False
        )

    # 3. Moves model to the device
    model = model.to(device)
    # 4. Create the Optimizer
    optimizer = torch.optim.Adam(model.task_head.parameters(), lr=optimizer_configs["task_head_lr"])
    
    best_val_loss = float('inf')
    trange_epochs = trange(epochs)

    # 5. Start the Epoch Loop
    for epoch in trange_epochs:
        # 6. Training Phase
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

        # 7. Validation Phase
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

        current_lr = optimizer.param_groups[0]['lr']

        # 8. Update Progress Bar
        trange_epochs.set_postfix({
            "Train Loss": train_loss, 
            "Validation Loss": val_loss,
            "LR": current_lr})
        
        # 9. Checkpoint: Save best state dict in memory
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = model.state_dict()

        # 10. Log metrics to Trackio (if enabled)
        if trackio_params:
            trackio.log({
                "Train loss": train_loss,
                "Validation loss": val_loss,
                "Learning rate": current_lr
            })
        # Log results

        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, train_loss, val_loss, current_lr])

    # 11. Save the best model artifact to disk
    model_path = os.path.join(run_dir, f"model_{start_time}.pth")
    torch.save(best_state_dict, model_path)

    # 12. Finish the Trackio run safely
    if trackio_params:
        trackio.finish()

    return model

def cosine_with_warmup_scheduler(optimizer: torch.optim.Optimizer,
                                 warmup_steps: int,
                                 total_steps: int) -> LambdaLR:
    """
    Creates a learning rate scheduler with a warm-up phase followed by a cosine decay.

    During the warm-up phase, the learning rate increases linearly from 0 to the initial learning rate.
    After the warm-up, it decays following a cosine curve down to 0.
    
    Inputs:
    optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
    warmup_steps (int): The number of steps for the linear warm-up phase.
    total_steps (int): The total number of training steps (warm-up + decay).

    Outputs:
    torch.optim.lr_scheduler.LambdaLR: The scheduler object.
    """
    def lr_lambda(step):
        """
        Calculates the multiplicative factor for the learning rate at a specific step.
        """
        # 1. Linear Warm-up Phase
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        # 2. Cosine Decay Phase
        # Calculate how far we are into the decay phase (0.0 to 1.0)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        # 3. Apply Cosine Function
        # The result scales from 1.0 (start of decay) down to 0.0 (end of training)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    # 4. Create and return the LambdaLR scheduler
    return LambdaLR(optimizer, lr_lambda)

def finetune(model: nn.Module,
             train_loader: DataLoader,
             val_loader: DataLoader,
             fine_tune_layers,
             optimizer_configs: OptimizerConfigs,
             criterion: nn.Module,
             epochs: int,
             warmup_epochs: int,
             device: torch.device,
             save_dir: str,
             trackio_params: Optional[TrackioParams] = None) -> nn.Module:
    """
    Finetunes a model with a warm-up phase and layer-specific learning rates.

    The training process consists of two phases:
    1. Warm-up: The model potentially stays frozen (depending on model logic) while the LR warms up.
    2. Fine-tuning: At `warmup_epochs`, specific layers are unfrozen via `model.fine_tune()`.

    Inputs:
    model (nn.Module): The model to fine-tune. Must implement a `.fine_tune(n_layers)` method.
    train_loader (DataLoader): Loader for training data.
    val_loader (DataLoader): Loader for validation data.
    fine_tune_layers (list or str): The number of layers to unfreeze after the warm-up phase.
    optimizer_config (OptimizerConfigs): Dictionary containing "task_head_lr" and "encoder_lr".
    criterion (nn.Module): Loss function.
    epochs (int): Total number of training epochs.
    warmup_epochs (int): Number of epochs before unfreezing layers.
    device (torch.device): Computation device.
    save_dir (str): Parent directory for saving results.
    trackio_params (TrackioParams, optional): Parameters for experiment tracking.

    Outputs:
    model (nn.Module): The model to fine-tuned
    """
    # 1. Setup Run Directory
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_dir = os.path.join(save_dir, start_time)
    os.makedirs(run_dir, exist_ok=True)
    log_file = os.path.join(run_dir, "training_log.csv")

    # Initialize the CSV log
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Validation Loss", "Learning Rate - Head", "Learning Rate - Encoder"])

    # 2. Initialize Trackio
    if trackio_params:
        trackio.init(
            project=trackio_params["project"],
            name=trackio_params.get("name", f"Run_{start_time}"), # Default to timestamp if no name
            group=trackio_params.get("group", ""),
            config=trackio_params.get("config", {}),
            embed=False
        )
    
    model = model.to(device)
    # 3. Set up Optimizer with Parameter Groups
    # We assign different learning rates to the Head vs the Encoder
    optimizer = torch.optim.AdamW([
        {
            "params": model.task_head.parameters(),
            "lr": optimizer_configs["task_head_lr"]
        },
        {
            "params": model.encoder.parameters(),
            "lr": optimizer_configs["encoder_lr"]
        }
    ])
    # 4. Set up Scheduler
    # Calculates total steps based on dataset size
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
        # 5. Unfreeze Logic
        # Trigger specific fine-tuning behavior in the model class once warm-up is done
        if epoch == warmup_epochs:
            model.fine_tune(fine_tune_layers)
        
        # 7. Training Phase
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
            scheduler.step() # Step the scheduler every batch

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # 8. Training Phase
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

        # 9. Retrieve Current Learning Rates
        # param_groups[0] is task_head, [1] is encoder
        head_lr = optimizer.param_groups[0]["lr"]
        encoder_lr = optimizer.param_groups[1]["lr"]

        trange_epochs.set_postfix({
            "Train Loss": train_loss, 
            "Validation Loss": val_loss,
            "Head LR": head_lr,
            "Encoder LR": encoder_lr})
        
        # 10. Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = model.state_dict()

        # 11. Logging
        if trackio_params:
            trackio.log({
                "Train loss": train_loss,
                "Validation loss": val_loss,
                "Learning rate - Head": head_lr,
                "Learning rate - Encoder": encoder_lr,
            })
        
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, train_loss, val_loss, head_lr, encoder_lr])

    # 12. Save Model Artifacts
    model_path = os.path.join(run_dir, f"model_{start_time}.pth")
    torch.save(best_state_dict, model_path)

    if trackio_params:
        trackio.finish()
    
    return model