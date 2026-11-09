import os
import csv
import math

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from dataset import ShapesDataset
from models import (
    JEPAEncoders,
    JEPAPredictor,
    CNNDecoder,
    combine_features,
)
from losses import (
    jepa_loss,
    reconstruction_loss,
    combined_loss,
)


# ============================================================
# Configuration
# ============================================================

BATCH_SIZE = 16
EPOCHS = 200
DATASET_SIZE = 8000
VAL_SIZE = 400

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

LAMBDA_RECONSTRUCTION = 1.0
EMA_DECAY = 0.996

NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Keep these separate from the previous experiment.
CHECKPOINT_DIR = "outputs/checkpoints_16x16"
SAMPLE_DIR = "outputs/samples_16x16"
METRICS_DIR = "outputs/metrics_16x16"

LATEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "latest.pth")
BEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_model.pth")
CSV_PATH = os.path.join(METRICS_DIR, "training_history.csv")

for directory in (CHECKPOINT_DIR, SAMPLE_DIR, METRICS_DIR):
    os.makedirs(directory, exist_ok=True)


# ============================================================
# Utility functions
# ============================================================

def save_samples(original, masked, reconstructed, epoch, count=8):
    """Save Original | Masked | Reconstructed comparison panels."""
    count = min(count, original.size(0))

    for i in range(count):
        images = [original[i], masked[i], reconstructed[i]]
        pil_images = []

        for tensor in images:
            array = (
                tensor.detach()
                .cpu()
                .clamp(0, 1)
                .permute(1, 2, 0)
                .numpy() * 255
            ).astype(np.uint8)

            pil_images.append(Image.fromarray(array))

        width, height = pil_images[0].size
        canvas = Image.new("RGB", (width * 3, height))

        for j, image in enumerate(pil_images):
            canvas.paste(image, (j * width, 0))

        canvas.save(
            os.path.join(
                SAMPLE_DIR,
                f"epoch_{epoch:03d}_sample_{i}.png",
            )
        )


def save_checkpoint(epoch, encoders, predictor, decoder, optimizer,
                    metrics, path):
    checkpoint = {
        "epoch": epoch,
        "context_encoder": encoders.context_encoder.state_dict(),
        "target_encoder": encoders.target_encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(checkpoint, path)


def calculate_pixel_metrics(reconstructed, original, pixel_mask):
    """
    Calculate MAE and PSNR only inside masked pixels.
    Images are assumed to be normalized to [0, 1].
    """
    mask = pixel_mask.expand_as(original)
    absolute_error = (reconstructed - original).abs() * mask
    squared_error = ((reconstructed - original) ** 2) * mask

    pixel_count = mask.sum().item()
    if pixel_count == 0:
        return 0.0, 100.0, 0.0, 0.0

    mae_sum = absolute_error.sum().item()
    mse_sum = squared_error.sum().item()

    mae = mae_sum / pixel_count
    mse = mse_sum / pixel_count
    psnr = 10.0 * math.log10(1.0 / max(mse, 1e-10))

    return mae, psnr, mae_sum, pixel_count


def save_metrics(history):
    """Write CSV history and refresh plots after every epoch."""
    if not history:
        return

    fieldnames = list(history[0].keys())

    with open(CSV_PATH, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    epochs = [row["epoch"] for row in history]

    plots = [
        ("train_total", "val_total", "Total loss", "total_loss.png"),
        ("train_jepa", "val_jepa", "JEPA loss", "jepa_loss.png"),
        (
            "train_reconstruction",
            "val_reconstruction",
            "Reconstruction loss",
            "reconstruction_loss.png",
        ),
        ("train_mae", "val_mae", "Masked-region MAE", "masked_mae.png"),
        ("train_psnr", "val_psnr", "Masked-region PSNR (dB)", "masked_psnr.png"),
    ]

    for train_key, val_key, title, filename in plots:
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, [row[train_key] for row in history], label="Training")
        plt.plot(epochs, [row[val_key] for row in history], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(METRICS_DIR, filename), dpi=150)
        plt.close()


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(encoders, predictor, decoder, val_loader, device):
    encoders.eval()
    predictor.eval()
    decoder.eval()

    total_jepa = 0.0
    total_reconstruction = 0.0
    total_loss = 0.0
    batch_count = 0

    mae_sum = 0.0
    pixel_count = 0.0
    squared_error_sum = 0.0

    for original, masked, pixel_mask in val_loader:
        original = original.to(device, non_blocking=True)
        masked = masked.to(device, non_blocking=True)
        pixel_mask = pixel_mask.to(device, non_blocking=True)

        target_mask = torch.nn.functional.adaptive_max_pool2d(
            pixel_mask, (16, 16)
        )

        context_features, skip32, target_features = encoders(
            masked, original
        )

        predicted_tokens = predictor(context_features, target_mask)

        combined_features = combine_features(
            context_features, predicted_tokens, target_mask
        )

        reconstructed = decoder(
            combined_features, skip32, pixel_mask
        )

        loss_jepa = jepa_loss(
            predicted_tokens, target_features, target_mask
        )
        loss_reconstruction = reconstruction_loss(
            reconstructed, original, pixel_mask
        )
        loss = combined_loss(
            loss_jepa,
            loss_reconstruction,
            LAMBDA_RECONSTRUCTION,
        )

        total_jepa += loss_jepa.item()
        total_reconstruction += loss_reconstruction.item()
        total_loss += loss.item()
        batch_count += 1

        mask = pixel_mask.expand_as(original)
        mae_sum += ((reconstructed - original).abs() * mask).sum().item()
        squared_error_sum += (
            ((reconstructed - original) ** 2) * mask
        ).sum().item()
        pixel_count += mask.sum().item()

    if batch_count == 0:
        raise RuntimeError("Validation loader is empty.")

    mae = mae_sum / max(pixel_count, 1.0)
    mse = squared_error_sum / max(pixel_count, 1.0)
    psnr = 10.0 * math.log10(1.0 / max(mse, 1e-10))

    return {
        "val_jepa": total_jepa / batch_count,
        "val_reconstruction": total_reconstruction / batch_count,
        "val_total": total_loss / batch_count,
        "val_mae": mae,
        "val_psnr": psnr,
    }


# ============================================================
# Training
# ============================================================

def train():
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Training dataset generates fresh random samples when accessed.
    train_dataset = ShapesDataset(length=DATASET_SIZE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=True,
    )

    # Generate validation samples once and reuse the same samples
    # for every epoch, making validation comparisons meaningful.
    print("Generating fixed validation set...")
    val_source = ShapesDataset(length=VAL_SIZE)
    val_samples = [val_source[i] for i in range(VAL_SIZE)]

    val_original = torch.stack([sample["original"] for sample in val_samples])
    val_masked = torch.stack([sample["masked"] for sample in val_samples])
    val_masks = torch.stack([sample["mask"] for sample in val_samples])

    val_dataset = TensorDataset(val_original, val_masked, val_masks)

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    encoders = JEPAEncoders(ema_decay=EMA_DECAY).to(DEVICE)
    predictor = JEPAPredictor().to(DEVICE)
    decoder = CNNDecoder().to(DEVICE)

    trainable_parameters = (
        list(encoders.context_encoder.parameters())
        + list(predictor.parameters())
        + list(decoder.parameters())
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    history = []
    best_val_reconstruction = float("inf")

    print("\nStarting training...")
    print(f"Epochs: {EPOCHS}")
    print(f"Training samples: {DATASET_SIZE}")
    print(f"Validation samples: {VAL_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Metrics directory: {METRICS_DIR}\n")

    for epoch in range(1, EPOCHS + 1):
        encoders.train()
        predictor.train()
        decoder.train()

        total_jepa = 0.0
        total_reconstruction = 0.0
        total_loss = 0.0

        train_mae_sum = 0.0
        train_squared_error_sum = 0.0
        train_pixel_count = 0.0
        batch_count = 0

        last_original = None
        last_masked = None
        last_reconstructed = None

        for step, batch in enumerate(train_loader, start=1):
            masked = batch["masked"].to(DEVICE, non_blocking=True)
            original = batch["original"].to(DEVICE, non_blocking=True)
            pixel_mask = batch["mask"].to(DEVICE, non_blocking=True)

            target_mask = torch.nn.functional.adaptive_max_pool2d(
                pixel_mask, (16, 16)
            )

            optimizer.zero_grad(set_to_none=True)

            context_features, skip32, target_features = encoders(
                masked, original
            )

            predicted_tokens = predictor(context_features, target_mask)

            combined_features = combine_features(
                context_features, predicted_tokens, target_mask
            )

            reconstructed = decoder(
                combined_features, skip32, pixel_mask
            )

            loss_jepa = jepa_loss(
                predicted_tokens, target_features, target_mask
            )
            loss_reconstruction = reconstruction_loss(
                reconstructed, original, pixel_mask
            )
            loss = combined_loss(
                loss_jepa,
                loss_reconstruction,
                LAMBDA_RECONSTRUCTION,
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                trainable_parameters, max_norm=1.0
            )

            optimizer.step()
            encoders.update_target_encoder()

            total_jepa += loss_jepa.item()
            total_reconstruction += loss_reconstruction.item()
            total_loss += loss.item()
            batch_count += 1

            mask = pixel_mask.expand_as(original)
            train_mae_sum += (
                (reconstructed.detach() - original).abs() * mask
            ).sum().item()
            train_squared_error_sum += (
                ((reconstructed.detach() - original) ** 2) * mask
            ).sum().item()
            train_pixel_count += mask.sum().item()

            last_original = original.detach()
            last_masked = masked.detach()
            last_reconstructed = reconstructed.detach()

            if step % 50 == 0:
                print(
                    f"Epoch [{epoch}/{EPOCHS}] "
                    f"Step [{step}/{len(train_loader)}] "
                    f"JEPA: {loss_jepa.item():.4f} "
                    f"Recon: {loss_reconstruction.item():.4f} "
                    f"Total: {loss.item():.4f}"
                )

        if batch_count == 0:
            raise RuntimeError(
                "Training loader produced no batches. "
                "Check DATASET_SIZE and BATCH_SIZE."
            )

        train_jepa = total_jepa / batch_count
        train_reconstruction = total_reconstruction / batch_count
        train_total = total_loss / batch_count

        train_mae = train_mae_sum / max(train_pixel_count, 1.0)
        train_mse = train_squared_error_sum / max(train_pixel_count, 1.0)
        train_psnr = 10.0 * math.log10(1.0 / max(train_mse, 1e-10))

        val_metrics = validate(
            encoders, predictor, decoder, val_loader, DEVICE
        )

        record = {
            "epoch": epoch,
            "train_jepa": train_jepa,
            "train_reconstruction": train_reconstruction,
            "train_total": train_total,
            "train_mae": train_mae,
            "train_psnr": train_psnr,
            **val_metrics,
        }
        history.append(record)

        # Persist measurements and refresh graphs every epoch.
        save_metrics(history)

        print("\n" + "=" * 65)
        print(f"Epoch {epoch}/{EPOCHS}")
        print(
            f"Train | JEPA: {train_jepa:.5f} | "
            f"Recon: {train_reconstruction:.5f} | "
            f"Total: {train_total:.5f}"
        )
        print(
            f"Val   | JEPA: {val_metrics['val_jepa']:.5f} | "
            f"Recon: {val_metrics['val_reconstruction']:.5f} | "
            f"Total: {val_metrics['val_total']:.5f}"
        )
        print(
            f"Masked pixels | Train MAE: {train_mae:.5f} | "
            f"Val MAE: {val_metrics['val_mae']:.5f}"
        )
        print(
            f"Masked PSNR | Train: {train_psnr:.2f} dB | "
            f"Val: {val_metrics['val_psnr']:.2f} dB"
        )
        print("=" * 65)

        if epoch == 1 or epoch % 5 == 0:
            save_samples(
                last_original,
                last_masked,
                last_reconstructed,
                epoch,
            )

        save_checkpoint(
            epoch,
            encoders,
            predictor,
            decoder,
            optimizer,
            record,
            LATEST_CHECKPOINT,
        )

        if val_metrics["val_reconstruction"] < best_val_reconstruction:
            best_val_reconstruction = val_metrics["val_reconstruction"]

            save_checkpoint(
                epoch,
                encoders,
                predictor,
                decoder,
                optimizer,
                record,
                BEST_CHECKPOINT,
            )
            print(
                f"New best model saved "
                f"(validation reconstruction: {best_val_reconstruction:.6f})"
            )

    print("\nTraining completed!")
    print(f"CSV metrics: {CSV_PATH}")
    print(f"Plots: {METRICS_DIR}")
    print(f"Best checkpoint: {BEST_CHECKPOINT}")


if __name__ == "__main__":
    train()
