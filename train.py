import csv
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from dataset import ShapesDataset
from models import JEPAEncoders, JEPAPredictor, CNNDecoder, combine_features
from losses import compute_losses
from postprocess import refine_colour_regions


# ---------------- Configuration ----------------
BATCH_SIZE = 16
EPOCHS = 100
DATASET_SIZE = 8000
VAL_SIZE = 400

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EMA_DECAY = 0.996

LAMBDA_JEPA = 1.0
LAMBDA_RECONSTRUCTION = 1.0

NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# A new run name prevents overwriting checkpoints from the previous experiment.
RUN_NAME = "jepa_16x16_joint_grad_nearest_boundary"
CHECKPOINT_DIR = f"outputs/checkpoints_{RUN_NAME}"
SAMPLE_DIR = f"outputs/samples_{RUN_NAME}"
METRICS_DIR = f"outputs/metrics_{RUN_NAME}"
LATEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "latest.pth")
BEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_model.pth")
CSV_PATH = os.path.join(METRICS_DIR, "training_history.csv")

for folder in (CHECKPOINT_DIR, SAMPLE_DIR, METRICS_DIR):
    os.makedirs(folder, exist_ok=True)


def move_batch_to_device(batch, device):
    return {
        key: batch[key].to(device, non_blocking=True)
        for key in ("original", "masked", "mask")
    }


def forward_representation(encoders, predictor, original, masked, pixel_mask):
    """
    JEPA branch. Returns features with gradients for the context encoder
    and predictor. Target encoder is internally run under no_grad().
    """
    latent_mask = F.adaptive_max_pool2d(pixel_mask, (16, 16))
    context_features, target_features = encoders(masked, original)
    predicted_tokens = predictor(context_features, latent_mask)

    combined_features = combine_features(
        context_features, predicted_tokens, latent_mask
    )
    return predicted_tokens, target_features, latent_mask, combined_features


def tensor_to_pil(tensor):
    array = (
        tensor.detach().float().cpu().clamp(0, 1)
        .permute(1, 2, 0).numpy() * 255
    ).astype(np.uint8)
    return Image.fromarray(array)


def completed_image(masked, reconstructed, pixel_mask):
    # Preserve observed pixels exactly in the displayed completion.
    return masked * (1.0 - pixel_mask) + reconstructed * pixel_mask


def save_samples(original, masked, reconstructed, pixel_mask, epoch, count=8):
    """Save four panels: original, masked, raw completion, refined completion."""
    count = min(count, original.size(0))
    for index in range(count):
        mask = pixel_mask[index:index + 1]
        raw_completed = completed_image(
            masked[index:index + 1], reconstructed[index:index + 1], mask
        )[0]

        # Visible, unmasked pixels act as exact colour seeds. The postprocessor
        # expands those colours through predicted foreground regions while
        # preserving the observed pixels.
        known_pixels = 1.0 - mask[0]
        refined_completed = refine_colour_regions(
            raw_completed,
            known_mask=known_pixels,
            background_threshold=0.16,
        )

        panels = [
            tensor_to_pil(original[index]),
            tensor_to_pil(masked[index]),
            tensor_to_pil(raw_completed),
            tensor_to_pil(refined_completed),
        ]
        canvas = Image.new("RGB", (64 * 4, 64))
        for panel_index, panel in enumerate(panels):
            canvas.paste(panel, (panel_index * 64, 0))
        canvas.save(
            os.path.join(SAMPLE_DIR, f"epoch_{epoch:03d}_sample_{index}.png")
        )


def image_metrics(predicted, target, mask=None):
    """
    Returns MAE and PSNR. If mask is None, evaluates the full image;
    otherwise evaluates only masked pixels.
    """
    if mask is None:
        expanded = torch.ones_like(target)
    else:
        expanded = mask.expand_as(target)

    count = expanded.sum().item()
    error = predicted - target
    mae = (error.abs() * expanded).sum().item() / max(count, 1.0)
    mse = (error.square() * expanded).sum().item() / max(count, 1.0)
    psnr = 10.0 * math.log10(1.0 / max(mse, 1e-10))
    return mae, psnr


def save_history(history):
    if not history:
        return

    with open(CSV_PATH, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    plots = [
        ("train_total", "val_total", "Total loss", "total_loss.png"),
        ("train_jepa", "val_jepa", "JEPA loss", "jepa_loss.png"),
        (
            "train_reconstruction", "val_reconstruction",
            "Full-image reconstruction L1", "reconstruction_loss.png"
        ),
        ("train_full_mae", "val_full_mae", "Full-image MAE", "full_mae.png"),
        ("train_masked_mae", "val_masked_mae", "Masked-region MAE", "masked_mae.png"),
    ]

    for train_key, val_key, title, filename in plots:
        fig = plt.figure(figsize=(8, 5))
        plt.plot(epochs, [row[train_key] for row in history], label="Training")
        plt.plot(epochs, [row[val_key] for row in history], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(METRICS_DIR, filename), dpi=140)
        plt.close(fig)


def save_checkpoint(epoch, encoders, predictor, decoder, optimizer, metrics, path):
    torch.save({
        "epoch": epoch,
        "context_encoder": encoders.context_encoder.state_dict(),
        "target_encoder": encoders.target_encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
        "loss_weights": {
            "jepa": LAMBDA_JEPA,
            "reconstruction": LAMBDA_RECONSTRUCTION,
        },
        "gradient_routing": {
            "jepa_updates": ["context_encoder", "predictor"],
            "reconstruction_updates": [
                "context_encoder", "predictor", "decoder"
            ],
            "target_encoder_update": "EMA only; no gradients",
        },
        "decoder_description": "nearest-colour expansion with learned boundary-gated refinement",
        "run_name": RUN_NAME,
    }, path)


@torch.no_grad()
def validate(encoders, predictor, decoder, loader):
    encoders.eval()
    predictor.eval()
    decoder.eval()

    jepa_sum = 0.0
    reconstruction_sum = 0.0
    total_sum = 0.0

    full_abs_sum = full_sq_sum = full_count = 0.0
    masked_abs_sum = masked_sq_sum = masked_count = 0.0
    batches = 0

    for original, masked, pixel_mask in loader:
        original = original.to(DEVICE, non_blocking=True)
        masked = masked.to(DEVICE, non_blocking=True)
        pixel_mask = pixel_mask.to(DEVICE, non_blocking=True)

        predicted_tokens, target_features, latent_mask, combined = (
            forward_representation(
                encoders, predictor, original, masked, pixel_mask
            )
        )

        # No gradients are enabled during validation. Decoder still receives
        # only combined features and the latent mask, never the original image.
        reconstructed = decoder(combined, latent_mask)

        losses = compute_losses(
            predicted_tokens,
            target_features,
            latent_mask,
            reconstructed,
            original,
            LAMBDA_JEPA,
            LAMBDA_RECONSTRUCTION,
        )
        jepa_sum += losses["jepa"].item()
        reconstruction_sum += losses["reconstruction"].item()
        total_sum += losses["total"].item()

        full_abs_sum += (reconstructed - original).abs().sum().item()
        full_sq_sum += (reconstructed - original).square().sum().item()
        full_count += original.numel()

        expanded_mask = pixel_mask.expand_as(original)
        masked_abs_sum += (
            (reconstructed - original).abs() * expanded_mask
        ).sum().item()
        masked_sq_sum += (
            (reconstructed - original).square() * expanded_mask
        ).sum().item()
        masked_count += expanded_mask.sum().item()
        batches += 1

    if batches == 0:
        raise RuntimeError("Validation loader is empty.")

    full_mae = full_abs_sum / max(full_count, 1.0)
    full_mse = full_sq_sum / max(full_count, 1.0)
    full_psnr = 10.0 * math.log10(1.0 / max(full_mse, 1e-10))

    masked_mae = masked_abs_sum / max(masked_count, 1.0)
    masked_mse = masked_sq_sum / max(masked_count, 1.0)
    masked_psnr = 10.0 * math.log10(1.0 / max(masked_mse, 1e-10))

    return {
        "val_jepa": jepa_sum / batches,
        "val_reconstruction": reconstruction_sum / batches,
        "val_total": total_sum / batches,
        "val_full_mae": full_mae,
        "val_full_psnr": full_psnr,
        "val_masked_mae": masked_mae,
        "val_masked_psnr": masked_psnr,
    }


def train():
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_dataset = ShapesDataset(length=DATASET_SIZE)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        drop_last=True,
    )

    print("Preparing fixed validation set...")
    validation_source = ShapesDataset(length=VAL_SIZE)
    validation_samples = [validation_source[i] for i in range(VAL_SIZE)]
    validation_dataset = TensorDataset(
        torch.stack([sample["original"] for sample in validation_samples]),
        torch.stack([sample["masked"] for sample in validation_samples]),
        torch.stack([sample["mask"] for sample in validation_samples]),
    )
    del validation_samples, validation_source

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    encoders = JEPAEncoders(ema_decay=EMA_DECAY).to(DEVICE)
    predictor = JEPAPredictor().to(DEVICE)
    decoder = CNNDecoder().to(DEVICE)

    # A shared optimizer is appropriate: both objectives update the online
    # representation, while reconstruction also updates the decoder.
    # The frozen target encoder is excluded and updated only through EMA.
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

    print(
        f"Starting {RUN_NAME}: epochs={EPOCHS}, train={DATASET_SIZE}, "
        f"val={VAL_SIZE}, batch={BATCH_SIZE}"
    )
    print(
        f"Loss weights: JEPA={LAMBDA_JEPA}, "
        f"full-image L1={LAMBDA_RECONSTRUCTION}"
    )
    print("Gradient routing: JEPA -> context encoder + predictor")
    print("Gradient routing: full-image L1 -> context encoder + predictor + decoder")
    print("Decoder: nearest-colour expansion + learned boundary-gated refinement")

    for epoch in range(1, EPOCHS + 1):
        encoders.train()
        predictor.train()
        decoder.train()

        totals = {"jepa": 0.0, "reconstruction": 0.0, "total": 0.0}

        full_abs_sum = full_sq_sum = full_count = 0.0
        masked_abs_sum = masked_sq_sum = masked_count = 0.0
        batches = 0

        last_original = last_masked = last_reconstructed = last_mask = None

        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, DEVICE)
            original = batch["original"]
            masked = batch["masked"]
            pixel_mask = batch["mask"]

            optimizer.zero_grad(set_to_none=True)

            predicted_tokens, target_features, latent_mask, combined = (
                forward_representation(
                    encoders, predictor, original, masked, pixel_mask
                )
            )

            # Keep the feature map attached: reconstruction gradients flow
            # through decoder -> combined features -> predictor/context encoder.
            reconstructed = decoder(combined, latent_mask)

            losses = compute_losses(
                predicted_tokens,
                target_features,
                latent_mask,
                reconstructed,
                original,
                LAMBDA_JEPA,
                LAMBDA_RECONSTRUCTION,
            )

            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
            optimizer.step()

            # EMA target encoder follows the updated context encoder.
            encoders.update_target_encoder()

            for key in totals:
                totals[key] += losses[key].detach().item()

            with torch.no_grad():
                full_abs_sum += (
                    (reconstructed - original).abs().sum().item()
                )
                full_sq_sum += (
                    (reconstructed - original).square().sum().item()
                )
                full_count += original.numel()

                expanded_mask = pixel_mask.expand_as(original)
                masked_abs_sum += (
                    (reconstructed - original).abs() * expanded_mask
                ).sum().item()
                masked_sq_sum += (
                    (reconstructed - original).square() * expanded_mask
                ).sum().item()
                masked_count += expanded_mask.sum().item()

            batches += 1

            last_original = original.detach()
            last_masked = masked.detach()
            last_reconstructed = reconstructed.detach()
            last_mask = pixel_mask.detach()

            if step % 50 == 0:
                print(
                    f"Epoch {epoch}/{EPOCHS} step {step}/{len(train_loader)} "
                    f"total={losses['total'].item():.4f} "
                    f"jepa={losses['jepa'].item():.4f} "
                    f"full_l1={losses['reconstruction'].item():.4f}"
                )

        if batches == 0:
            raise RuntimeError("Training loader is empty.")

        train_full_mae = full_abs_sum / max(full_count, 1.0)
        train_full_mse = full_sq_sum / max(full_count, 1.0)
        train_full_psnr = 10.0 * math.log10(
            1.0 / max(train_full_mse, 1e-10)
        )
        train_masked_mae = masked_abs_sum / max(masked_count, 1.0)
        train_masked_mse = masked_sq_sum / max(masked_count, 1.0)
        train_masked_psnr = 10.0 * math.log10(
            1.0 / max(train_masked_mse, 1e-10)
        )

        train_metrics = {key: value / batches for key, value in totals.items()}
        validation = validate(
            encoders, predictor, decoder, validation_loader
        )

        record = {
            "epoch": epoch,
            "train_jepa": train_metrics["jepa"],
            "train_reconstruction": train_metrics["reconstruction"],
            "train_total": train_metrics["total"],
            "train_full_mae": train_full_mae,
            "train_full_psnr": train_full_psnr,
            "train_masked_mae": train_masked_mae,
            "train_masked_psnr": train_masked_psnr,
            **validation,
        }
        history.append(record)
        save_history(history)

        print("\n" + "-" * 72)
        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train total={train_metrics['total']:.5f}, "
            f"JEPA={train_metrics['jepa']:.5f}, "
            f"full L1={train_metrics['reconstruction']:.5f}"
        )
        print(
            f"Validation total={validation['val_total']:.5f}, "
            f"JEPA={validation['val_jepa']:.5f}, "
            f"full L1={validation['val_reconstruction']:.5f}"
        )
        print(
            f"Full-image MAE: train={train_full_mae:.5f}, "
            f"val={validation['val_full_mae']:.5f}"
        )
        print(
            f"Masked-region MAE: train={train_masked_mae:.5f}, "
            f"val={validation['val_masked_mae']:.5f}"
        )

        if epoch == 1 or epoch % 5 == 0:
            save_samples(
                last_original, last_masked, last_reconstructed,
                last_mask, epoch
            )

        save_checkpoint(
            epoch, encoders, predictor, decoder, optimizer,
            record, LATEST_CHECKPOINT
        )

        if validation["val_reconstruction"] < best_val_reconstruction:
            best_val_reconstruction = validation["val_reconstruction"]
            save_checkpoint(
                epoch, encoders, predictor, decoder, optimizer,
                record, BEST_CHECKPOINT
            )
            print("Saved new best checkpoint.")

    print("\nTraining finished.")
    print(f"Samples: {SAMPLE_DIR}")
    print(f"Metrics: {METRICS_DIR}")
    print(f"Checkpoints: {CHECKPOINT_DIR}")


if __name__ == "__main__":
    train()
