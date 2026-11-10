"""FastAPI inference server for the JEPA Live Studio.

This server deliberately refuses to run inference when a checkpoint is missing
its trained decoder weights. It never substitutes a randomly initialized
decoder, because that would look like a neural prediction while being invalid.

Expected model contract (from the project's 16x16 models.py):
  context_encoder(image) -> (B, 64, 16, 16)
  predictor(context_features, latent_mask) -> (B, 256, 64)
  combine_features(context_features, predicted_tokens, latent_mask)
  decoder(combined_features, latent_mask) -> (B, 3, 64, 64)

Optional environment variables:
  JEPA_CHECKPOINT          Full training checkpoint (.pth)
  JEPA_DECODER_CHECKPOINT  A separate decoder state_dict/checkpoint, if available
  JEPA_DEVICE              cpu, cuda, or another valid torch device string
"""

from __future__ import annotations

import base64
import binascii
import io
import os
from collections.abc import Mapping
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from models import JEPAEncoders, JEPAPredictor, CNNDecoder, combine_features
from postprocess import refine_colour_regions

ROOT = Path(__file__).resolve().parent
DEVICE = torch.device(
    os.getenv("JEPA_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
)
MAGENTA = np.array([255, 0, 255], dtype=np.uint8)
IMAGE_SIZE = 64
LATENT_SIZE = 16
MAX_IMAGE_BYTES = 2_000_000

app = FastAPI(title="JEPA Live Studio")
_model_lock = Lock()
_cached_checkpoint: str | None = None
_cached_models: tuple[JEPAEncoders, JEPAPredictor, CNNDecoder] | None = None


class CheckpointProblem(RuntimeError):
    """A checkpoint exists but cannot be used for this model architecture."""


class MissingDecoderWeights(CheckpointProblem):
    """The checkpoint does not contain trained decoder parameters."""


def checkpoint_path() -> Path:
    configured = os.getenv("JEPA_CHECKPOINT")
    if configured:
        return Path(configured).expanduser().resolve()

    # Prefer the repaired run, which is written to a separate folder so the
    # legacy checkpoints remain untouched. Fall back to the original path only
    # when no recovered checkpoint has been produced yet.
    candidates = [
        Path.cwd()
        / "outputs/checkpoints_jepa_16x16_joint_grad_nearest_boundary_decoder_recovery/best_model.pth",
        ROOT
        / "outputs/checkpoints_jepa_16x16_joint_grad_nearest_boundary_decoder_recovery/best_model.pth",
        ROOT.parent
        / "outputs/checkpoints_jepa_16x16_joint_grad_nearest_boundary_decoder_recovery/best_model.pth",
        Path.cwd()
        / "outputs/checkpoints_jepa_16x16_joint_grad_nearest_boundary/best_model.pth",
        ROOT
        / "outputs/checkpoints_jepa_16x16_joint_grad_nearest_boundary/best_model.pth",
        ROOT.parent
        / "outputs/checkpoints_jepa_16x16_joint_grad_nearest_boundary/best_model.pth",
        Path.cwd() / "outputs/checkpoints/best_model.pth",
        ROOT / "outputs/checkpoints/best_model.pth",
        ROOT.parent / "outputs/checkpoints/best_model.pth",
    ]
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])


def _load_torch_file(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Compatibility with older PyTorch versions that do not accept
        # weights_only. Checkpoints should still only be loaded from trusted files.
        return torch.load(path, map_location="cpu")
    except Exception as exc:
        raise CheckpointProblem(f"Could not read checkpoint '{path}': {exc}") from exc


def _extract_decoder_state(value: Any) -> Mapping[str, torch.Tensor] | None:
    """Accept a direct state_dict or a small wrapper containing one."""
    if isinstance(value, Mapping):
        if value and all(isinstance(key, str) for key in value):
            # A normal PyTorch state_dict maps parameter names to tensors.
            if all(torch.is_tensor(item) for item in value.values()):
                return value
        for key in ("decoder", "decoder_state_dict", "state_dict"):
            nested = value.get(key)
            if isinstance(nested, Mapping) and nested and all(
                torch.is_tensor(item) for item in nested.values()
            ):
                return nested
    return None


def _decoder_state_from_checkpoint(checkpoint: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    decoder_state = _extract_decoder_state(checkpoint.get("decoder"))
    if decoder_state is not None:
        return decoder_state

    # This optional route is useful if a decoder was saved separately after
    # retraining. The decoder must have the exact same architecture as models.py.
    separate_path_value = os.getenv("JEPA_DECODER_CHECKPOINT")
    if separate_path_value:
        separate_path = Path(separate_path_value).expanduser().resolve()
        if not separate_path.is_file():
            raise CheckpointProblem(
                f"JEPA_DECODER_CHECKPOINT was set, but the file does not exist: {separate_path}"
            )
        separate_checkpoint = _load_torch_file(separate_path)
        separate_state = _extract_decoder_state(separate_checkpoint)
        if separate_state is not None:
            return separate_state
        raise CheckpointProblem(
            f"'{separate_path}' does not contain a decoder state_dict."
        )

    found = checkpoint.get("decoder")
    if isinstance(found, str):
        raise MissingDecoderWeights(
            "This checkpoint has no trained decoder weights. Its 'decoder' entry is "
            f"only metadata: {found!r}. The training checkpoint dictionary writes "
            "'decoder': decoder.state_dict() and later writes another 'decoder' key "
            "for the description. Python keeps the later value and silently discards "
            "the state_dict. Both best_model.pth and latest.pth have this problem. "
            "Fix the metadata key in train.py to 'decoder_description', then train "
            "and save a new checkpoint. The missing decoder parameters cannot be "
            "recovered from these files."
        )

    raise MissingDecoderWeights(
        "The checkpoint does not contain a decoder state_dict. Fix the checkpoint "
        "saving code and train/save the decoder before running live inference."
    )


def load_models(force_reload: bool = False):
    global _cached_checkpoint, _cached_models

    path = checkpoint_path()
    if not path.is_file():
        raise CheckpointProblem(
            f"Checkpoint not found: {path}. Set JEPA_CHECKPOINT to the full best_model.pth path."
        )

    cache_key = str(path.resolve())
    if (
        not force_reload
        and _cached_checkpoint == cache_key
        and _cached_models is not None
    ):
        return _cached_models

    checkpoint = _load_torch_file(path)
    if not isinstance(checkpoint, Mapping):
        raise CheckpointProblem("Checkpoint root must be a dictionary-like object.")

    for key in ("context_encoder", "predictor"):
        if not isinstance(checkpoint.get(key), Mapping):
            actual = type(checkpoint.get(key)).__name__
            raise CheckpointProblem(
                f"Checkpoint field '{key}' must be a state_dict; found {actual}."
            )

    decoder_state = _decoder_state_from_checkpoint(checkpoint)

    encoders = JEPAEncoders().to(DEVICE)
    predictor = JEPAPredictor().to(DEVICE)
    decoder = CNNDecoder().to(DEVICE)

    try:
        encoders.context_encoder.load_state_dict(
            checkpoint["context_encoder"], strict=True
        )
        predictor.load_state_dict(checkpoint["predictor"], strict=True)
        decoder.load_state_dict(decoder_state, strict=True)
    except Exception as exc:
        raise CheckpointProblem(
            "Checkpoint parameters do not match the model architecture in this "
            f"project's models.py: {exc}"
        ) from exc

    encoders.eval()
    predictor.eval()
    decoder.eval()

    _cached_checkpoint = cache_key
    _cached_models = (encoders, predictor, decoder)
    return _cached_models


def _decode_image_data(value: Any, field: str, mode: str) -> np.ndarray:
    if not isinstance(value, str) or not value:
        raise ValueError(f"'{field}' must be a non-empty PNG data URL string.")

    encoded = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"'{field}' is not valid base64 image data.") from exc

    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"'{field}' exceeds the {MAX_IMAGE_BYTES // 1_000_000} MB limit.")

    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            if image.size != (IMAGE_SIZE, IMAGE_SIZE):
                raise ValueError(
                    f"'{field}' must be {IMAGE_SIZE}x{IMAGE_SIZE}; received {image.size}."
                )
            return np.asarray(image.convert(mode), dtype=np.uint8).copy()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"'{field}' does not contain a readable image.") from exc


def _encode_png(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB").save(
        buffer, format="PNG"
    )
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def predict_arrays(
    original: np.ndarray,
    mask: np.ndarray,
    models,
    threshold: float = 0.16,
    margin: float = 0.02,
):
    encoders, predictor, decoder = models
    binary_mask = (mask > 127).astype(np.float32)

    masked = original.copy()
    masked[binary_mask > 0] = MAGENTA

    input_tensor = (
        torch.from_numpy(masked.astype(np.float32) / 255.0)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(DEVICE)
    )
    pixel_mask = (
        torch.from_numpy(binary_mask)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.inference_mode():
        context_features = encoders.context_encoder(input_tensor)
        latent_mask = F.adaptive_max_pool2d(
            pixel_mask, (LATENT_SIZE, LATENT_SIZE)
        )
        predicted_tokens = predictor(context_features, latent_mask)
        combined = combine_features(
            context_features, predicted_tokens, latent_mask
        )
        reconstructed = decoder(combined, latent_mask)

        expected_shape = (1, 3, IMAGE_SIZE, IMAGE_SIZE)
        if tuple(reconstructed.shape) != expected_shape:
            raise RuntimeError(
                f"Decoder returned {tuple(reconstructed.shape)}; expected {expected_shape}."
            )

        raw = (
            reconstructed[0]
            .clamp(0.0, 1.0)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
            * 255.0
        ).round().astype(np.uint8)

    # Preserve visible user pixels exactly and fill only the masked region.
    completed = original.copy()
    completed[binary_mask > 0] = raw[binary_mask > 0]

    refined = refine_colour_regions(
        completed.astype(np.float32) / 255.0,
        known_mask=1.0 - binary_mask,
        background_threshold=threshold,
        background_margin=margin,
    )
    refined = np.clip(np.rint(refined * 255.0), 0, 255).astype(np.uint8)
    refined[binary_mask == 0] = original[binary_mask == 0]

    return raw, refined, masked


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/static/{filename}")
def static_file(filename: str):
    if filename not in {"style.css", "app.js"}:
        raise HTTPException(status_code=404, detail="Static file not found.")
    return FileResponse(ROOT / "static" / filename)


@app.get("/api/status")
def status():
    path = checkpoint_path()
    try:
        with _model_lock:
            load_models()
        return {
            "ready": True,
            "state": "ready",
            "device": str(DEVICE),
            "checkpoint": str(path),
            "message": "Trained encoder, predictor, and decoder loaded.",
        }
    except MissingDecoderWeights as exc:
        return {
            "ready": False,
            "state": "decoder_weights_missing",
            "device": str(DEVICE),
            "checkpoint": str(path),
            "error": str(exc),
            "action": "Fix the duplicate decoder key in train.py and save a new checkpoint.",
        }
    except CheckpointProblem as exc:
        return {
            "ready": False,
            "state": "checkpoint_invalid",
            "device": str(DEVICE),
            "checkpoint": str(path),
            "error": str(exc),
        }
    except Exception as exc:
        return {
            "ready": False,
            "state": "server_error",
            "device": str(DEVICE),
            "checkpoint": str(path),
            "error": f"Unexpected model-loading error: {exc}",
        }


@app.post("/api/predict")
async def predict(request: Request):
    """Parse the body explicitly and return useful errors for client requests."""
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Body must be valid JSON containing original and mask PNG data URLs.",
        ) from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object.")

    original_value = payload.get("original", payload.get("image"))
    mask_value = payload.get("mask", payload.get("pixel_mask"))

    try:
        threshold = float(payload.get("threshold", 0.16))
        margin = float(payload.get("background_margin", payload.get("margin", 0.02)))
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="threshold and background_margin must be numeric values.",
        ) from exc

    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.73:
        raise HTTPException(
            status_code=400, detail="threshold must be between 0 and 1.73."
        )
    if not np.isfinite(margin) or not 0.0 <= margin <= 1.0:
        raise HTTPException(
            status_code=400, detail="background_margin must be between 0 and 1."
        )

    try:
        original = _decode_image_data(original_value, "original", "RGB")
        mask = _decode_image_data(mask_value, "mask", "L")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        with _model_lock:
            models = load_models()
            raw, refined, masked = predict_arrays(
                original, mask, models, threshold, margin
            )
    except MissingDecoderWeights as exc:
        # 409 makes clear that the request is well-formed, but the checkpoint
        # is not capable of satisfying inference until decoder weights exist.
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except CheckpointProblem as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Model inference failed: {exc}"
        ) from exc

    return {
        "raw": _encode_png(raw),
        "refined": _encode_png(refined),
        "masked": _encode_png(masked),
        "device": str(DEVICE),
    }
