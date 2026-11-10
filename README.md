# Mini JEPA Shapes Prediction

A compact research-style image completion project that trains a self-supervised JEPA-style model to predict missing regions of simple geometric shapes. The model learns latent representations from partially masked images and reconstructs the hidden parts using a predictor + decoder pipeline.

This repository combines:

- a synthetic shape dataset generator
- a JEPA-inspired encoder/predictor architecture
- a reconstruction decoder for image completion
- a FastAPI inference server for live predictions
- a lightweight frontend for interactive masking and filling

It is designed as a small, readable, and hackable playground for latent representation learning and masked image modeling.

## Why this project?

The goal is to explore a minimal version of the JEPA idea: instead of predicting the full image directly, the model learns a latent representation of visible content and predicts what should exist in masked areas.

The project focuses on a clean, training-friendly pipeline:

- encode the visible image
- predict latent tokens for missing regions
- combine latent context with predicted values
- decode back into a completed image
- refine the result with a colour-region postprocessing pass

The result is a model that can plausibly inpaint simple geometry while preserving the parts of the image that are already known.

## Project highlights

- JEPA-style masked prediction for synthetic shapes
- Reconstruction objective with latent prediction
- Decoder trained with a boundary-aware refinement strategy
- FastAPI backend for inference requests
- HTML/JS demo UI for running predictions interactively
- Clear checkpoint handling so invalid decoder checkpoints fail loudly instead of silently producing fake outputs

## Repository structure

```text
.
├── dataset.py              # Synthetic shape dataset generation
├── models.py               # JEPA encoder, predictor, decoder, feature fusion
├── train.py                # Training loop, checkpointing, metrics
├── losses.py               # JEPA and reconstruction losses
├── masking.py              # Mask generation utilities
├── postprocess.py          # Colour-region refinement / completion cleanup
├── server.py               # FastAPI inference API
├── evaluate.py             # Evaluation entry point
├── requirements.txt        # Python dependencies
├── static/                 # Frontend assets (HTML/CSS/JS)
├── outputs/                # Training outputs, checkpoints, plots, samples
├── test/                   # Test directory
└── README.md               # Project documentation
```

## Architecture at a glance

The training pipeline follows a JEPA-inspired structure:

1. A masked image is passed into the context encoder.
2. A predictor estimates latent tokens for the missing region.
3. The model combines context features with predicted tokens.
4. The decoder reconstructs the full image.
5. The postprocessor refines the result while preserving visible pixels exactly.

This is intentionally compact and auditable, making it a useful learning project for understanding latent reconstruction models.

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

Start training with:

```bash
python train.py
```

Training writes output to the `outputs/` directory, including:

- checkpoints
- sample visualizations
- metric charts
- CSV history

## Running the inference server

```bash
python server.py
```

The project includes a browser-based UI served by FastAPI. You can use it to upload an image, paint or mask a region, and request a prediction.

### API status

```bash
curl http://localhost:8000/api/status
```

### Prediction endpoint

The server accepts JSON payloads with the original image and a mask, then returns processed outputs including the raw completion and refined completion.

## Checkpoint handling

The project is careful about model validation. If a checkpoint is missing a trained decoder, the server refuses to run inference rather than loading a random decoder. This avoids a common failure mode where a broken or incomplete checkpoint looks valid but produces misleading results.

If you are training new checkpoints, make sure the saved dictionary contains the decoder state correctly.

## Example workflow

```bash
# 1. Train the model
python train.py

# 2. Run the server
python server.py

# 3. Open the local web UI and test inpainting on masked shapes
```

## Notes

- The dataset is synthetic and intentionally simple, which makes the model easy to inspect and debug.
- The project is well suited for experimentation with loss weighting, latent mask strategies, and decoder architecture changes.
- It is a compact but realistic demonstration of masked prediction in low-complexity image domains.

## License

This project does not currently include an explicit license file. If you plan to reuse or distribute it, check the repository settings or add a license before publishing externally.

## Acknowledgements

This project follows the spirit of JEPA-inspired masked modeling and latent reconstruction, adapted to a minimal self-contained shape completion setting.

---

Built as a compact visual learning project for latent prediction and image inpainting.
