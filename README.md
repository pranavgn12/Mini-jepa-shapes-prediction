# JEPA + nearest-colour reconstruction experiment

This experiment changes two things from the previous separated-gradient run:

1. **Reconstruction gradients now flow through the representation.** The full-image L1 loss updates the context encoder, JEPA predictor, and decoder. JEPA loss updates the context encoder and predictor. The target encoder remains frozen and is updated only by EMA.
2. **The decoder uses nearest-colour expansion.** It predicts a 16x16 coarse RGB grid, expands it to 64x64 using nearest-neighbour interpolation, and applies a learned boundary-gated colour correction. There is no explicit edge, corner, Laplacian, or palette loss.

## Files

- `models.py` — context/target encoders, transformer predictor, and nearest-expansion decoder
- `losses.py` — masked JEPA MSE and full-image L1 reconstruction loss
- `dataset.py` — synthetic geometric-shape dataset
- `masking.py` — rectangular magenta masks and their binary pixel masks
- `train.py` — training and validation loop, metrics, samples, and checkpoints
- `test_gradients.py` — CPU smoke test for shapes and gradient flow

## Run

Install the dependencies in the active Python environment:

```bash
python -m pip install torch numpy pillow matplotlib
```

Run the smoke test first:

```bash
python test_gradients.py
```

Then start training:

```bash
python train.py
```

The script automatically uses CUDA when PyTorch detects it. Outputs are written beneath `outputs/`.

## Gradient routing

| Loss | Context encoder | Predictor | Decoder | Target encoder |
|---|---:|---:|---:|---:|
| JEPA masked-feature MSE | Yes | Yes | No | No gradients |
| Full-image reconstruction L1 | Yes | Yes | Yes | No gradients |
| EMA update | No optimizer gradient | No | No | Updated from context encoder |

The decoder receives only combined 16x16 features and the latent mask. It does not receive the original image or any high-resolution encoder skip connection.
