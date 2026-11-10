import os
from io import BytesIO

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas

from models import JEPAEncoders, JEPAPredictor, CNNDecoder, combine_features

st.set_page_config(page_title="Mini-JEPA Shape Inpainting", layout="wide")
st.title("Mini-JEPA · Shape Inpainting Studio")
st.caption(
    "Draw vector shapes, rotate and resize them, place up to two rectangular masks, "
    "and inspect the context encoder's spatial responses."
)

CHECKPOINT_PATH = os.environ.get(
    "JEPA_CHECKPOINT", "outputs/checkpoints_16x16/best_model.pth"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 64
CANVAS_SIZE = 512


@st.cache_resource
def load_model(checkpoint_path, device_name):
    device = torch.device(device_name)
    encoders = JEPAEncoders().to(device)
    predictor = JEPAPredictor().to(device)
    decoder = CNNDecoder().to(device)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Train the 16x16 model first or set JEPA_CHECKPOINT."
        )

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    encoders.context_encoder.load_state_dict(checkpoint["context_encoder"])
    predictor.load_state_dict(checkpoint["predictor"])
    decoder.load_state_dict(checkpoint["decoder"])

    encoders.eval()
    predictor.eval()
    decoder.eval()
    return encoders, predictor, decoder, device


def canvas_to_image(canvas_result):
    """
    streamlit-drawable-canvas raises RuntimeError when the canvas has not
    emitted an image yet. Treat that as a blank white canvas.
    """
    if canvas_result is None:
        return Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "white")

    try:
        rgba = canvas_result.image_data
    except (RuntimeError, AttributeError, ValueError):
        rgba = None

    if rgba is None:
        return Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "white")

    array = np.asarray(rgba).astype(np.uint8)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        return Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "white")

    image = Image.fromarray(array).convert("RGB")
    return image.resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS
    )


def apply_rect_masks(image, rectangles):
    original = image.convert("RGB")
    masked = original.copy()
    mask = Image.new("L", original.size, 0)

    draw_mask = ImageDraw.Draw(mask)
    draw_image = ImageDraw.Draw(masked)

    for enabled, x, y, width, height in rectangles:
        if not enabled:
            continue
        x1 = int(np.clip(x, 0, IMAGE_SIZE - 1))
        y1 = int(np.clip(y, 0, IMAGE_SIZE - 1))
        x2 = int(np.clip(x + width, 1, IMAGE_SIZE))
        y2 = int(np.clip(y + height, 1, IMAGE_SIZE))
        if x2 <= x1 or y2 <= y1:
            continue
        draw_mask.rectangle((x1, y1, x2 - 1, y2 - 1), fill=255)
        draw_image.rectangle(
            (x1, y1, x2 - 1, y2 - 1), fill=(255, 0, 255)
        )

    return masked, mask


def tensor_from_pil(image):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def make_heatmap(features):
    # Channel-mean absolute activation: response strength, not class detection.
    activation = features.abs().mean(dim=1, keepdim=True)
    activation = activation - activation.amin(dim=(2, 3), keepdim=True)
    activation = activation / activation.amax(
        dim=(2, 3), keepdim=True
    ).clamp_min(1e-8)
    heat = F.interpolate(
        activation, size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear", align_corners=False
    )[0, 0]
    gray = (heat.detach().cpu().numpy() * 255).astype(np.uint8)
    rgb = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    rgb[..., 0] = gray
    rgb[..., 1] = (gray.astype(np.float32) * 0.65).astype(np.uint8)
    return Image.fromarray(rgb)


def make_grid_image(grid, scale=16):
    array = grid.detach().float().cpu().numpy()
    array = array - array.min()
    array = array / max(float(array.max()), 1e-8)
    array = (array * 255).astype(np.uint8)
    image = Image.fromarray(array, mode="L").resize(
        (16 * scale, 16 * scale), Image.Resampling.NEAREST
    ).convert("RGB")
    draw = ImageDraw.Draw(image)
    for index in range(17):
        position = index * scale
        draw.line(
            (position, 0, position, 16 * scale),
            fill=(90, 90, 90), width=1
        )
        draw.line(
            (0, position, 16 * scale, position),
            fill=(90, 90, 90), width=1
        )
    return image


def run_inference(masked_pil, pixel_mask_pil, models):
    encoders, predictor, decoder, device = models
    masked_tensor = tensor_from_pil(masked_pil).to(device)
    pixel_mask = (
        torch.from_numpy(
            np.asarray(pixel_mask_pil, dtype=np.float32) / 255.0
        ).unsqueeze(0).unsqueeze(0).to(device)
    )

    with torch.inference_mode():
        skip32, context_features = encoders.context_encoder(masked_tensor)
        target_mask = F.adaptive_max_pool2d(pixel_mask, (16, 16))
        predicted_tokens = predictor(context_features, target_mask)
        combined = combine_features(
            context_features, predicted_tokens, target_mask
        )
        reconstructed = decoder(combined, skip32, pixel_mask)

        # Keep visible pixels exactly as drawn; fill only masked locations.
        final = (
            reconstructed * pixel_mask
            + masked_tensor * (1.0 - pixel_mask)
        )

    array = (
        final[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
    ).astype(np.uint8)

    return (
        Image.fromarray(array),
        make_heatmap(context_features),
        make_grid_image(context_features.abs().mean(dim=1)[0]),
        target_mask[0, 0].detach().cpu(),
    )


with st.sidebar:
    st.subheader("Model")
    st.write(f"Device: `{DEVICE}`")
    st.text_input(
        "Checkpoint path",
        value=CHECKPOINT_PATH,
        key="checkpoint_path",
    )
    st.caption("Use a checkpoint trained with the 16×16 architecture.")

    st.divider()
    st.subheader("Vector drawing")
    tool = st.selectbox(
        "Drawing tool",
        ["rect", "circle", "line", "polygon", "freedraw", "transform"],
        help=(
            "Draw with a shape tool, then choose Transform to select, move, "
            "resize, or rotate vector objects."
        ),
    )
    stroke_color = st.color_picker("Outline colour", "#2563EB")
    fill_color = st.color_picker("Fill colour", "#60A5FA")
    stroke_width = st.slider("Outline width", 1, 20, 4)
    st.info(
        "To rotate: select Transform, click a drawn object, then drag its "
        "rotation handle. Use the corner handles to resize."
    )

    st.divider()
    st.subheader("Rectangular masks")
    st.caption(
        "Up to two axis-aligned masks. Adjust their position and dimensions."
    )
    rectangles = []
    for index in (1, 2):
        with st.expander(f"Mask {index}", expanded=(index == 1)):
            enabled = st.checkbox(
                f"Enable mask {index}",
                value=(index == 1),
                key=f"mask{index}_enabled",
            )
            x = st.slider(
                f"X position · mask {index}", 0, 63,
                12 if index == 1 else 36, key=f"mask{index}_x"
            )
            y = st.slider(
                f"Y position · mask {index}", 0, 63,
                12 if index == 1 else 32, key=f"mask{index}_y"
            )
            width = st.slider(
                f"Width · mask {index}", 1, 64, 18,
                key=f"mask{index}_w"
            )
            height = st.slider(
                f"Height · mask {index}", 1, 64, 18,
                key=f"mask{index}_h"
            )
            rectangles.append((enabled, x, y, width, height))

    run_button = st.button(
        "Reconstruct image", type="primary", use_container_width=True
    )

left, right = st.columns([1.1, 1])

with left:
    st.subheader("1 · Draw your shapes")
    canvas_result = st_canvas(
        fill_color=fill_color + "55",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="#FFFFFF",
        height=CANVAS_SIZE,
        width=CANVAS_SIZE,
        drawing_mode=tool,
        key="shape_canvas",
        update_streamlit=True,
        return_image_data=True,
    )
    st.caption(
        "Shapes are Fabric.js canvas objects. Select Transform to move, "
        "resize, and rotate them. The drawing is resized to 64×64 for inference."
    )
    original = canvas_to_image(canvas_result)
    masked, pixel_mask = apply_rect_masks(original, rectangles)

with right:
    st.subheader("2 · Mask preview")
    col1, col2 = st.columns(2)
    col1.image(
        original.resize((256, 256)),
        caption="Original drawing",
        use_container_width=True,
    )
    col2.image(
        masked.resize((256, 256)),
        caption="Masked input (magenta)",
        use_container_width=True,
    )
    st.caption(
        "The two mask controls update this preview. Masks are applied after "
        "the vector drawing is rasterized to the model's 64×64 input."
    )

st.divider()
st.subheader("3 · Context encoder and reconstruction")

if run_button:
    try:
        models = load_model(
            st.session_state.checkpoint_path, str(DEVICE)
        )
        reconstructed, heatmap, feature_grid, token_mask = run_inference(
            masked, pixel_mask, models
        )

        c1, c2, c3 = st.columns(3)
        c1.image(
            masked.resize((256, 256)),
            caption="Context encoder input",
            use_container_width=True,
        )
        c2.image(
            heatmap.resize((256, 256)),
            caption="Context feature activation",
            use_container_width=True,
        )
        c3.image(
            reconstructed.resize((256, 256)),
            caption="Reconstructed result",
            use_container_width=True,
        )

        c4, c5 = st.columns(2)
        c4.image(
            feature_grid,
            caption="16×16 spatial feature strength",
            use_container_width=True,
        )
        c5.image(
            Image.fromarray(
                (token_mask.numpy() * 255).astype(np.uint8)
            ).resize((256, 256), Image.Resampling.NEAREST),
            caption="16×16 cells selected as masked targets",
            use_container_width=True,
        )

        st.info(
            "The activation heatmap visualizes stronger CNN feature responses. "
            "It is not a semantic detector and does not label circles, squares, "
            "or other object classes."
        )

        buffer = BytesIO()
        reconstructed.save(buffer, format="PNG")
        st.download_button(
            "Download reconstructed image",
            data=buffer.getvalue(),
            file_name="jepa_reconstruction.png",
            mime="image/png",
        )
    except Exception as exc:
        st.error(f"Could not run reconstruction: {exc}")
else:
    st.info(
        "Draw shapes, adjust the masks, then click Reconstruct image."
    )
