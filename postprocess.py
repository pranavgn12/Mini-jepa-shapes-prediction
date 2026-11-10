"""Boundary-aware colour expansion for reconstructed geometric images.

Visible, unmasked shape pixels act as colour seeds. Predicted pixels are
eligible for colour expansion only when they are sufficiently far from the
estimated background AND closer to a known shape colour than to the background.
Background-like pixels therefore become barriers between neighbouring shapes.

A multi-source breadth-first expansion assigns eligible pixels to the nearest
visible colour seed. This is a presentation-only post-process; it is not part
of training or the loss.
"""

from collections import deque

import numpy as np
import torch


def _mode_colour(pixels):
    """Most frequent exact RGB value, suitable for the flat synthetic background."""
    pixels_u8 = np.clip(np.rint(pixels * 255.0), 0, 255).astype(np.uint8)
    packed = (
        (pixels_u8[:, 0].astype(np.uint32) << 16)
        | (pixels_u8[:, 1].astype(np.uint32) << 8)
        | pixels_u8[:, 2].astype(np.uint32)
    )
    values, counts = np.unique(packed, return_counts=True)
    mode = values[np.argmax(counts)]
    return np.array(
        [(mode >> 16) & 255, (mode >> 8) & 255, mode & 255],
        dtype=np.float32,
    ) / 255.0


def _fallback_palette(predicted_pixels, background, max_colours=8):
    """Estimate a limited palette if no visible shape-colour seeds exist."""
    if predicted_pixels.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    pixels = np.asarray(predicted_pixels, dtype=np.float32)
    pixels = pixels[np.linalg.norm(pixels - background[None, :], axis=1) > 0.10]
    if len(pixels) == 0:
        return np.empty((0, 3), dtype=np.float32)

    unique = np.unique(np.rint(pixels * 255).astype(np.uint8), axis=0)
    points = unique.astype(np.float32) / 255.0
    if len(points) == 1:
        return points

    k = min(max_colours, len(points))
    first = int(np.argmax(np.linalg.norm(points - background[None, :], axis=1)))
    centers = [points[first]]
    while len(centers) < k:
        distances = np.stack(
            [np.linalg.norm(points - center[None, :], axis=1) for center in centers],
            axis=1,
        )
        centers.append(points[int(np.argmax(np.min(distances, axis=1)))])
    centers = np.asarray(centers, dtype=np.float32)

    for _ in range(12):
        distances = np.linalg.norm(pixels[:, None, :] - centers[None, :, :], axis=2)
        labels = distances.argmin(axis=1)
        updated = centers.copy()
        for cluster in range(k):
            members = pixels[labels == cluster]
            if len(members):
                updated[cluster] = members.mean(axis=0)
        if np.allclose(updated, centers, atol=1e-4):
            break
        centers = updated

    result = []
    for center in centers:
        if np.linalg.norm(center - background) > 0.10 and all(
            np.linalg.norm(center - old) > 0.08 for old in result
        ):
            result.append(center)
    return np.asarray(result, dtype=np.float32).reshape(-1, 3)


def refine_colour_regions(
    image,
    known_mask=None,
    background_threshold=0.16,
    background_margin=0.02,
):
    """Expand visible shape colours without painting across background gaps.

    Args:
        image: HWC RGB NumPy array or CHW RGB torch tensor in [0, 1]. Prefer
            the raw completion (observed pixels + decoder predictions).
        known_mask: Optional HxW or 1xHxW mask where 1 marks observed,
            unmasked input pixels. These pixels provide exact colour seeds and
            are preserved unchanged.
        background_threshold: Minimum Euclidean RGB distance from the
            estimated background for a predicted pixel to be considered shape.
        background_margin: A predicted pixel must be at least this much closer
            (in RGB distance) to a known shape colour than to the background.
            This makes faint background-colour bleed act as a separator.

    Returns:
        Refined image in the same representation family as the input.
    """
    is_tensor = torch.is_tensor(image)
    if is_tensor:
        source_device = image.device
        source_dtype = image.dtype
        array = image.detach().float().cpu().permute(1, 2, 0).numpy()
    else:
        source_device = source_dtype = None
        array = np.asarray(image, dtype=np.float32)

    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected RGB image (H,W,3), got {array.shape}")
    array = np.clip(array, 0.0, 1.0)
    height, width, _ = array.shape

    if known_mask is None:
        known = np.zeros((height, width), dtype=bool)
    else:
        if torch.is_tensor(known_mask):
            mask_array = known_mask.detach().float().cpu().numpy()
        else:
            mask_array = np.asarray(known_mask, dtype=np.float32)
        mask_array = np.squeeze(mask_array)
        if mask_array.shape != (height, width):
            raise ValueError(
                f"known_mask must have shape {(height, width)}, got {mask_array.shape}"
            )
        known = mask_array > 0.5

    observed_pixels = array[known]
    if len(observed_pixels) == 0:
        observed_pixels = array.reshape(-1, 3)

    background = _mode_colour(observed_pixels)
    observed_u8 = np.clip(np.rint(observed_pixels * 255), 0, 255).astype(np.uint8)
    observed_colours = np.unique(observed_u8, axis=0).astype(np.float32) / 255.0
    palette = observed_colours[
        np.linalg.norm(observed_colours - background[None, :], axis=1) > 1e-5
    ]

    predicted_region = ~known
    predicted_pixels = array[predicted_region]
    if len(palette) == 0:
        palette = _fallback_palette(predicted_pixels, background)

    refined = array.copy()
    background_distance = np.linalg.norm(
        array - background[None, None, :], axis=2
    )
    refined[predicted_region] = background

    if len(palette) == 0:
        # There are no trustworthy colour seeds; keep the known pixels and
        # conservatively mark uncertain predicted pixels as background.
        refined[known] = array[known]
        if is_tensor:
            return torch.from_numpy(refined).permute(2, 0, 1).to(
                device=source_device, dtype=source_dtype
            )
        return refined

    # For each pixel, compute the closest known shape colour. A predicted pixel
    # becomes foreground only if it is sufficiently unlike the background and
    # shape-colour evidence beats background evidence by a safety margin.
    colour_distances = np.linalg.norm(
        array[:, :, None, :] - palette[None, None, :, :], axis=3
    )
    nearest_colour_label = colour_distances.argmin(axis=2)
    nearest_colour_distance = colour_distances.min(axis=2)

    predicted_foreground = (
        predicted_region
        & (background_distance > float(background_threshold))
        & (
            nearest_colour_distance + float(background_margin)
            < background_distance
        )
    )

    # Exact observed shape pixels remain valid seeds even when a shape colour
    # happens to be relatively close to the background colour.
    seed_mask = known & (background_distance > 1e-5)
    traversable = predicted_foreground | seed_mask

    labels = np.full((height, width), -1, dtype=np.int32)
    distances = np.full((height, width), np.iinfo(np.int32).max, dtype=np.int32)
    queue = deque()

    for y, x in np.argwhere(seed_mask):
        label = int(nearest_colour_label[y, x])
        labels[y, x] = label
        distances[y, x] = 0
        queue.append((int(y), int(x)))

    # Multi-source BFS cannot cross background-like pixels. Those pixels stay
    # background, forming gaps between adjacent/touching predicted regions.
    neighbours = ((-1, 0), (1, 0), (0, -1), (0, 1))
    while queue:
        y, x = queue.popleft()
        next_distance = distances[y, x] + 1
        label = labels[y, x]
        for dy, dx in neighbours:
            ny, nx = y + dy, x + dx
            if (
                0 <= ny < height
                and 0 <= nx < width
                and traversable[ny, nx]
                and next_distance < distances[ny, nx]
            ):
                distances[ny, nx] = next_distance
                labels[ny, nx] = label
                queue.append((ny, nx))

    seeded = predicted_foreground & (labels >= 0)
    refined[seeded] = palette[labels[seeded]]

    # A fully hidden foreground island has no visible colour seed. Only assign
    # such pixels when their predicted colour is clearly closer to a palette
    # colour than to the background; otherwise leave them as background.
    unseeded = predicted_foreground & (labels < 0)
    if unseeded.any():
        refined[unseeded] = palette[nearest_colour_label[unseeded]]

    # Preserve every observed input pixel exactly.
    refined[known] = array[known]

    if is_tensor:
        return torch.from_numpy(refined).permute(2, 0, 1).to(
            device=source_device, dtype=source_dtype
        )
    return refined
