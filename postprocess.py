"""Deterministic colour-region cleanup for reconstructed RGB images.

The background is estimated from the image border. Pixels sufficiently far
from that colour become foreground candidates. Each 4-connected foreground
component is filled with its median RGB colour; small components are treated
as isolated noise and returned to the background.

This is an evaluation/presentation post-process, not a learned model component.
"""

from collections import deque

import numpy as np
import torch


def _border_pixels(image):
    """Return unique border pixels from an HWC float RGB image."""
    return np.concatenate(
        [image[0, :, :], image[-1, :, :], image[1:-1, 0, :], image[1:-1, -1, :]],
        axis=0,
    )


def refine_colour_regions(
    image,
    background_threshold=0.16,
    min_component_area=2,
):
    """Convert a reconstructed image into flat-colour connected regions.

    Args:
        image: HWC RGB NumPy array or CHW RGB torch tensor, values in [0, 1].
        background_threshold: Euclidean RGB distance from estimated background.
            0.16 is approximately 41 intensity levels in 8-bit RGB distance.
        min_component_area: Smaller foreground components are removed as noise.

    Returns:
        An image in the same representation family as the input (NumPy HWC or
        torch CHW), with the same range and dimensions.
    """
    is_tensor = torch.is_tensor(image)
    if is_tensor:
        source_device = image.device
        source_dtype = image.dtype
        array = image.detach().float().cpu().permute(1, 2, 0).numpy()
    else:
        array = np.asarray(image, dtype=np.float32)

    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected RGB image (H,W,3) or (3,H,W), got {array.shape}")

    array = np.clip(array, 0.0, 1.0)
    height, width, _ = array.shape

    # The synthetic dataset has a uniform background. A border median is
    # robust to occasional shapes touching one or more image edges.
    background = np.median(_border_pixels(array), axis=0)
    distance = np.linalg.norm(array - background[None, None, :], axis=2)
    foreground = distance > float(background_threshold)

    cleaned = np.empty_like(array)
    cleaned[:] = background
    visited = np.zeros((height, width), dtype=bool)

    # Four-neighbour connectivity avoids joining regions that only touch at
    # a single diagonal corner.
    neighbours = ((-1, 0), (1, 0), (0, -1), (0, 1))

    for start_y in range(height):
        for start_x in range(width):
            if not foreground[start_y, start_x] or visited[start_y, start_x]:
                continue

            queue = deque([(start_y, start_x)])
            visited[start_y, start_x] = True
            component = []

            while queue:
                y, x = queue.popleft()
                component.append((y, x))
                for dy, dx in neighbours:
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and foreground[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        queue.append((ny, nx))

            if len(component) < int(min_component_area):
                continue

            ys, xs = zip(*component)
            region_colour = np.median(array[np.asarray(ys), np.asarray(xs)], axis=0)
            cleaned[np.asarray(ys), np.asarray(xs)] = region_colour

    if is_tensor:
        return torch.from_numpy(cleaned).permute(2, 0, 1).to(
            device=source_device, dtype=source_dtype
        )
    return cleaned
