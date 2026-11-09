
import math
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw

from masking import apply_random_mask


IMAGE_SIZE = 64


def rotate_points(points, angle, cx, cy):
    """Rotate polygon points around their center."""

    angle = math.radians(angle)
    rotated = []

    for x, y in points:
        dx = x - cx
        dy = y - cy

        rx = dx * math.cos(angle) - dy * math.sin(angle)
        ry = dx * math.sin(angle) + dy * math.cos(angle)

        rotated.append((cx + rx, cy + ry))

    return rotated


def generate_shape_image():
    image = Image.new(
        "RGB",
        (IMAGE_SIZE, IMAGE_SIZE),
        (
            random.randint(180, 245),
            random.randint(180, 245),
            random.randint(180, 245)
        )
    )

    draw = ImageDraw.Draw(image)

    num_shapes = random.randint(3, 8)

    for _ in range(num_shapes):
        cx = random.randint(8, 56)
        cy = random.randint(8, 56)

        width = random.randint(10, 26)
        height = random.randint(10, 26)

        angle = random.randint(0, 360)

        # Avoid using the reserved magenta mask colour.
        color = tuple(
            random.randint(20, 235) for _ in range(3)
        )

        shape = random.choice([
            "circle",
            "square",
            "rectangle",
            "triangle"
        ])

        if shape == "circle":
            radius = min(width, height) // 2

            draw.ellipse(
                (
                    cx - radius,
                    cy - radius,
                    cx + radius,
                    cy + radius
                ),
                fill=color
            )

        else:
            if shape == "square":
                width = height = min(width, height)

            if shape == "triangle":
                points = [
                    (cx, cy - height // 2),
                    (cx - width // 2, cy + height // 2),
                    (cx + width // 2, cy + height // 2)
                ]
            else:
                points = [
                    (cx - width // 2, cy - height // 2),
                    (cx + width // 2, cy - height // 2),
                    (cx + width // 2, cy + height // 2),
                    (cx - width // 2, cy + height // 2)
                ]

            points = rotate_points(points, angle, cx, cy)

            draw.polygon(points, fill=color)

    return image


def image_to_tensor(image):
    array = np.array(image, dtype=np.float32) / 255.0

    return torch.from_numpy(array).permute(2, 0, 1)


def mask_to_tensor(mask):
    array = np.array(mask, dtype=np.float32) / 255.0

    return torch.from_numpy(array).unsqueeze(0)


class ShapesDataset(Dataset):

    def __init__(self, length=10000):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        # Generate complete image.
        original = generate_shape_image()

        # Generate masked image and ground-truth mask.
        masked, mask = apply_random_mask(original)

        return {
            "original": image_to_tensor(original),
            "masked": image_to_tensor(masked),
            "mask": mask_to_tensor(mask)
        }

