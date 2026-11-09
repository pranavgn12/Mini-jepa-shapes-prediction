
import random
from PIL import Image, ImageDraw

IMAGE_SIZE = 64
MASK_COLOR = (255, 0, 255)


def apply_random_mask(image, min_blocks=1, max_blocks=3):
    """
    Args:
        image: Original PIL RGB image.

    Returns:
        masked_image: Image with masked regions replaced by magenta.
        mask: PIL grayscale image (255 = masked, 0 = visible).
    """

    masked_image = image.copy()

    mask = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
    draw = ImageDraw.Draw(mask)

    num_blocks = random.randint(min_blocks, max_blocks)

    for _ in range(num_blocks):
        width = random.randint(8, 24)
        height = random.randint(8, 24)

        x1 = random.randint(0, IMAGE_SIZE - width)
        y1 = random.randint(0, IMAGE_SIZE - height)

        x2 = x1 + width - 1
        y2 = y1 + height - 1

        draw.rectangle((x1, y1, x2, y2), fill=255)

    # Replace masked pixels with magenta.
    pixels = masked_image.load()
    mask_pixels = mask.load()

    for y in range(IMAGE_SIZE):
        for x in range(IMAGE_SIZE):
            if mask_pixels[x, y] == 255:
                pixels[x, y] = MASK_COLOR

    return masked_image, mask

