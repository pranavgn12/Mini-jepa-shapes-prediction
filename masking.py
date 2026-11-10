import random

from PIL import Image, ImageDraw


MASK_COLOUR = (255, 0, 255)


def apply_random_mask(image, min_rectangles=1, max_rectangles=3):
    """Apply 1-3 random rectangular magenta masks to a PIL image.

    Returns:
      masked_image: RGB image with hidden regions filled in magenta
      pixel_mask:   L image, 255 for hidden pixels and 0 for visible pixels
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image")

    image = image.convert("RGB")
    width, height = image.size
    masked_image = image.copy()
    pixel_mask = Image.new("L", (width, height), 0)

    image_draw = ImageDraw.Draw(masked_image)
    mask_draw = ImageDraw.Draw(pixel_mask)

    rectangle_count = random.randint(min_rectangles, max_rectangles)
    for _ in range(rectangle_count):
        rect_width = random.randint(max(8, width // 8), max(12, width // 2))
        rect_height = random.randint(max(8, height // 8), max(12, height // 2))
        rect_width = min(rect_width, width)
        rect_height = min(rect_height, height)

        left = random.randint(0, width - rect_width)
        top = random.randint(0, height - rect_height)
        right = left + rect_width - 1
        bottom = top + rect_height - 1
        rectangle = (left, top, right, bottom)

        image_draw.rectangle(rectangle, fill=MASK_COLOUR)
        mask_draw.rectangle(rectangle, fill=255)

    return masked_image, pixel_mask
