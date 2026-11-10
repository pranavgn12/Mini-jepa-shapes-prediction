from PIL import Image

# Open images in separate windows
original_img = Image.fromarray(
    (original.permute(1, 2, 0).numpy() * 255).astype("uint8")
)

masked_img = Image.fromarray(
    (masked.permute(1, 2, 0).numpy() * 255).astype("uint8")
)

mask_img = Image.fromarray(
    (mask.squeeze(0).numpy() * 255).astype("uint8")
)

original_img.show(title="Original Image")
masked_img.show(title="Masked Image")
mask_img.show(title="Ground Truth Mask")
