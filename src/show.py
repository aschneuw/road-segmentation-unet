from PIL import Image


def black_to_red_alpha(pix, alpha):
    if pix[0] > 50:  # white becomes red transparent
        return (255, 0, 0, alpha)
    else:  # other become transparent
        return (0, 0, 0, 0)


def satimage_with_mask(satimage: Image, roads_mask: Image = None):
    if not roads_mask:
        return satimage

    mask = roads_mask.copy()
    transformed_mask = [black_to_red_alpha(pix, 60) for pix in mask.getdata()]
    mask.putdata(transformed_mask)
    return Image.alpha_composite(satimage, mask)


def satimage_with_mask_files(satimage_path: str, roads_mask_path: str = None):
    im = Image.open(satimage_path)
    im = im.convert("RGBA")

    mask = None
    if roads_mask_path:
        mask = Image.open(roads_mask_path)
        mask = mask.convert("RGBA")

    return satimage_with_mask(im, mask)
