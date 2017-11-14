import numpy as np
from lib.config import cfg
from PIL import Image


def image_transform(img, crop_x, crop_y, crop_loc=None, color_tint=None):
    """
    Takes numpy.array img
    """

    # Slight translation
    if cfg.TRAIN.RANDOM_CROP and not crop_loc:
        crop_loc = [np.random.randint(0, crop_y), np.random.randint(0, crop_x)]

    if crop_loc:
        cr, cc = crop_loc
        height, width, _ = img.shape
        img_h = height - crop_y
        img_w = width - crop_x
        img = img[cr:cr + img_h, cc:cc + img_w]
        # depth = depth[cr:cr+img_h, cc:cc+img_w]

    flipped = 0
    if cfg.TRAIN.FLIP and np.random.rand() > 0.5:
        img = img[:, ::-1, ...]
        flipped = 1

    return img, cr, cc, flipped


def crop_center(im, new_height, new_width):
    height = im.shape[0]  # Get dimensions
    width = im.shape[1]
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2
    return im[top:bottom, left:right], top, left, 0


def add_random_color_background(im, color_range):
    r, g, b = [np.random.randint(color_range[i][0], color_range[i][1] + 1) for i in range(3)]

    if isinstance(im, Image.Image):
        im = np.array(im)

    # Add color background. Preserve alpha channel as a ground-truth mask.
    assert im.shape[2] == 4, 'Image must have an alpha channel'
    bg = (np.expand_dims(im[:, :, 3], axis=2) == 0).astype(np.float)
    bg_color = np.array([[[r, g, b]]])
    im[:, :, :3] = bg * bg_color + (1 - bg) * im[:, :, :3]
    im[:, :, 3] = 1 - bg[..., 0]

    return im


def preprocess_img(im, train=True):
    # add random background
    im = add_random_color_background(im, cfg.TRAIN.NO_BG_COLOR_RANGE if train else
                                     cfg.TEST.NO_BG_COLOR_RANGE)

    if train:
        transformed = image_transform(im, cfg.TRAIN.PAD_X, cfg.TRAIN.PAD_Y)
    else:
        transformed = crop_center(im, cfg.CONST.IMG_H, cfg.CONST.IMG_W)

    t_im, cr, cc, flipped = transformed

    # Scale image
    t_im = t_im / 255.

    return t_im, cr, cc, flipped


def test(fn):
    import matplotlib.pyplot as plt
    cfg.TRAIN.RANDOM_CROP = True
    im = Image.open(fn)
    im = np.asarray(im)[:, :, :3]
    imt = image_transform(im, 10, 10)
    plt.imshow(imt)
    plt.show()
