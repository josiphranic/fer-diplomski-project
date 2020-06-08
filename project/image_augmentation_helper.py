import cv2


def get_augmented_images(img):
    images = [horizontal_shift(img, ratio=0.3),
              horizontal_shift(img, ratio=0.1),
              horizontal_shift(img, ratio=0.2),
              vertical_shift(img, ratio=0.3),
              vertical_shift(img, ratio=0.1),
              vertical_shift(img, ratio=0.2),
              horizontal_shift(img, ratio=-0.3),
              horizontal_shift(img, ratio=-0.1),
              horizontal_shift(img, ratio=-0.2),
              vertical_shift(img, ratio=-0.3),
              vertical_shift(img, ratio=-0.1),
              vertical_shift(img, ratio=-0.2),
              zoom(img),
              # vertical_shift(horizontal_shift(img, ratio=0.2), ratio=-0.2),
              # vertical_shift(horizontal_shift(img, ratio=-0.2), ratio=0.2),
              # vertical_shift(horizontal_shift(img, ratio=-0.2), ratio=-0.2),
              # vertical_shift(horizontal_shift(img, ratio=0.2), ratio=0.2),
              horizontal_flip(vertical_shift(img, ratio=-0.2)),
              horizontal_flip(horizontal_shift(img, ratio=-0.2)),
              horizontal_flip(vertical_shift(img, ratio=0.2)),
              horizontal_flip(horizontal_shift(img, ratio=0.2)),
              horizontal_flip(zoom(img)),
              horizontal_flip(img),
              horizontal_flip(rotation(img, angle=5)),
              horizontal_flip(rotation(img, angle=-5)),
              rotation(img, angle=5),
              rotation(img, angle=-5),
              rotation(img, angle=12),
              rotation(img, angle=-12)]
    return images


def fill(img, h, w):
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
    return img


def horizontal_shift(img, ratio=0.3):
    h, w = img.shape[:2]
    to_shift = w*ratio
    if ratio > 0:
        img = img[:, :int(w-to_shift), :]
    if ratio < 0:
        img = img[:, int(-1*to_shift):, :]
    img = fill(img, h, w)
    return img


def vertical_shift(img, ratio=0.3):
    h, w = img.shape[:2]
    to_shift = h*ratio
    if ratio > 0:
        img = img[:int(h-to_shift), :, :]
    if ratio < 0:
        img = img[int(-1*to_shift):, :, :]
    img = fill(img, h, w)
    return img


def zoom(img, start=0.1, value=0.8):
    if value > 1 or value < 0:
        raise Exception()
    h, w = img.shape[:2]
    h_taken = int(value*h)
    w_taken = int(value*w)
    h_start = int(start*h)
    w_start = int(start*w)
    img = img[h_start:h_start+h_taken, w_start:w_start+w_taken, :]
    img = fill(img, h, w)
    return img


def horizontal_flip(img):
    return cv2.flip(img, 1)


def vertical_flip(img):
    return cv2.flip(img, 0)


def rotation(img, angle=30):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, m, (w, h))
    return img
