import cv2
from pathlib import Path


def read_image(path: Path):
    """ Read an image from path as RGB """
    if not Path(path).exists():
        raise FileNotFoundError(f"No image at {path}.")
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise IOError(f"Could not read image at {path}.")
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image