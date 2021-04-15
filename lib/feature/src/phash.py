import imagehash
import cv2
from PIL import Image

def get_hash(image_path) :

    try:
        hash = imagehash.average_hash(Image.open(image_path))
    except cv2.error:
        ret = 100000

    return hash
