import cv2

def get_akaze_featrue(image_path) :

    akaze = cv2.AKAZE_create()

    img = cv2.imread(image_path)
    kp, des = akaze.detectAndCompute(img, None)

    return des
