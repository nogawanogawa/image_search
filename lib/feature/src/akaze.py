import cv2
from typing import List

def get_akaze_feature(image_path) :

    akaze = cv2.AKAZE_create()

    img = cv2.imread(image_path)
    kp, des = akaze.detectAndCompute(img, None)

    return des

def build_kmeans_model(features: List[List[int]], filepath:str)->None:
    """ build kmeans model from akaze features

    Args:
        features (List[List[int]]): [description]
        filepath (str): [description]

    """    

    model = KMeans(n_clusters=20, init='k-means++', random_state=0).fit(b)
    pickle.dump(model, open(filepath, 'wb'))

    return


def get_akaze_histgram(features: List[List[int]]) :
    """[summary]

    Args:
        features (List[List[int]]): [description]
    """
    return 