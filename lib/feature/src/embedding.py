import os
import glob
import re 
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet34
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import classification_report
from sklearn import preprocessing
import datetime


class ImageDataSet(Dataset):
    def __init__(self, path, l):
        
        self.images = []
        self.paths = []

        for filemames in l:
            self.images.append(os.path.join(path, filemames))
            self.paths.append(os.path.join(path, filemames))

        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        filepath = self.paths[idx]

        image = Image.open(self.images[idx])
        image = image.convert('RGB')
        return self.transform(image), filepath


