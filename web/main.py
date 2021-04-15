import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import time
from lib.es.document import Document
import nlplot
from lib.feature.src.phash import *
from lib.feature.src.embedding import *
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

document = Document()

st.title('Image Search sample app')

#### Input
uploaded_file = st.sidebar.file_uploader("ファイルアップロード", type='jpg')

#### Input -> Output
if uploaded_file is not None:
  image = Image.open(uploaded_file)
  st.image(image, caption='検索画像',use_column_width=True)

  answer = st.button('search')
    
#### Output
  if answer == True:
    phash = str(get_hash(uploaded_file))

    document = Document()
    for row in document.search_by_phash(phash):
      img = Image.open("/app/images/" + row["filename"])
      st.image(img, caption='pHash 検索結果',use_column_width=True)

    model = models.resnet34(pretrained=True)
    model.fc = nn.Identity()

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    
    with torch.no_grad():
      target = transform(image.convert('RGB'))
      output = model(target[None, ...])

    for row in document.search_by_embedding(output[0].tolist()):
      image = Image.open("/app/images/" + row["filename"])
      st.image(image, caption='Embedding 検索結果',use_column_width=True)



