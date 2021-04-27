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
from lib.feature.src.akaze import *
import pickle
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

  hash_search = st.button('hash')
  embedding_search = st.button('embedding')
  akaze_search = st.button('akaze')
    
#### Output
  # 1. Hashによる検索結果のソート
  if hash_search == True:
    document = Document()
    phash = str(get_hash(uploaded_file))
    for row in document.search_by_phash(phash):
      img = Image.open("/app/images/" + row["filename"])
      st.image(img, caption='pHash 検索結果',use_column_width=True)

  # 2. Embeddingによる検索結果のソート
  if embedding_search == True:
    document = Document()

    model = models.resnet34(pretrained=True)
    model.fc = nn.Identity()

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    
    with torch.no_grad():
      target = transform(image.convert('RGB'))
      output = model(target[None, ...])

    for row in document.search_by_embedding(output[0].tolist()):
      image = Image.open("/app/images/" + row["filename"])
      st.image(image, caption='Embedding 検索結果',use_column_width=True)

  # 3. 局所特徴量による検索結果のソート
  if akaze_search == True:
    document = Document()
    docs = document.search_all()

    l_str = [str(i) for i in range(20)]
    columns = ['filename'] + l_str

    res_df  = pd.DataFrame(docs)[columns]

    model_name = '/app/model/kmeans_model.pkl'

    kmeans_model = pickle.load(open(model_name, 'rb'))

    with open(uploaded_file.name,"wb") as f:
      f.write(uploaded_file.getbuffer())
      
    features = get_akaze_feature(uploaded_file.name)
    features = kmeans_model.predict(features)

    d = {
          0:0, 1:0, 2:0, 3:0, 4:0, 
          5:0, 6:0, 7:0, 8:0, 9:0, 
          10:0, 11:0, 12:0, 13:0, 14:0, 
          15:0, 16:0, 17:0, 18:0, 19:0    
        }
    
    for f in features:
      d[f] = d[f] + 1

    def intersection(x, a):
      return x if x < a else a

    for i in range(20):
      res_df[str(i)] = res_df[str(i)].apply(intersection, a=d[i])

    res_df['intersection'] = res_df[l_str].sum(axis=1)
    res_df = res_df.sort_values('intersection', ascending=False)

    for img in res_df['filename']:
      image = Image.open("/app/images/" + img)
      st.image(image, caption='Akaze 検索結果',use_column_width=True)