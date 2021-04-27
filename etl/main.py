import os
import glob
from metaflow import FlowSpec, step
from lib.es.document import Document
from lib.es.index import Index
from step.extract.extract import *
from lib.feature.src.embedding import *
from lib.feature.src.phash import *
from lib.feature.src.akaze import *

import torch
from torchvision import models
from torchvision.models import resnet34
from torch import nn
import pickle
from sklearn.cluster import KMeans


model_name = '/app/model/kmeans_model.pkl'

class Workflow(FlowSpec):

    @step
    def start(self):
        self.next(self.initialize_index)

    @step
    def initialize_index(self):
        """indexを削除し、初期のマッピングを作成"""
        index = Index()

        # indexの一覧を取得
        index_list = index.get_all()

        # 残っているindexの削除
        for i in index_list:
            res = index.delete(i)
            assert res["acknowledged"] == True

        # mappingが存在するindexを作成
        path = os.getcwd()
        mapping_dir = os.path.join(path, "lib/es/mapping")
        mappings = glob.glob(os.path.join(mapping_dir, '*.json'))

        for mapping in mappings:
            filename = mapping.split("/")[-1]
            index_name = filename.split(".")[0]
            res = index.create(index_name=index_name)
            assert res["acknowledged"] == True

        self.next(self.build_model)

    @step
    def build_model(self):
        """ build kmeans model  """
        
        PATH = "/app/images"
        l = glob.glob(os.path.join(PATH, '*.jpg'))

        features = None
        for filepath in l:
            try:
                if features is None:
                    features = get_akaze_feature(filepath)
                else :
                    features = np.append(features, get_akaze_feature(filepath), axis=0)
            except:
                pass

        model = KMeans(n_clusters=20, init='k-means++', random_state=0).fit(features)
        pickle.dump(model, open(model_name, 'wb'))

        self.next(self.extract_load)

    @step
    def extract_load(self):
        """dictの内容をESに挿入"""
        document = Document()

        PATH = "/app/images"
        l = glob.glob(os.path.join(PATH, '*.jpg'))
        dataset = ImageDataSet(PATH, l)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        model = models.resnet34(pretrained=True)
        model.fc = nn.Identity()

        kmeans_model = pickle.load(open(model_name, 'rb'))

        for i, (data, filepath) in enumerate(dataloader):

            with torch.no_grad():
                output = model(data)

            doc = {}

            filename = filepath[0].split("/")[-1]

            doc["filename"] = filename
            doc["phash"] = str(get_hash(filepath[0]))
            doc["embedding"] = output[0].tolist()

            d = {
                0:0, 1:0, 2:0, 3:0, 4:0, 
                5:0, 6:0, 7:0, 8:0, 9:0, 
                10:0, 11:0, 12:0, 13:0, 14:0, 
                15:0, 16:0, 17:0, 18:0, 19:0    
                }

            try:
                features = get_akaze_feature(filepath[0])
                features = kmeans_model.predict(features)

                for f in features:
                    d[f] = d[f] + 1

            except:
                pass
            
            doc.update(d)

            res = document.register(doc)
            #print(res)
            #print(filepath)

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == '__main__':
    Workflow()