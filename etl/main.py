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

        for i, (data, filepath) in enumerate(dataloader):
            
            with torch.no_grad():
                output = model(data)

            doc = {}

            filename = filepath[0].split("/")[-1]

            doc["filename"] = filename
            doc["phash"] = str(get_hash(filepath[0]))
            doc["embedding"] = output[0].tolist()

            res = document.register(doc)
            print(res)


        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == '__main__':
    Workflow()