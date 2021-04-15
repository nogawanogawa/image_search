from typing import Dict, List
from lib.es.crud import CRUD
from lib.feature.src.phash import *
from scipy.spatial import distance
import pandas as pd

class Document(CRUD):

    def __init__(self):
        super().__init__()
        self.index = "image"

    def search(self, text:str, category:str) -> List[Dict]:
        """キーワードで検索

        Args:
            text (str): 検索したいキーワード

        Returns:
            List[Dict]: 検索結果
        """
        if category == "":
            body = {
                "query": {
                    "match": {"content": text}
                }
            }
        else : 
            body = {
                "query": {
                    "bool" : {
                        "must" : [
                            {"match" : { "category" : category}}
                        ],
                        "should" : [
                            {"match": {"content": text}}
                        ]
                    }
                }
            }
        
        return self.execute_query(index=self.index, body=body, size=1000)

    def search_by_phash(self, hash_str:str) -> List[Dict]:
        """ハッシュ値での検索

        Args:
            text (str): 検索したいハッシュ

        Returns:
            List[Dict]: 検索結果
        """
        
        body = {
            "query": {
                "fuzzy": {
                "phash": {
                    "value": hash_str,
                    "fuzziness": "AUTO",
                    "max_expansions": 50,
                    "prefix_length": 0,
                    "transpositions": True,
                    "rewrite": "constant_score"
                    }
                }
            }
        }

        res = self.execute_query(index=self.index, body=body, size=100)
        df = pd.DataFrame(res)
        df['phash_list'] = df['phash'].str.split()
        df['query_hash'] = hash_str
        df['query_hash'] = df['query_hash'].str.split()

        df['distance'] = df['query_hash'].apply(distance.hamming, v="phash_list")
        
        # sort
        df = df.sort_values('distance')
        df = df.drop(["phash_list", "query_hash", "distance"], axis=1)

        return df.to_dict(orient='records')


    def search_by_embedding(self, embedding: List[float]) -> List[Dict]:
        """embeddingでの検索

        Args:
            text (str): 検索したいembedding

        Returns:
            List[Dict]: 検索結果
        """
        
        body = {
            "query" : {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": embedding}
                    }
                }
            }
        }

        return self.execute_query(index=self.index, body=body, size=100)