# Copyright 2022 Cristian Grosu
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import annoy
import os
from utils.Logger import Logger
from utils.Database import Database

EMBEDDING_DIMENSION = 134
INDEX_FILENAME = os.path.join("database", "indexes", "features_index.ann")
TREES_NUMBER = 500

class ANN:
    """_summary_: this is a class that contains the ANN model and its methods
    
    We use ANNOY which is a library for approximate nearest neighbors in Python.
    Abreviations: ANNOY = Approximate Nearest Neighbors Oh Yeah (funny, right?)
    """
    
    def __init__(self, log = False):
        self.logger = Logger(active=log)
        self.mapping = None
        self.db = Database()
    
    def build_index(self, metric='angular', num_trees=TREES_NUMBER, index_filename = INDEX_FILENAME):
        """Builds an ANNOY index for the features in the database"""
        
        annoy_index = annoy.AnnoyIndex(EMBEDDING_DIMENSION, metric=metric)

        # Iterate over the embeddings in the database
        rows = self.db.get_table_data(table = "features")
        
        self.logger.log("Getting the embeddings from the database...")
        for row in rows:
            # Get the identifier
            id = row[-2] # shape id
            # Get the embedding
            embedding = self.get_embedding_from_feature_row(row)        
            # Add the embedding to the index
            annoy_index.add_item(id, embedding)
        
        self.logger.log("Building index...")
        self.logger.log("Building index with {} trees...".format(num_trees))
        annoy_index.build(n_trees=num_trees)
        self.logger.success("Index is built!")        
        
        self.save_index(annoy_index, index_filename)
    
    @staticmethod
    def get_embedding_from_feature_row(row):
        r = row[1:-2]
        [a3, d1, d2, d3, d4] = r[8:]
        embedding = [float(x) for x in r[:8]]
        for v in [a3, d1, d2, d3, d4]:
            for x in v:
                embedding.append(x)
        return embedding    
        
    def load_index(self, index_filename = INDEX_FILENAME):
        index = annoy.AnnoyIndex(EMBEDDING_DIMENSION)
        index.load(index_filename, prefault=True)
        self.logger.log("Annoy index is loaded from disk.")
        return index        

    def save_index(self, annoy_index, index_filename):
        self.logger.log("Saving index to disk...")        
        annoy_index.save(index_filename)
        self.logger.log("Index is saved to disk.")
        self.logger.log("Index file size: {} MB".format(round(os.path.getsize(index_filename) / float(1024 ** 2), 2)))                
        annoy_index.unload()

    def get_nearest_neighbors(self, annoy_index, embedding, n=10):
        """Returns the n nearest neighbors of the embedding"""
        neighbors = annoy_index.get_nns_by_vector(embedding, n, include_distances=True)
        return neighbors 