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
import pickle
import annoy
import os
from utils.Logger import Logger
from utils.Database import Database

class ANN:
    """_summary_: this is a class that contains the ANN model and its methods
    
    We use ANNOY which is a library for approximate nearest neighbors in Python.
    Abreviations: ANNOY = Approximate Nearest Neighbors Oh Yeah (funny, right?)
    """
    
    def __init__(self, log = False):
        self.logger = Logger(active=log)
        self.mapping = None
        self.db = Database()
    
    def build_index(self, index_filename, vector_length, metric='angular', num_trees=100):
        """Builds an ANNOY index for the features in the database"""
        
        annoy_index = annoy.AnnoyIndex(vector_length, metric=metric)

        # Iterate over the embeddings in the database
        rows = self.db.get_table_data(table = "features")
        
        for row in rows:
            # Get the embedding
            embedding = row[1:-2]
            # Get the identifier
            id = row[-2] # shape id
            # Add the embedding to the index
            annoy_index.add_item(id, embedding)
        
        self.logger.log("Building index...")
        self.logger.log("Building index with {} trees...".format(num_trees))
        annoy_index.build(n_trees=num_trees)
        self.logger.success("Index is built!")        

        self.save_index(annoy_index, index_filename)
        
        # Save the mapping to disk
        
    def load_index(self, index_filename):
        embedding_dimension = 512 # wtf?
        index = annoy.AnnoyIndex(embedding_dimension)
        index.load(index_filename, prefault=True)
        self.logger.log("Annoy index is loaded from disk.")
        
        with open(index_filename + '.mapping', 'rb') as handle:
            self.mapping = pickle.load(handle)
            self.logger.log("Mapping is loaded from disk.")            

    def save_index(self, annoy_index, index_filename):
        self.logger.log("Saving index to disk...")        
        annoy_index.save(index_filename)
        self.logger.log("Index is saved to disk.")
        self.logger.log("Index file size: {} MB".format(round(os.path.getsize(index_filename) / float(1024 ** 2), 2)))                
        annoy_index.unload()

    def save_mapping(self, mapping, mapping_filename):
        self.logger.log("Saving mapping to disk...")        
        with open(mapping_filename, 'wb') as handle:
            pickle.dump(mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.logger.log("Mapping is saved to disk.")
        self.logger.log("Mapping file size: {} MB".format(round(os.path.getsize(mapping_filename) / float(1024 ** 2), 2)))

    def get_nearest_neighbors(self, annoy_index, embedding, n=10):
        """Returns the n nearest neighbors of the embedding"""
        neighbors = annoy_index.get_nns_by_vector(embedding, n, include_distances=True)
        return neighbors 