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

import math
from Shape import Shape
from utils.Logger import Logger
from utils.Database import Database
import os
import numpy as np
from matplotlib import pyplot as plt

class FeatureExtractor:
    def __init__(self, log=False):
        self.logger = Logger(active=log)
        self.db = Database(log=log)
        self.db.create_features_table()
        
    def extract_features(self):     
        shapes = self.db.get_table_data(table='shapes', columns=['id', 'file_name'])
        for shape_data in shapes:
            [id, file_name] = shape_data
            
            sql = f"""SELECT id from features where shape_id = {id}"""
            self.db.cursor.execute(sql)
            row = self.db.cursor.fetchone()
            
            if row is None or row == []:
                self.logger.log("Extracting features for shape: " + file_name)
                shape = Shape(file_name=file_name)
                feature_id  = self.db.insert_features_data(shape, shape_id = id)
                self.db.update_shape_feature_id(id, feature_id = feature_id)
                
                self.logger.log(f"Extracted features for shape with id: {id}")
            else:
                self.logger.log(f"Features for shape with id: {id} already exist in database")
        # closing the db connection
        self.db.close()
    
    def extract_feature(self, feature = 'A3'):
        # this function is used to extract a single feature
        # it assumes that the features table is already created and populated
        shapes = self.db.get_table_data(table='shapes', columns=['id', 'file_name'])
        for shape_data in shapes:
            [id, file_name] = shape_data
            sql = f"""SELECT id from features where shape_id = {id}"""
            self.db.cursor.execute(sql)
            row = self.db.cursor.fetchone()
            if row:
                self.logger.log(f"Extracting feature {feature} for shape with id: {id}")
                shape = Shape(file_name=file_name)
                self.db.insert_feature_data(shape, shape_id = id, feature = feature)
                self.logger.log(f"Extracted feature {feature} for shape with id: {id}")
    
    def compute_statistics(self, type="A3", limit = 10):
        self.logger.log("Computing statistics for " + type + " feature")
        
        os.makedirs(f"statistics/features/{type}", exist_ok=True)
        
        classes = np.unique(self.db.get_table_data(table='shapes', columns=['class']))
        
        for class_name in classes:
            self.logger.log("Computing statistics for class: " + class_name)
            
            sql = f"""SELECT {type} from features join shapes on features.shape_id = shapes.id where shapes.class = '{class_name}' LIMIT {limit}"""
            self.db.cursor.execute(sql)
            rows = self.db.cursor.fetchall()
            rows = [row[0] for row in rows]
            
            if type in ['A3', 'D1', 'D2', 'D3', 'D4']:
                self._plot_signature(rows, filename = f"statistics/features/{type}/{class_name}", type = type)
            else:
                self._plot_distribution(rows, filename = f"statistics/features/{type}/{class_name}", type = type)
            self.logger.log("Plotted signature for: " + class_name)
            
        self.logger.log("Computed statistics for " + type + " feature")
        
    def _plot_signature(self, data, filename = "furniture", type = "A3"):
        upper_bounds = {
            "A3": math.pi,
            "D1": 1,
            "D2": math.sqrt(3),
            "D3": (math.sqrt(3) / 2) ** (1/2),
            "D4": (1 / 3) ** (1/3)
        }

        upper_bound_x = upper_bounds[type]
        
        plt.figure(figsize=(10, 10))
        plt.clf()
        plt.title(f"Signature for {filename} for feature {type}")
        plt.xlim(0, upper_bound_x)
        
        filename = filename + ".png"
        for row in data:
            x = np.linspace(0, upper_bound_x, len(row))
            plt.plot(x, row)
            
        plt.savefig(filename)
        plt.close()
        
    def _plot_distribution(self, data, filename = "furniture", type = "volume"):
        plt.figure(figsize=(10, 10))
        plt.clf()
        plt.title(f"Distribution for {filename} for feature {type}")
        plt.hist(data, bins=20)
        filename = filename + ".png"
        plt.savefig(filename)
        plt.close()
        
    def __del__ (self):
        # closing the db connection
        self.db.close()