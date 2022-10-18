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

from utils.Database import Database
from utils.Logger import Logger
from Shape import Shape
import numpy as np
from scipy.stats import wasserstein_distance

class QueryHandler:
    def __init__(self, log: bool = True):
        self.logger = Logger(active = log)
        self.db = Database(log=log)
        self.normalization_factors = self._get_normalization_factors()
    
    def _get_normalization_factors(self):
        sql = """ SELECT      
                       min("surface_area")      as surface_area_min
                     , max("surface_area")      as surface_area_max
                     , avg("surface_area")      as surface_area_avg
                     , stddev("surface_area")   as surface_area_stddev
                     
                     , min("compactness")       as compactness_min
                     , max("compactness")       as compactness_max
                     , avg("compactness")       as compactness_avg
                     , stddev("compactness")    as compactness_stddev
                     
                     , min(ratio_bbox_volume) as ratio_bbox_volume_min
                     , max(ratio_bbox_volume) as ratio_bbox_volume_max
                     , avg(ratio_bbox_volume) as ratio_bbox_volume_avg
                     , stddev(ratio_bbox_volume) as ratio_bbox_volume_stddev                     
                     
                     , min("volume")       as volume_min
                     , max("volume")       as volume_max
                     , avg("volume")       as volume_avg
                     , stddev("volume")    as volume_stddev
                    
                     , min(ratio_ch_volume) as ratio_ch_volume_min
                     , max(ratio_ch_volume) as ratio_ch_volume_max
                     , avg(ratio_ch_volume) as ratio_ch_volume_avg
                     , stddev(ratio_ch_volume) as ratio_ch_volume_stddev

                     , min(ratio_ch_area) as ratio_ch_area_min
                     , max(ratio_ch_area) as ratio_ch_area_max
                     , avg(ratio_ch_area) as ratio_ch_area_avg
                     , stddev(ratio_ch_area) as ratio_ch_area_stddev


                     , min("diameter")          as diameter_min
                     , max("diameter")          as diameter_max
                     , avg("diameter")          as diameter_avg
                     , stddev("diameter")       as diameter_stddev
                     
                     , min("eccentricity")      as eccentricity_min
                     , max("eccentricity")      as eccentricity_max
                     , avg("eccentricity")      as eccentricity_avg
                     , stddev("eccentricity")   as eccentricity_stddev
                FROM features
        """
        self.db.execute_query(sql, "select")
        row = list(self.db.cursor.fetchone())
        normalization_factors = []
        
        for i in range(0, len(row), 4):
            normalization_factors.append(row[i:i+4])
        
        return normalization_factors
    
    def normalize_features(self, features: list, normalization_factors: list, normalization_type: str = "minmax"):
        normalized_features = []
        if normalization_type == "minmax":
            for i in range(len(features)):
                normalized_features.append((features[i] - normalization_factors[i][0]) / (normalization_factors[i][1] - normalization_factors[i][0]))
        else:
            for i in range(len(features)):
                normalized_features.append((features[i] - normalization_factors[i][2]) / normalization_factors[i][3])
        
        return normalized_features
    
    def fetch_shape(self, filename: str):
        self.logger.log(f"Running query on {filename}")
        
        self.shape = Shape(filename, log = self.logger.active)

        self.logger.log(f"Shape {filename} has {self.shape.get_features()} features")
        
        # Fetch shape id
        self.db.execute_query(f"select \"id\" from \"shapes\" where \"file_name\" = '{self.shape.file_name}'", "select")
        
        shape_ids = self.db.cursor.fetchone()
        if shape_ids == [] or shape_ids == None:
            raise Exception(f"No shape found in database with file name '{self.shape.file_name}'!")
        self.shape_id = shape_ids[0]

        self.logger.log(f"Shape {filename} has shape id: {self.shape_id}")
        
        return self.shape_id
    
    def fetch_features(self, filename: str):
        shape_id = self.fetch_shape(filename)
        sql = f"select * from \"features\" where \"shape_id\" = {shape_id}"
        self.db.execute_query(sql, "select")
        row = self.db.cursor.fetchone()
        return row
    
    def find_similar_shapes(self, filename,
                            target_nr_shape_to_return = None, 
                            distance_measure_scalars = 'L2', 
                            distance_measure_histograms = 'Earth Mover Distance', 
                            normalization_type = 'minmax',
                            global_weights = [0.5, 0.5],
                            scalar_weights = [1]
                            ):
        """
            Find similar shapes to the target shape using the specified distance measure and normalization method.
            TODO: I would like to compute a distance matrix beforehand for all the shapes in the db and store it somewhere and then fetch just based on that matrix.
            e.g.: 5 shapes that on the row of the target shape gives smallest distance.
            
            TODO: check problems with nan from D1
            normalize scalars to [0,1] range
            
        """
        
        # ------------------- Choosing the distance function ------------------------ 
        scalars_distances = {
            "L1": self.get_lp_distance(1),
            "L2": self.get_lp_distance(2),
            "Linf": self.get_lp_distance("inf"),
            "Cosine": self._cosine_distance,
        }
        
        histograms_distances = {
            "Earth Mover Distance": self._earth_moving_distance,
            "other": self._earth_moving_distance,
        }
        
        scalars_distance_measure = scalars_distances[distance_measure_scalars]
        histograms_distance_measure = histograms_distances[distance_measure_histograms]
        # ------------------------------------------------------------------------------
        
        
        # ------------- Normalizing the weight vectors --------------------------------
        if len(global_weights) == 2:
            global_weights = [global_weights[0]] + [global_weights[1]] * 5
        assert(len(global_weights) == 6)
        
        global_weights_sum = sum(global_weights)
        global_weights = [w / global_weights_sum for w in global_weights]
        assert(sum(global_weights) == 1)
        
        if (len(scalar_weights) == 1):
            scalar_weights = [scalar_weights[0]] * 8
        assert(len(scalar_weights) == 8)   
    
        scalar_weights_sum = sum(scalar_weights)
        scalar_weights = [w / scalar_weights_sum for w in scalar_weights]
        assert(sum(scalar_weights) == 1)
         
        # ---------------------------------------------------------------------------------    
            
        
        # ---------------------- Fetching the shape from the database ---------------------    
        try:
            shape_id = self.fetch_shape(filename=filename)
            
            # getting the features of the target shape
            sql = f""" SELECT * FROM features WHERE shape_id = {shape_id} """
            self.db.execute_query(sql, "select")
            features = self.db.cursor.fetchone()
            
            [target_A3, 
             target_D1, 
             target_D2, 
             target_D3, 
             target_D4] = features[9:-2]            
            
            target_scalars = self.normalize_features(features[1:9], normalization_factors=self.normalization_factors, normalization_type=normalization_type)
            
            # getting the features of all the shapes in the db except the target shape
            sql = f""" SELECT * FROM features WHERE shape_id <> {shape_id} """
            self.db.execute_query(sql, "select")
            rows = self.db.cursor.fetchall()
            
            distances_A3 = []
            distances_D1 = []
            distances_D2 = []
            distances_D3 = []
            distances_D4 = []
            distances_scalars = []
            shape_ids = []
            
            for row in rows:
                current_shape_id = row[-2]
                
                [A3, D1, D2, D3, D4] = row[9:-2]
                current_scalars = self.normalize_features(row[1:9], normalization_factors=self.normalization_factors, normalization_type=normalization_type)
                
                scalars_distance = scalars_distance_measure(target_scalars, current_scalars, scalar_weights)
                
                distances_A3.append(histograms_distance_measure(target_A3, A3))
                distances_D1.append(histograms_distance_measure(target_D1, D1))
                distances_D2.append(histograms_distance_measure(target_D2, D2))
                distances_D3.append(histograms_distance_measure(target_D3, D3))
                distances_D4.append(histograms_distance_measure(target_D4, D4))
                distances_scalars.append(scalars_distance)
                shape_ids.append(current_shape_id)
                
            # normalizing
            distances_A3 = np.array(distances_A3) / np.sum(distances_A3)
            distances_D1 = np.array(distances_D1) / np.sum(distances_D1)
            distances_D2 = np.array(distances_D2) / np.sum(distances_D2)
            distances_D3 = np.array(distances_D3) / np.sum(distances_D3)
            distances_D4 = np.array(distances_D4) / np.sum(distances_D4)
            distances_scalars = np.array(distances_scalars) / np.sum(distances_scalars)
            
            distances = []
            for i in range(len(distances_A3)):
                feature_vector = np.array([distances_scalars[i], distances_A3[i], distances_D1[i], distances_D2[i], distances_D3[i], distances_D4[i] ])
                total_distance = np.dot(global_weights, feature_vector) 
                distances.append((shape_ids[i] , total_distance))
            
            distances.sort(key=lambda x: x[1])
            
            distances = distances[:target_nr_shape_to_return]
            
            result = []
            for (shape_id, distance) in distances:
                sql = f""" SELECT file_name FROM shapes WHERE id = {shape_id} """
                self.db.execute_query(sql, "select")
                row = self.db.cursor.fetchone()
                result.append((row[0], distance))
            
            return result        
        except Exception as e:
            raise Exception("Shape not found in database.")
        
        
        
    @staticmethod
    def _earth_moving_distance(A = None, B = None):
        """Earth's Mover's Distance"""
        assert(len(A) == len(B))
        u_values = v_values = np.arange(len(A))
        return wasserstein_distance(u_values = u_values, v_values = v_values, u_weights = A, v_weights = B)
    
    
    @staticmethod
    def _cosine_distance(x = None, y = None, w = None):
        """Cosine Distance"""
        return 1 - (dot(x,y,w)) / (np.sqrt(dot(x, x, w)) * np.sqrt(dot(y, y, w)))
    
    @staticmethod
    def get_lp_distance(p = 2):
        if p == "inf":
            return lambda x,y, w: max(np.abs(np.array(x)-np.array(y)))
        def _lp_distance(x=None, y = None, w = None, p=2):
            """Lp Distance: default is Euclidean Distance"""
            return np.sum((np.array(w) * np.abs(np.array(x) - np.array(y)))**p)**(1/p)
        return _lp_distance
    
    
def dot(x,y,w):
        assert(len(y) == len(x))
        assert(len(y) == len(w))
        sum = 0
        for i in range(len(x)):
            sum += w[i] * x[i] * y[i]
        return sum 