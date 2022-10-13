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
from utils.Database import Database
from utils.Logger import Logger
from Shape import Shape
import numpy as np
from scipy.stats import wasserstein_distance

class QueryHandler:
    def __init__(self, log: bool = True):
        self.logger = Logger(active = log)
        self.db = Database(log=log)
    
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
        
    def find_similar_shapes(self, n = None, distance_measure = None, normalization = None):

        # Validate input parameters
        if distance_measure not in ["Euclidean Distance", "Cosine Distance", "Earth Mover''s Distance"]:
            distance_measure = "Euclidean Distance"

        if normalization not in ["Minimum Maximum Normalization", "Standard Score Normalization"]:
            normalization = "Minimum Maximum Normalization"

        # Query for finding similar shapes
        query = """
            with aggregated_features as
            (
                select min("surface_area")      as surface_area_min
                     , max("surface_area")      as surface_area_max
                     , avg("surface_area")      as surface_area_avg
                     , stddev("surface_area")   as surface_area_stddev
                     , min("compactness")       as compactness_min
                     , max("compactness")       as compactness_max
                     , avg("compactness")       as compactness_avg
                     , stddev("compactness")    as compactness_stddev
                     , min("volume")       as volume_min
                     , max("volume")       as volume_max
                     , avg("volume")       as volume_avg
                     , stddev("volume")    as volume_stddev
                     , min("diameter")          as diameter_min
                     , max("diameter")          as diameter_max
                     , avg("diameter")          as diameter_avg
                     , stddev("diameter")       as diameter_stddev
                     , min("eccentricity")      as eccentricity_min
                     , max("eccentricity")      as eccentricity_max
                     , avg("eccentricity")      as eccentricity_avg
                     , stddev("eccentricity")   as eccentricity_stddev

                from "features"  
            )

            , normalized_features as
            (
                select "id"
                     , "shape_id"
                     , 'Minimum Maximum Normalization' as "normalization_type"
                     , ("surface_area" - "surface_area_min") / ("surface_area_max" - "surface_area_min") as "surface_area_normalized"
                     , ("compactness"  - "compactness_min" ) / ("compactness_max"  - "compactness_min" ) as "compactness_normalized"
                     , ("volume"  - "volume_min" ) / ("volume_max"  - "volume_min" ) as "volume_normalized"
                     , ("diameter"     - "diameter_min"    ) / ("diameter_max"     - "diameter_min"    ) as "diameter_normalized"
                     , ("eccentricity" - "eccentricity_min") / ("eccentricity_max" - "eccentricity_min") as "eccentricity_normalized"

                from "features" 

                join aggregated_features 
                    on 1 = 1

                union all

                select "id"
                     , "shape_id"
                     , 'Standard Score Normalization' as "normalization_type"
                     , ("surface_area" - "surface_area_avg") / "surface_area_stddev" as "surface_area_normalized"
                     , ("compactness"  - "compactness_avg" ) / "compactness_stddev"  as "compactness_normalized"
                     , ("volume"  - "volume_avg" ) / "volume_stddev"  as "volume_normalized"
                     , ("diameter"     - "diameter_avg"    ) / "diameter_stddev"     as "diameter_normalized"
                     , ("eccentricity" - "eccentricity_avg") / "eccentricity_stddev" as "eccentricity_normalized"

                from "features" 

                join aggregated_features 
                    on 1 = 1
            )

            select features."shape_id"
                 , shapes."file_name"
                 , case '""" + distance_measure + """'
                       when 'Euclidean Distance'
                       then sqrt( pow(features."surface_area_normalized" - target."surface_area_normalized", 2)
                                + pow(features."compactness_normalized"  - target."compactness_normalized" , 2)
                                + pow(features."volume_normalized"  - target."volume_normalized" , 2)
                                + pow(features."diameter_normalized"     - target."diameter_normalized"    , 2)
                                + pow(features."eccentricity_normalized" - target."eccentricity_normalized", 2)
                            )   
                       when 'Cosine Distance' 
                       then /* 1 - similarity to get a 'distance', now higher similarities will give higher values */
                            1 -
                            /* Dot Product */
                            ( features."surface_area_normalized" * target."surface_area_normalized"
                            + features."compactness_normalized"  * target."compactness_normalized" 
                            + features."volume_normalized"  * target."volume_normalized" 
                            + features."diameter_normalized"     * target."diameter_normalized"    
                            + features."eccentricity_normalized" * target."eccentricity_normalized"
                            )
                            /
                            /* Product of Magnitudes */
                            (
                                sqrt( features."surface_area_normalized" * features."surface_area_normalized"
                                    + features."compactness_normalized"  * features."compactness_normalized" 
                                    + features."volume_normalized"  * features."volume_normalized" 
                                    + features."diameter_normalized"     * features."diameter_normalized"    
                                    + features."eccentricity_normalized" * features."eccentricity_normalized"
                                )
                                *
                                sqrt( target."surface_area_normalized" * target."surface_area_normalized"
                                    + target."compactness_normalized"  * target."compactness_normalized" 
                                    + target."volume_normalized"  * target."volume_normalized" 
                                    + target."diameter_normalized"     * target."diameter_normalized"    
                                    + target."eccentricity_normalized" * target."eccentricity_normalized"
                                )
                            )
                       when 'Earth''s Mover Distance'
                       then null
                   end as "distance"

            from "normalized_features" features

            join "shapes" shapes
                on shapes."id" = features."shape_id"

            join "normalized_features" target
                on target."normalization_type" = features."normalization_type"
                and target."shape_id" <> features."shape_id"
                and target."shape_id" = """ + str(self.shape_id) + """

            where features."normalization_type" = '""" + normalization + """'

            order by 3 asc

            """ + ("" if n is None else ("limit " + str(n) + ""))

        self.db.execute_query(query, "select")
        return self.db.cursor.fetchall()
    
    
    def find_similar_shapes_v1(self, filename, target_nr_shape_to_return = None):
        """
            Find similar shapes to the target shape using the specified distance measure and normalization method.
            TODO: I would like to compute a distance matrix beforehand for all the shapes in the db and store it somewhere and then fetch just based on that matrix.
            e.g.: 5 shapes that on the row of the target shape gives smallest distance.
            
            TODO: check problems with nan from D1
            normalize scalars to [0,1] range
            
        """
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
            
            target_scalars = features[1:9]
            
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
                current_scalars = row[1:9]
                
                # TODO: Delete this check, we have some nan in D1 for some reasons
                if math.isnan(D1[0]):
                    continue
                
                scalars_distance = self._cosine_distance(target_scalars, current_scalars)
                
                
                #self.logger.debug(f""" D1: {D1}""")
                #self.logger.debug(f""" Current shape id: {current_shape_id}""")
                #self.logger.debug(f""" Current feature id: {row[0]}""")
                
                distances_A3.append(self._earth_moving_distance(target_A3, A3))
                distances_D1.append(self._earth_moving_distance(target_D1, D1))
                distances_D2.append(self._earth_moving_distance(target_D2, D2))
                distances_D3.append(self._earth_moving_distance(target_D3, D3))
                distances_D4.append(self._earth_moving_distance(target_D4, D4))
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
                total_distance = distances_A3[i] + distances_D1[i] + distances_D2[i] + distances_D3[i] + distances_D4[i] + 1/5 * distances_scalars[i]
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
    def _cosine_distance(x = None, y = None):
        """Cosine Distance"""
        return 1 - (np.dot(x,y)) / (np.sqrt(np.dot(x,x)) * np.sqrt(np.dot(y,y)))
    
    @staticmethod
    def _lp_distance(x=None, y = None, p=2):
        """Lp Distance: default is Euclidean Distance"""
        return np.sum(np.abs(np.array(x) - np.array(y))**p)**(1/p)