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
        
        shape_ids = self.db.cursor.fetchall()
        if shape_ids == []:
            raise Exception(f"No shape found in database with file name '{self.shape.file_name}'!")
        self.shape_id = shape_ids[0][0]

        self.logger.log(f"Shape {filename} has shape id: {self.shape_id}")
        
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
                     , min("bbox_volume")       as bbox_volume_min
                     , max("bbox_volume")       as bbox_volume_max
                     , avg("bbox_volume")       as bbox_volume_avg
                     , stddev("bbox_volume")    as bbox_volume_stddev
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
                     , ("bbox_volume"  - "bbox_volume_min" ) / ("bbox_volume_max"  - "bbox_volume_min" ) as "bbox_volume_normalized"
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
                     , ("bbox_volume"  - "bbox_volume_avg" ) / "bbox_volume_stddev"  as "bbox_volume_normalized"
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
                                + pow(features."bbox_volume_normalized"  - target."bbox_volume_normalized" , 2)
                                + pow(features."diameter_normalized"     - target."diameter_normalized"    , 2)
                                + pow(features."eccentricity_normalized" - target."eccentricity_normalized", 2)
                            )   
                       when 'Cosine Distance' 
                       then /* 1 - similarity to get a 'distance', now higher similarities will give higher values */
                            1 -
                            /* Dot Product */
                            ( features."surface_area_normalized" * target."surface_area_normalized"
                            + features."compactness_normalized"  * target."compactness_normalized" 
                            + features."bbox_volume_normalized"  * target."bbox_volume_normalized" 
                            + features."diameter_normalized"     * target."diameter_normalized"    
                            + features."eccentricity_normalized" * target."eccentricity_normalized"
                            )
                            /
                            /* Product of Magnitudes */
                            (
                                sqrt( features."surface_area_normalized" * features."surface_area_normalized"
                                    + features."compactness_normalized"  * features."compactness_normalized" 
                                    + features."bbox_volume_normalized"  * features."bbox_volume_normalized" 
                                    + features."diameter_normalized"     * features."diameter_normalized"    
                                    + features."eccentricity_normalized" * features."eccentricity_normalized"
                                )
                                *
                                sqrt( target."surface_area_normalized" * target."surface_area_normalized"
                                    + target."compactness_normalized"  * target."compactness_normalized" 
                                    + target."bbox_volume_normalized"  * target."bbox_volume_normalized" 
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