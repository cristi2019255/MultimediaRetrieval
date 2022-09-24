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

from Shape import Shape
from utils.db_tools import Database


class FeatureExtractor:
    def __init__(self, log=False):
        self.log = log
        self.db = Database(log=self.log)
        self.db.create_features_table()
        
    def extract_features(self):
        shapes = self.db.get_table_data(table='shapes', columns=['id', 'file_name'])
        for shape_data in shapes:
            [id, file_name] = shape_data
            shape = Shape(file_name=file_name)
            feature_id  = self.db.insert_features_data(shape, shape_id = id)
            self.db.update_shape_feature_id(id, feature_id = feature_id)
            if self.log:
                print("[INFO] Extracted features for shape with id: " + str(id))

        # closing the db connection
        self.db.close()
    