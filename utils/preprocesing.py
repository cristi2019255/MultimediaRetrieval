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

from utils.tools import *
from utils.db_tools import *
from utils.statistics import *

class Prepocesor:
    def __init__(self):
        self.db = Database()
    
    def preprocess(self):
        # plotting histograms before resampling
        plot_histogram(self.db.get_column_data(by = "vertices_count"), by = "vertices_count")
        plot_histogram(self.db.get_column_data(by = "faces_count"), by = "faces__count")
        plot_histogram(self.db.get_column_data(by = "class"), by = "class")
                 
        # getting statistics about the database and resampling the outliers
        [_, _, _, _, avg_vertices_count, _, _, _, _, _, _, ] = self.db.get_average_shape(by = "vertices_count")
        [_, _, _, avg_faces_count, _, _, _, _, _, _, _, ] = self.db.get_average_shape(by = "faces_count")
    
        # resampling the outliers
        self.resample_outliers(avg=avg_vertices_count, std=500, by = "vertices_count")
        self.resample_outliers(avg=avg_faces_count, std=500, by = "faces_count")
    
        # plotting histograms after resampling
        plot_histogram(self.db.get_column_data(by = "vertices_count"), by = "vertices_count")
        plot_histogram(self.db.get_column_data(by = "faces_count"), by = "faces__count")
        
        # Closing the connection with db
        self.db.close()
    
    
    def resample_outliers(self, avg = 1000, std = 500, by = "vertices_count"):
         
        def resample(data, resample_function):
            try:
                for row in data:
                    filename = row[0]
                    resample_function(filename, avg)
                    self.db.update_data(filename)
            except Exception as e:
                print(e)
        
        # query to get the outliers
        
        sql_sub_samples = '''SELECT file_name FROM shapes WHERE {0} < {1};'''.format(by, (avg - std))
        self.db.cursor.execute(sql_sub_samples)
        rows_sub_sample = self.db.cursor.fetchall()
        resample(rows_sub_sample, sub_sample)
    
        sql_super_samples = '''SELECT file_name FROM shapes WHERE {0} > {1};'''.format(by, (avg + std))
        self.db.cursor.execute(sql_super_samples)
        rows_super_sample = self.db.cursor.fetchall()
        
        resample(rows_super_sample, super_sample)
       


def super_sample(filename, to = 1000):
    print(filename)
    ms = MeshSet()
    ms.load_new_mesh(filename)
    mesh = ms.current_mesh()
    # TODO: figure out a way to resample the mesh to a specific number of vertices)
    #mesh.save(filename)

def sub_sample(filename, to = 1000):
    print(filename)
    