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

NR_DESIRED_FACES = 21758


class Prepocesor:
    def __init__(self, log = False):
        self.log = log
        self.db = Database(log = self.log)

    def preprocess(self):
        # plotting histograms before resampling
        plot_histogram(self.db.get_column_data(by="vertices_count"), title="Histogram of vertex counts")
        plot_histogram(self.db.get_column_data(by="faces_count"), title="Histogram of faces counts")
        plot_histogram(self.db.get_column_data(by="class"), title="Histogram of shape classes")

        # getting statistics about the database and resampling the outliers
        #[_, _, _, _, avg_vertices_count, _, _, _, _, _, _, ] = self.db.get_average_shape(by="vertices_count")
        
        #[_, _, _, avg_faces_count, _, _, _, _, _, _, _, ] = self.db.get_average_shape(by="faces_count")

        # resampling the outliers
        
        avg_faces_count = NR_DESIRED_FACES # hardcoded because of the time it takes to resample
         
        self.resample_outliers_and_normalize(target_faces_nr=avg_faces_count)
        
        # plotting histograms after resampling
        plot_histogram(self.db.get_column_data(by="vertices_count"), title="Histogram of vertex counts after resampling")
        plot_histogram(self.db.get_column_data(by="faces_count"), title="Histogram of faces counts after resampling")

        # Closing the connection with db
        self.db.close()
        
        if self.log:
            print("[Success] Preprocessing completed!!!")

    def resample_outliers_and_normalize(self, target_faces_nr=NR_DESIRED_FACES):

        # query to get all the shapes to be resampled
        self.db.cursor.execute('''SELECT file_name FROM shapes''')
        rows = self.db.cursor.fetchall()
            
        try:
            for row in rows:
                filename = row[0]
                if self.log:
                    print("[INFO] Resampling shape: ", filename)    
                
                shape = Shape(filename, log = False)
                
                # changing order of operations helps to faster computation
                
                # TODO: is this change of order of operations correct? i.e. does it affect the results?
                
                if shape.mesh.face_number() > target_faces_nr:
                    shape.resample(target_faces=target_faces_nr)
                    shape.normalize()
                else:
                    shape.normalize()
                    shape.resample(target_faces=target_faces_nr)
                    
                    
                original_file_name = shape.file_name
                shape.file_name = shape.file_name.replace("./","./preprocessed/")
                shape.save_mesh()
                    
                self.db.update_data(shape, original_file_name)
        except Exception as e:
            print(f"[Error] {e}")
