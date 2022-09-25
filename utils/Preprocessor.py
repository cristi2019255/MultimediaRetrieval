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
from utils.Logger import Logger
from utils.tools import *
from utils.Database import Database
from utils.statistics import *

NR_DESIRED_FACES = 21758

class Preprocessor:
    def __init__(self, log = False):
        self.logger = Logger(active=log)
        self.db = Database(log = log)

    def preprocess(self):
        # getting statistics about the database and resampling the outliers
        avg_faces_count = self.db.get_average(by="faces_count", table="shapes")

        avg_faces_count = NR_DESIRED_FACES # hardcoded because of the time it takes to resample
        
             
        # plotting histograms before resampling
        
        # for checking resampling
        plot_histogram(self.db.get_column_data(by="vertices_count"), title="Histogram of vertex counts")
        plot_histogram(self.db.get_column_data(by="faces_count"), title="Histogram of faces counts")
        
        plot_histogram(self.db.get_column_data(by="class"), title="Histogram of shape classes")
        
        # for checking normalization
        plot_histogram(self.db.get_column_data(by="bounding_box_dim_x"), title="Histogram of bounding box dim_x")
        plot_histogram(self.db.get_column_data(by="bounding_box_dim_y"), title="Histogram of bounding box dim_y")
        plot_histogram(self.db.get_column_data(by="bounding_box_dim_z"), title="Histogram of bounding box dim_z")
        
         
        # resampling the outliers
        statistics_before_normalization, statistics_after_normalization = self.resample_outliers_and_normalize(target_faces_nr=avg_faces_count)
        
        [barycenters, diff_x, diff_y, diff_z] = statistics_before_normalization
        barycenters_x, barycenters_y, barycenters_z = [], [], []
        for barycenter in barycenters:
            barycenters_x.append(barycenter[0])
            barycenters_y.append(barycenter[1])
            barycenters_z.append(barycenter[2])
        
        plot_histogram(barycenters_x, title="Histogram of barycenters x-coord before normalization")
        plot_histogram(barycenters_y, title="Histogram of barycenters y-coord before normalization")
        plot_histogram(barycenters_z, title="Histogram of barycenters z-coord before normalization")
        plot_histogram(diff_x, title="Histogram of diff x-coord before normalization")
        plot_histogram(diff_y, title="Histogram of diff y-coord before normalization")
        plot_histogram(diff_z, title="Histogram of diff z-coord before normalization")
         
        # plotting histograms after resampling
        plot_histogram(self.db.get_column_data(by="vertices_count"), title="Histogram of vertex counts after resampling")
        plot_histogram(self.db.get_column_data(by="faces_count"), title="Histogram of faces counts after resampling")
        plot_histogram(self.db.get_column_data(by="bounding_box_dim_x"), title="Histogram of bounding box dim_x after resampling")
        plot_histogram(self.db.get_column_data(by="bounding_box_dim_y"), title="Histogram of bounding box dim_y after resampling")
        plot_histogram(self.db.get_column_data(by="bounding_box_dim_z"), title="Histogram of bounding box dim_z after resampling")

        [barycenters, diff_x, diff_y, diff_z] = statistics_after_normalization
        
        barycenters_x, barycenters_y, barycenters_z = [], [], []
        for barycenter in barycenters:
            barycenters_x.append(barycenter[0])
            barycenters_y.append(barycenter[1])
            barycenters_z.append(barycenter[2])
        
        plot_histogram(barycenters_x, title="Histogram of barycenters x-coord after normalization")
        plot_histogram(barycenters_y, title="Histogram of barycenters y-coord after normalization")
        plot_histogram(barycenters_z, title="Histogram of barycenters z-coord after normalization")
        plot_histogram(diff_x, title="Histogram of diff x-coord after normalization")
        plot_histogram(diff_y, title="Histogram of diff y-coord after normalization")
        plot_histogram(diff_z, title="Histogram of diff z-coord after normalization")
        

        # Closing the connection with db
        self.db.close()
        
        self.logger.success("Preprocessing completed!!!")
    
    def resample_outliers_and_normalize(self, target_faces_nr=NR_DESIRED_FACES):

        os.makedirs("preprocessed", exist_ok=True)
        
        # query to get all the shapes to be resampled
        self.db.cursor.execute('''SELECT file_name FROM shapes''')
        rows = self.db.cursor.fetchall()
        
        # shapes data before normalization
        barycenters_before_normalization = []
        diff_axis_x_before_normalization = []
        diff_axis_y_before_normalization = []
        diff_axis_z_before_normalization = []
        
        # shapes data after normalization
        barycenters_after_normalization = []
        diff_axis_x_after_normalization = []
        diff_axis_y_after_normalization = []
        diff_axis_z_after_normalization = []
            
        try:
            for row in rows:
                filename = row[0]
                self.logger.log("Resampling shape: " + filename)    
                
                shape = Shape(filename, log = False)
                
                shape.resample(target_faces=target_faces_nr)
                
                # getting shapes data before normalization
                barycenters_before_normalization.append(shape.get_barycenter())
                diff_axis_x_before_normalization.append(shape.get_nr_of_vertices_diff_on_positive_axis(axis=0))
                diff_axis_y_before_normalization.append(shape.get_nr_of_vertices_diff_on_positive_axis(axis=1))
                diff_axis_z_before_normalization.append(shape.get_nr_of_vertices_diff_on_positive_axis(axis=2))
                
                shape.normalize()
    
                # getting shapes data after normalization
                barycenters_after_normalization.append(shape.get_barycenter())
                diff_axis_x_after_normalization.append(shape.get_nr_of_vertices_diff_on_positive_axis(axis=0))
                diff_axis_y_after_normalization.append(shape.get_nr_of_vertices_diff_on_positive_axis(axis=1))
                diff_axis_z_after_normalization.append(shape.get_nr_of_vertices_diff_on_positive_axis(axis=2))
                                
                original_file_name = shape.file_name
                shape.file_name = shape.file_name.replace("./","./preprocessed/")
                shape.save_mesh()
                    
                self.db.update_shape_data(shape, original_file_name)
        except Exception as e:
            self.logger.error("Error while resampling and normalizing shapes: " + str(e))

        return ([barycenters_before_normalization, diff_axis_x_before_normalization, diff_axis_y_before_normalization, diff_axis_z_before_normalization],
               [barycenters_after_normalization, diff_axis_x_after_normalization, diff_axis_y_after_normalization, diff_axis_z_after_normalization])