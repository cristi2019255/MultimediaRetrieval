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
import numpy as np

NR_DESIRED_FACES = 5000
ORIGIN = np.array([0, 0, 0])
X_AXIS = np.array([1, 0, 0])
Y_AXIS = np.array([0, 1, 0])
Z_AXIS = np.array([0, 0, 1])


class Preprocessor:
    def __init__(self, log=False):
        self.logger = Logger(active=log)
        self.db = Database(log=log)

    def preprocess(self):
        # getting statistics about the database and resampling the outliers

        # avg_faces_count = self.db.get_average(by="faces_count", table="shapes")

        avg_faces_count = NR_DESIRED_FACES  # hardcoded because of the time it takes to resample

        # plotting histograms before resampling

        # for checking resampling
        plot_histogram(self.db.get_column_data(by="vertices_count"),
                       title="Number of vertices before resampling")

        plot_histogram(self.db.get_column_data(by="faces_count"),
                       title="Number of faces before resampling")

        plot_histogram(self.db.get_column_data(by="class"),
                       title="Distribution of shape classes", ticks=True)  # bins should be dynamic

        # for checking normalization
        plot_histogram(self.db.get_column_data(by="bounding_box_diagonal"),
                       title="Length of bounding box diagonal before normalization")

        # resampling the outliers
        statistics_before_normalization, statistics_after_normalization = self.resample_outliers_and_normalize(target_faces_nr=avg_faces_count)

        [barycenters, diff_x, diff_y, diff_z, eigenvectors_x, eigenvectors_y, eigenvectors_z] = statistics_before_normalization

        barycenters_dist = []
        for barycenter in barycenters:
            barycenters_dist.append(np.linalg.norm(barycenter - ORIGIN))

        # plotting distribution of distance from barycenters to origin before normalization
        plot_histogram(barycenters_dist,
                       title="Distance from barycenters to origin before normalization")

        # plotting histograms before normalization to check eigenvectors alignment
        axes_eigen = [[eigenvectors_x, X_AXIS], [eigenvectors_y, Y_AXIS], [eigenvectors_z, Z_AXIS]]
        for item in axes_eigen:
            [eigenvectors, axis] = item
            dot_products = []
            for eigenvector in eigenvectors:
                dot_products.append(abs(np.dot(eigenvector, axis)))
            plot_histogram(np.average(dot_products),
                           title="Average of dot products between all eigenvectors and corresponding axes before normalization")

        # plotting distribution of differences between the amount of points on positive and negative part of axis before normalization
        plot_histogram(diff_x, title="Histogram of diff x-coord before normalization")
        plot_histogram(diff_y, title="Histogram of diff y-coord before normalization")
        plot_histogram(diff_z, title="Histogram of diff z-coord before normalization")

        # plotting histograms after resampling
        plot_histogram(self.db.get_column_data(by="vertices_count"),
                       title="Number of vertices after resampling")
        plot_histogram(self.db.get_column_data(by="faces_count"),
                       title="Number of faces after resampling")

        plot_histogram(self.db.get_column_data(by="bounding_box_diagonal"),
                       title="Length of bounding box diagonal after normalization")

        [barycenters, diff_x, diff_y, diff_z, eigenvectors_x, eigenvectors_y,
         eigenvectors_z] = statistics_after_normalization

        barycenters_dist = []
        for barycenter in barycenters:
            barycenters_dist.append(np.linalg.norm(barycenter - np.array([0, 0, 0])))

        # plotting distribution of distance from barycenters to origin after normalization    
        plot_histogram(barycenters_dist,
                       title="Distance from barycenters to origin after normalization")

        # plotting histograms after normalization to check eigenvectors alignment
        axes_eigen = [[eigenvectors_x, X_AXIS], [eigenvectors_y, Y_AXIS], [eigenvectors_z, Z_AXIS]]
        for item in axes_eigen:
            [eigenvectors, axis] = item
            dot_products = []
            for eigenvector in eigenvectors:
                dot_products.append(abs(np.dot(eigenvector, axis)))
            plot_histogram(np.average(dot_products),
                           title="Average of dot products between all eigenvectors and corresponding axes after normalization")

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
        eigenvector_x_before_normalization = []
        eigenvector_y_before_normalization = []
        eigenvector_z_before_normalization = []

        # shapes data after normalization
        barycenters_after_normalization = []
        diff_axis_x_after_normalization = []
        diff_axis_y_after_normalization = []
        diff_axis_z_after_normalization = []
        eigenvector_x_after_normalization = []
        eigenvector_y_after_normalization = []
        eigenvector_z_after_normalization = []

        
        errors = 0
        for row in rows:
                filename = row[0]
                
                shape = Shape(filename, log=False)

                try:
                    self.logger.log("Resampling shape: " + filename)
                    shape.resample(target_faces=target_faces_nr)            
                    
                    # getting shapes data before normalization
                    barycenters_before_normalization.append(shape.get_barycenter())

                    diff_axis_x_before_normalization.append(shape.get_nr_of_vertices_diff_on_positive_axis(axis=0))
                    diff_axis_y_before_normalization.append(shape.get_nr_of_vertices_diff_on_positive_axis(axis=1))
                    diff_axis_z_before_normalization.append(shape.get_nr_of_vertices_diff_on_positive_axis(axis=2))

                    _, [x, y, z] = shape.principal_component_analysis()
                    eigenvector_x_before_normalization.append(x)
                    eigenvector_y_before_normalization.append(y)
                    eigenvector_z_before_normalization.append(z)

                    # normalizing the shape
                    shape.normalize()

                    # getting shapes data after normalization
                    barycenters_after_normalization.append(shape.get_barycenter())

                    diff_axis_x_after_normalization.append(shape.get_nr_of_vertices_diff_on_positive_axis(axis=0))
                    diff_axis_y_after_normalization.append(shape.get_nr_of_vertices_diff_on_positive_axis(axis=1))
                    diff_axis_z_after_normalization.append(shape.get_nr_of_vertices_diff_on_positive_axis(axis=2))

                    _, [x, y, z] = shape.principal_component_analysis()
                    eigenvector_x_after_normalization.append(x)
                    eigenvector_y_after_normalization.append(y)
                    eigenvector_z_after_normalization.append(z)

                    original_file_name = shape.file_name
                    shape.file_name = shape.file_name.replace("data", "preprocessed")
                    shape.save_mesh(shape.file_name)

                    self.db.update_shape_data(shape, original_file_name)
                    
                except Exception as e:
                    self.logger.error("Error while resampling and normalizing shapes: " + str(e))
                    errors += 1
                
                # deleting shape from memory
                del shape
        
        self.logger.warn("Number of errors: " + str(errors))
        
        return ([barycenters_before_normalization,
                 diff_axis_x_before_normalization,
                 diff_axis_y_before_normalization,
                 diff_axis_z_before_normalization,
                 eigenvector_x_before_normalization,
                 eigenvector_y_before_normalization,
                 eigenvector_z_before_normalization,
                 ],

                [barycenters_after_normalization,
                 diff_axis_x_after_normalization,
                 diff_axis_y_after_normalization,
                 diff_axis_z_after_normalization,
                 eigenvector_x_after_normalization,
                 eigenvector_y_after_normalization,
                 eigenvector_z_after_normalization,
                 ])
