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

from utils.Shape import Shape
from utils.Logger import Logger
from utils.tools import *
from utils.Database import Database
from utils.statistics import *
import numpy as np
from tqdm import tqdm

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
        self.resample_outliers_and_normalize(target_faces_nr=NR_DESIRED_FACES)
        # Closing the connection with db
        self.db.close()
        
        self.logger.success("Preprocessing completed!!!")

    def resample_outliers_and_normalize(self, target_faces_nr=NR_DESIRED_FACES):

        os.makedirs("preprocessed", exist_ok=True)

        # query to get all the shapes to be resampled
        self.db.cursor.execute('''SELECT file_name FROM shapes''')
        rows = self.db.cursor.fetchall()

        
        errors = 0
        for row in tqdm(rows):
                filename = row[0]
                
                shape = Shape(filename, log=False)

                try:
                    self.logger.log("Resampling shape: " + filename)
                    shape.resample(target_faces=target_faces_nr)            
                    
                    # normalizing the shape
                    shape.normalize()

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


    def compute_class_distribution(self):
        files = scan_files("data")
        data = []
        for shape_class in files.keys():
            for file in files[shape_class]:
                data.append(shape_class)
        plot_histogram(data, title = "Distribution of shape classes" , bins = len(files.keys()))
    
    def compute_statistics(self, type = "before"):
        # plotting histograms before resampling and normalization
        self.logger.log("Computing statistics before resampling and normalization ... ")
        
        # shapes data before normalization
        barycenters_dists = []
        diff_axis_x = []
        diff_axis_y = []
        diff_axis_z = []
        eigenvectors_dot_prods_x = []
        eigenvectors_dot_prods_y = []
        eigenvectors_dot_prods_z = []
        faces_count = []
        vertices_count = []
        bbox_diagonals = []
        
        folder = "data" if type == "before" else "preprocessed"
        
        for r, d, f in os.walk(folder):
            self.logger.log("Computing statistics for folder: " + r)
            if  "test" in r:
                continue
            for file in f:    
                if ('.ply' in file):
                    shape = Shape(os.path.join(r, file), log=False)
                    faces_count.append(shape.faces_count)
                    vertices_count.append(shape.vertices_count)
                    
                    barycenters_dists.append(np.linalg.norm(shape.get_barycenter() - ORIGIN))
                    
                    diff_axis_x.append(shape.get_nr_of_vertices_diff_on_positive_axis(axis=0))            
                    diff_axis_y.append(shape.get_nr_of_vertices_diff_on_positive_axis(axis=1))
                    diff_axis_z.append(shape.get_nr_of_vertices_diff_on_positive_axis(axis=2))
                    
                    _, [x, y, z] = shape.principal_component_analysis()
                    
                    eigenvectors_dot_prods_x.append(abs(np.dot(x, X_AXIS)))
                    eigenvectors_dot_prods_y.append(abs(np.dot(y, Y_AXIS)))
                    eigenvectors_dot_prods_z.append(abs(np.dot(z, Z_AXIS)))
                    
                    bbox_diagonals.append(shape.bbox_diagonal)
        
        
        # saving computed statistics
        os.makedirs("statistics", exist_ok=True)
        with open(f"statistics/statistics_{type}.txt", "w") as f:
            f.write(f"Number of faces {type} resampling: {faces_count} \n\n")
            f.write(f"Number of vertices {type} resampling: {vertices_count} \n\n")
            f.write(f"Distance from barycenters to origin {type} normalization: {barycenters_dists} \n\n")
            f.write(f"Dot product between eigenvectors and x axis {type} normalization: {eigenvectors_dot_prods_x} \n\n")
            f.write(f"Dot product between eigenvectors and y axis {type} normalization: {eigenvectors_dot_prods_y} \n\n")
            f.write(f"Dot product between eigenvectors and z axis {type} normalization: {eigenvectors_dot_prods_z} \n\n")
            f.write(f"Histogram of diff x-coord {type} normalization: {diff_axis_x} \n\n")
            f.write(f"Histogram of diff y-coord {type} normalization: {diff_axis_y} \n\n")
            f.write(f"Histogram of diff z-coord {type} normalization: {diff_axis_z} \n\n")
            f.write(f"Length of bounding box diagonal {type} normalization: {bbox_diagonals} \n\n")
        
        # Plotting histograms
                    
        # for checking resampling
        plot_histogram(vertices_count, title=f"Number of vertices {type} resampling")
        plot_histogram(faces_count, title=f"Number of faces {type} resampling")

        # for checking normalization
        
        # plotting distribution of distance from barycenters to origin before normalization
        plot_histogram(barycenters_dists, title=f"Distance from barycenters to origin {type} normalization")

        # plotting histograms before normalization to check eigenvectors alignment
        plot_histogram(eigenvectors_dot_prods_x, title=f"Dot product between eigenvectors and x axis {type} normalization")
        plot_histogram(eigenvectors_dot_prods_y, title=f"Dot product between eigenvectors and y axis {type} normalization")
        plot_histogram(eigenvectors_dot_prods_z, title=f"Dot product between eigenvectors and z axis {type} normalization")

        # plotting distribution of differences between the amount of points on positive and negative part of axis 
        plot_histogram(diff_axis_x, title=f"Histogram of diff x-coord {type} normalization")
        plot_histogram(diff_axis_y, title=f"Histogram of diff y-coord {type} normalization")
        plot_histogram(diff_axis_z, title=f"Histogram of diff z-coord {type} normalization")

        # plotting distribution of bbox diagonals
        plot_histogram(bbox_diagonals,title=f"Length of bounding box diagonal {type} normalization")
        
        self.logger.success("Statistics computed!!!")