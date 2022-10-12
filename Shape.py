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

from utils.Logger import Logger
import math
from pymeshlab import MeshSet, Mesh
import numpy as np
from utils.renderer import render
import os
from utils.tools import get_features
from numba import jit

NR_DESIRED_FACES = 5000
NR_SAMPLES_FOR_FEATURE_DESCRIPTORS_D1 = 1500
NR_SAMPLES_FOR_FEATURE_DESCRIPTORS_D2 = 1000
NR_SAMPLES_FOR_FEATURE_DESCRIPTORS_D3 = NR_SAMPLES_FOR_FEATURE_DESCRIPTORS_A3 = 700
NR_SAMPLES_FOR_FEATURE_DESCRIPTORS_D4 = 500
 
FEATURE_DESCRIPTORS_DIMENSIONS_D1 = 21
FEATURE_DESCRIPTORS_DIMENSIONS_D2 = 23
FEATURE_DESCRIPTORS_DIMENSIONS_D3 = FEATURE_DESCRIPTORS_DIMENSIONS_A3 = 26
FEATURE_DESCRIPTORS_DIMENSIONS_D4 = 30
  

class Shape:
    # ------------------ Class Methods ------------------
    
    # ------------------ 1. Constructors ------------------
    def __init__(self, vertices, faces, log: bool = False):
        """_summary_ construct a shape from vertices and faces

        Args:
            vertices (numpy.ndarray): vertices matrix of the shape
            faces (numpy.ndarray): faces matrix of the shape
        """
        self.vertices = vertices
        self.faces = faces
        self.mesh = Mesh(self.vertices, self.faces)
        self.file_name = None
        self.ms = MeshSet()
        self.ms.add_mesh(self.mesh)
        self.logger = Logger(active=log)
        
    
    def __init__(self, file_name: str, log: bool = False):
        """_summary_ construct a shape from file_name

        Args:
            file_name (str): the file name of the shape
        """
        self.ms = MeshSet()
        self.ms.load_new_mesh(file_name)
        self.file_name = file_name
        self.mesh = self.ms.current_mesh()
        self.vertices = self.mesh.vertex_matrix()
        self.faces = self.mesh.face_matrix()
        self.faces_count = self.mesh.face_number()
        self.vertices_count = self.mesh.vertex_number()
        self.bbox_diagonal = self.mesh.bounding_box().diagonal()
        self.logger = Logger(active=log)
    
    # ----------------- 2. General shape I/O methods -----------------
    
    def save_mesh(self, file_name: str = None):
        """_summary_: save the mesh to a file

        Args:
            file_name (str, optional): Defaults to None.

        Raises:
            ValueError: No filename provided
        """
        if file_name is None:
            file_name = self.file_name
            if file_name is None:
                raise ValueError("No file name provided")
        
        self.ms.add_mesh(self.mesh)
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        self.ms.save_current_mesh(file_name)
    
    def render(self):
        render([self.file_name])
        
    def __str__(self):
        return self.file_name
    
    
    # -------------------- 3. Shape resampling -------------------------
    
    def resample(self, target_faces = NR_DESIRED_FACES):
        """_summary_ resample the shape to a fixed number of faces

        Args:
            target_faces (int, optional): Defaults to NR_DESIRED_FACES.
        """
        
        assert(target_faces > 0)
        faces = self.mesh.face_number()
        
        self.logger.log(f"Resampling shape faces: {faces} -> {target_faces}")
        
        if faces == target_faces:
            return
        
        if faces > target_faces:
            self._sub_sample(target_faces)
        else:
            self._super_sample(target_faces)
        
        
        self.vertices = self.mesh.vertex_matrix()
        self.faces = self.mesh.face_matrix()
        faces = self.mesh.face_number()
        
        self.logger.log(f"Resampling succeeded with {faces} faces")
        
    
    def _sub_sample(self, target_faces = NR_DESIRED_FACES):
        """_summary_ Sub sampling is done by using the Quadric Edge Collapse Decimation filter
            Sub sampling to a fixed number of faces

        Args:
            target_faces (int, optional): Target number of faces. Defaults to NR_DESIRED_FACES = 5000.
        """

        # https://pymeshlab.readthedocs.io/en/latest/filter_list.html?highlight=Quadratic%20Edge%20Collapse%20Detection#meshing_decimation_quadric_edge_collapse
        # https://support.shapeways.com/hc/en-us/articles/360022742294-Polygon-reduction-with-MeshLab
        # https://help.sketchfab.com/hc/en-us/articles/205852789-MeshLab-Decimating-a-model

        
        # This would be done before normalisation, so we dont need to preserve boundaries, normal, etc

        self.ms.apply_filter("meshing_decimation_quadric_edge_collapse", targetfacenum=target_faces, qualitythr=1)

        self.mesh = self.ms.current_mesh()
        
    def _super_sample(self, target_faces = NR_DESIRED_FACES):
        """_summary_  Super sampling is done by using the Subdivision Surfaces filter
           Super sampling a certain amount of iterations until the number of faces is greater than the desired number of faces
           When greater than the desired number of faces, sub sample to the desired number of faces 

        Args:
            target_faces (int, optional): Target number of faces. Defaults to NR_DESIRED_FACES = 5000.
        """
        # https://pymeshlab.readthedocs.io/en/latest/filter_list.html?highlight=Remeshing%2C%20Simplification%20and%20Reconstruction#meshing_surface_subdivision_butterfly
        # all filters for Subdivision Surfaces below could be used, PyMeshLab has implementations for all of them
        # email sent out to ask which one is most appropriate to use
        # https://www.universal-robots.com/media/1818206/12.png

        
        faces = self.mesh.face_number()
        
        self.logger.log(f"Super sampling... {faces} -> {target_faces}")
        
        iterations = int(math.log(target_faces // faces, 4)) + 1 # with each iteration the number of faces is 4 times the previous number of faces
        self.logger.log(f"Planning to do {iterations} iterations for midpoint subdivision")
        
        # trying to apply subdivision filter, in case non manifold mesh is given repairing it
        try:
            self.ms.apply_filter("meshing_surface_subdivision_midpoint", iterations=iterations)
        except Exception as e:
            self.logger.error(f"Super sampling failed with {faces} faces due to {e}")
            self.logger.log(f"Trying to repair mesh...")
            self.ms.apply_filter("meshing_repair_non_manifold_edges")
            self.logger.log("Mesh repaired, trying to super sample again")
            
            self.mesh = self.ms.current_mesh()
            faces = self.mesh.face_number()
            self.logger.log(f"Faces after repair: {faces}")
            try:
                self.ms.apply_filter("meshing_surface_subdivision_midpoint", iterations=iterations)        
            except Exception as e:
                self.logger.error(f"Super sampling failed with {faces} faces due to {e}")
                
        self.mesh = self.ms.current_mesh()
        faces = self.mesh.face_number()
            
        self.logger.log(f"\t Current number of faces: {faces}")
        
                   
        if faces > target_faces:
            self._sub_sample(target_faces)
         
    # -------------------- 4. Shape Normalization ----------------------
    
    def get_barycenter(self):
        """
        _summary_: Computing the barycenter of a shape
        """ 
        x = 0
        y = 0
        z = 0
        N = len(self.faces)
        
        for face in self.faces:
            triangle = self.vertices[face]
            area = self.get_triangle_area(triangle) 
            x += area * (triangle[0][0] + triangle[1][0] + triangle[2][0]) / 3
            y += area * (triangle[0][1] + triangle[1][1] + triangle[2][1]) / 3
            z += area * (triangle[0][2] + triangle[1][2] + triangle[2][2]) / 3
        
        return [x / N, y / N, z / N]

    def translate_barycenter(self):
        """
            _summary_ translate the shape so that its barycenter is in the origin of the coordinate system
            _param_ vertices(list of 3D vectors): the vertices matrix of the shape
            _param_ barycenter(3D vector): the barycenter of the shape
        """
        
        self.logger.log("Translating the barycenter to the origin of the coordinate system")
        
        barycenter = self.get_barycenter()
        for vertex in self.vertices:
            vertex[0] -= barycenter[0]
            vertex[1] -= barycenter[1]
            vertex[2] -= barycenter[2]
        
    def principal_component_analysis(self):
        """_summary_ compute the principal components of the shape

        Args:
            vertices (_type_): _description_

        Returns:
            (np.ndarray, np.ndarray): (eigenvalues, eigenvectors)
        """
        
        # computing the covariance matrix
        # When computing the covariance matrix each row of the input represents a variable, 
        # and each column a single observation of all those variables,
        # therefore we need to transpose the matrix of vertices

        covariance_matrix = np.cov(self.vertices.T)
        
        # computing the principal components
        eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
        
        eigen_components = [(eigen_values[i], eigen_vectors[:, i]) for i in range(len(eigen_values))]
        
        # sorting the eigen vectors according to the eigen values
        eigen_components.sort(key=lambda x: x[0], reverse=True)
        
        eigen_vectors = [eigen_components[i][1] for i in range(3)]
        eigen_values = [eigen_components[i][0] for i in range(3)]
        
        return eigen_values, eigen_vectors
    
    def align_with_principal_components(self):
        """
        _summary_ align the shape with the principal components
        """
        self.logger.log("Aligning the shape with the principal components")
        
        _, eigen_vectors = self.principal_component_analysis()

        # computing the rotation matrix
        rotation_matrix = np.array(eigen_vectors)
        
        # applying the rotation matrix
        self.vertices = np.dot(rotation_matrix, self.vertices.T).T
        
        self.mesh = Mesh(self.vertices, self.faces)
            
    def flip_on_moment(self):
        """_summary_ Flipping the shape based on moment test
        """
        self.logger.log("Flipping the shape based on moment test")
        
        f_x = 0
        f_y = 0
        f_z = 0
        
        for face in self.faces:
            triangle = self.vertices[face]
            triangle_center = (triangle[0] + triangle[1] + triangle[2]) / 3
            
            f_x += self.sign(triangle_center[0]) * triangle_center[0] ** 2
            f_y += self.sign(triangle_center[1]) * triangle_center[1] ** 2
            f_z += self.sign(triangle_center[2]) * triangle_center[2] ** 2
        
        
        transformation_matrix = np.array([[self.sign(f_x), 0, 0], [0, self.sign(f_y), 0], [0, 0, self.sign(f_z)]])
        
        self.vertices = np.matmul(transformation_matrix, self.vertices.T).T
        
        self.mesh = Mesh(self.vertices, self.faces)
    
    def get_nr_of_vertices_diff_on_positive_axis(self, axis: int = 0):
        """_summary_

        Args:
            axis (int, optional): 0:x 1:y 2:z. Defaults to 0.
        """
        assert(axis >= 0 and axis <= 2)
        nr_diff_vertices = 0
        for vertex in self.vertices:
            nr_diff_vertices += self.sign(vertex[axis]) * vertex[axis] ** 2 
        
        return nr_diff_vertices
                    
    def get_bounding_box_dimensions(self):
        bbox = self.mesh.bounding_box()
        return [bbox.dim_x(), bbox.dim_y(), bbox.dim_z()]
    
    def rescale_shape(self):
        """
        _summary_ rescale the shape so that the bounding box is the unit cube
        """
        self.logger.log("Rescaling the shape so that the bounding box is the unit cube")
        
        [x_max, y_max, z_max] = list(np.max(self.vertices, axis=0))
        [x_min, y_min, z_min] = list(np.min(self.vertices, axis=0))
        
        m = max(abs(x_max - x_min), abs(y_max - y_min), abs(z_max - z_min))
        
        if m == 1.0:
            return
        
        scale_factor = 1 / m
        
        self.logger.log("Scaling factor: " + str(scale_factor))
        
        for vertex in self.vertices:
            vertex[0] = vertex[0] * scale_factor   
            vertex[1] = vertex[1] * scale_factor
            vertex[2] = vertex[2] * scale_factor

        self.mesh = Mesh(self.vertices, self.faces)
            
    def normalize(self):
        """
        _summary_ normalize the shape: 
        
        1) Translate the barycenter to the origin of the coordinate system
        2) Align the shape with the principal components
        3) Flip the shape if necessary
        4) Scale the shape so that the bounding box is the unit cube
        """
        self.translate_barycenter()
        self.align_with_principal_components()
        self.flip_on_moment()
        self.rescale_shape()
        
        self.mesh = Mesh(self.vertices, self.faces)
        self.ms.add_mesh(self.mesh, "normalized")

    def get_features(self):
        """_summary_ compute the features of the shape

        Returns:
            list: the features of the shape
        """
        return get_features(self.file_name)

    #---------------------------------------#       
    @staticmethod
    @jit 
    def sign(x):
        if x == 0:
            return 0
        return 1 if x > 0 else -1
    
    @staticmethod
    def get_triangle_area(triangle):
        return 0.5 * np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))
    
    @staticmethod
    @jit
    def get_angle_between_vertices(v1, v2, v3):
        return np.arccos(np.dot(v1 - v2, v3 - v2) / (np.linalg.norm(v1 - v2) * np.linalg.norm(v3 - v2)))
    
    @staticmethod
    @jit
    def get_tetrahedron_volume(v1,v2,v3,v4 = np.array([0,0,0])):
        return np.dot(v1 - v4, np.cross(v2 - v4, v3 - v4)) / 6
    
    @staticmethod
    def compute_volume(vertices, faces):
        volume = 0
        for face in faces:
            [v1,v2,v3] = vertices[face]
            volume += Shape.get_tetrahedron_volume(v1,v2,v3)
        return abs(volume)

    # ----------------- 5. Feature extraction from shape ---------------------#
    def get_elongation(self):
        """
        _summary_ compute the elongation of the shape
        """
        [dim_x, dim_y, dim_z] = self.get_bounding_box_dimensions()
        return max(dim_x, dim_y, dim_z) / min(dim_x, dim_y, dim_z)
    
    def get_surface_area(self):
        """
        _summary_ compute the surface area of the shape
        """
        surface_area = 0
        
        for face in self.faces:
            triangle = self.vertices[face]
            surface_area += self.get_triangle_area(triangle)
        
        return surface_area
    
    def get_bbox_volume(self):
        """
        _summary_ compute the volume of bounding box of the shape
        """
        
        [dim_x, dim_y, dim_z] = self.get_bounding_box_dimensions()
        return dim_x * dim_y * dim_z

    def get_volume(self):
        """
        _summary_ compute the volume of the shape
        
        APPROVED!
        
        Returns:
            float: the volume of the shape
        """
        
        volume = 0
        
        for face in self.faces:
            [v1,v2,v3] = self.vertices[face]
            volume += self.get_tetrahedron_volume(v1,v2,v3)
        
        return abs(volume)
    
    def get_convex_hull_measures(self):
        """_summary_ compute the convex hull measures of the shape

        Returns:
            [float, float]: [volume, surface_area]
        """
        self.ms.generate_convex_hull()
        measures = self.ms.get_geometric_measures()
        
        mesh = self.ms.current_mesh()
        
        if "mesh_volume" not in measures.keys():
            volume = Shape.compute_volume(mesh.vertex_matrix(), mesh.face_matrix())
        else:
            volume = measures["mesh_volume"]    
            
        surface_area = measures["surface_area"]
        
        self.ms.delete_current_mesh()
        
        return [volume, surface_area]    
    
    def get_compactness(self):
        """
        _summary_ compute the compactness of the shape
        """
        # Formula: S^3 / (36 * \pi * V^2)
        return (self.get_surface_area() ** 3) / (36 * math.pi * self.get_volume() ** 2)

    def get_diameter(self):
        """
        _summary_ compute the diameter of the shape
        """
        
        # TODO: not very scalable, O(N^2), where N is the number of vertices
        # TODO: find a better way to compute the diameter
        diameter = 0
        
        for i in range(len(self.vertices)):
            for j in range(i,len(self.vertices)):
                distance = np.linalg.norm(self.vertices[i] - self.vertices[j])
                if distance > diameter:
                    diameter = distance
        return diameter
    
    def get_eccentricity(self):
        """
        _summary_ compute the eccentricity of the shape
        """
        eigenvalues, _ = self.principal_component_analysis()
        
        # lambda_1 >= lambda_2 >= lambda_3
        # returning |lambda_1| / |lambda_3|        
        return abs(eigenvalues[0]) / abs(eigenvalues[2])
       
    def get_A3(self, samples = NR_SAMPLES_FOR_FEATURE_DESCRIPTORS_A3, dimension = FEATURE_DESCRIPTORS_DIMENSIONS_A3):
        """
        _summary_ compute the A3 of the shape
        
        distribution histogram of the angles between 3 random vertices
        """
        
        n = self.vertices.shape[0] # number of vertices
        
        angles = []
        
        for _ in range(samples):
            v1, v2, v3 = self.vertices[np.random.choice(n, 3, replace=False)]
            angles.append(self.get_angle_between_vertices(v1, v2, v3))
                    
        hist, _ = np.histogram(angles, bins=dimension, range=(0, math.pi), density=True)
        hist = list(hist / np.sum(hist)) # normalizing
        
        self.logger.log("Histogram for A3 feature vector is: " + str(hist))
        
        return hist
    
    def get_D1(self, samples = NR_SAMPLES_FOR_FEATURE_DESCRIPTORS_D1, dimension = FEATURE_DESCRIPTORS_DIMENSIONS_D1):
        """
        _summary_ compute the D1 of the shape
        
        distribution histogram of the distances between barycenter and a random vertex
        """
        n = self.vertices.shape[0] # number of vertices
        barycenter = self.get_barycenter()
        distances = []
        
        for _ in range(samples):
            # getting a random vertex
            v = self.vertices[np.random.choice(n, 1, replace=False)]
            
            distances.append(np.linalg.norm(v - barycenter))
        
        hist, _ = np.histogram(distances, bins=dimension, range=(0,1), density=True)
        hist = list(hist / np.sum(hist)) # normalizing
        
        self.logger.log(f"Histogram for D1 feature vector is: {hist}")
        
        return hist
    
    def get_D2(self, samples = NR_SAMPLES_FOR_FEATURE_DESCRIPTORS_D2, dimension = FEATURE_DESCRIPTORS_DIMENSIONS_D2):
        """
        _summary_ compute the D2 of the shape
        
        distance between 2 random vertices
        """
        n = self.vertices.shape[0] # number of vertices
        
        distances = []
           
        for _ in range(samples):
            v1, v2 = self.vertices[np.random.choice(n, 2, replace=False)]
            distances.append(np.linalg.norm(v1 - v2))
        
        hist, _ = np.histogram(distances, bins=dimension, range=(0, math.sqrt(3)), density=True)
        hist = list(hist / np.sum(hist)) # normalizing
        
        self.logger.log(f"Histogram for D2 feature vector is: {hist}")
        
        return hist
    
    def get_D3(self, samples = NR_SAMPLES_FOR_FEATURE_DESCRIPTORS_D3, dimension = FEATURE_DESCRIPTORS_DIMENSIONS_D3):
        """
        _summary_ compute the D3 of the shape
        
        square root of area of triangle given by 3 random vertices
        """
         
        n = self.vertices.shape[0] # number of vertices
        
        areas = []
        for _ in range(samples):
            v1, v2, v3 = self.vertices[np.random.choice(n, 3, replace=False)]
            areas.append(math.sqrt(self.get_triangle_area([v1, v2, v3])))
        
        hist, _ = np.histogram(areas, bins=dimension, range=(0, (math.sqrt(3) / 2) ** (1/2) ), density=True)
        hist = list(hist / np.sum(hist)) # normalizing
        
        self.logger.log(f"Histogram for D3 feature vector is: {hist}")
        
        return hist
    
    def get_D4(self, samples = NR_SAMPLES_FOR_FEATURE_DESCRIPTORS_D4, dimension = FEATURE_DESCRIPTORS_DIMENSIONS_D4):
        """
        _summary_ compute the D4 of the shape
        cube root of volume of tetrahedron formed by 4 random vertices
        """
        n = self.vertices.shape[0] # number of vertices
        
        volumes = []
        
        for _ in range(samples):
            v1, v2, v3, v4 = self.vertices[np.random.choice(n, 4, replace=False)]
            volumes.append(abs(self.get_tetrahedron_volume(v1, v2, v3, v4)) ** (1/3))
        
        hist, _ = np.histogram(volumes, bins=dimension, range=(0,(1/3) ** (1/3)), density=True)
        hist = list(hist / np.sum(hist)) # normalizing
        
        self.logger.log(f"Histogram for D4 feature vector is: {hist}")
        
        return hist
    