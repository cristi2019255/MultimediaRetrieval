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

from pymeshlab import MeshSet, Mesh
import numpy as np
from utils.renderer import render
import os

NR_DESIRED_FACES = 5000

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
        self.log = log
        
    
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
        self.log = log
    
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
        
        if self.log:
            print("[INFO] Resampling shape from ", faces," to ", target_faces, " faces")
        
        if faces == target_faces:
            return
        
        if faces > target_faces:
            self._sub_sample(target_faces)
        else:
            self._super_sample(target_faces)
        
        
        self.vertices = self.mesh.vertex_matrix()
        self.faces = self.mesh.face_matrix()
        
        if self.log:
            print("[INFO] Resampling succeeded with ", self.mesh.face_number(), " faces")
            
    
    def _sub_sample(self, target_faces = NR_DESIRED_FACES):
        """_summary_ Sub sampling is done by using the Quadric Edge Collapse Decimation filter
            Sub sampling to a fixed number of faces

        Args:
            target_faces (int, optional): Target number of faces. Defaults to NR_DESIRED_FACES = 5000.
        """
        # https://pymeshlab.readthedocs.io/en/latest/filter_list.html?highlight=Quadratic%20Edge%20Collapse%20Detection#meshing_decimation_quadric_edge_collapse
        # https://support.shapeways.com/hc/en-us/articles/360022742294-Polygon-reduction-with-MeshLab
        # 1: calculate the mean number of faces in the distribution for NR_DESIRED_FACES
        # for testing, 5000
        # 2: TODO: need to pick an appropriate quality threshold (qualitythr), right now it's just a number pulled out of my ass

        # This would be done before normalisation, so we dont need to preserve boundaries, normal, etc

        self.ms.apply_filter("meshing_decimation_quadric_edge_collapse", targetfacenum=target_faces, qualitythr=0.9)
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
        
        if self.log:
            print(f'[INFO] Super sampling... {faces} -> {target_faces}')
        
        # It works but it is provoking me anxiety the way we do it (Cristian Grosu), I wanna change it
        
        old_faces = faces
        while faces < target_faces:
            self.ms.apply_filter("meshing_surface_subdivision_butterfly", iterations=1)
            self.mesh = self.ms.current_mesh()
            faces = self.mesh.face_number()
            
            if self.log:
                print('[INFO] \t Current number of faces: ', faces)
            
            
            if old_faces == faces:
                break
            old_faces = faces
            
                    
        if faces > target_faces:
            self._sub_sample(target_faces)
            return
            
        self.mesh = self.ms.current_mesh()

    # -------------------- 4. Shape Normalization ----------------------
    
    def get_barycenter(self):
        """
        _summary_: Computing the barycenter of a shape
        
        TODO: is this the correct way to compute the barycenter?
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
        if self.log:
            print("[INFO] Translating the barycenter to the origin of the coordinate system")
        
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
            _type_: _description_
        """
        
        # computing the covariance matrix
        # When computing the covariance matrix each row of the input represents a variable, 
        # and each column a single observation of all those variables, therefore we need to transpose the matrix of vertices
        covariance_matrix = np.cov(self.vertices.T)
        
        # computing the principal components
        eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
        
        return eigen_values, eigen_vectors
    
    def align_with_principal_components(self):
        """
        _summary_ align the shape with the principal components
        """
        if self.log:
            print("[INFO] Aligning the shape with the principal components")
        
        eigen_values, eigen_vectors = self.principal_component_analysis()

        eigen_components = [(eigen_values[i], eigen_vectors[:, i]) for i in range(len(eigen_values))]
        
        # sorting the eigen vectors according to the eigen values
        eigen_components.sort(key=lambda x: x[0], reverse=True)
        
        x_vec = eigen_components[0][1]
        y_vec = eigen_components[1][1]
        z_vec = eigen_components[2][1]
        
        # computing the rotation matrix
        rotation_matrix = np.array([x_vec, y_vec, z_vec])
        
        # applying the rotation matrix
        self.vertices = np.dot(rotation_matrix.T, self.vertices.T).T
        
        self.mesh = Mesh(self.vertices, self.faces)
            
    def flip_on_moment(self):
        """_summary_ Flipping the shape based on moment test
        """
        if self.log:
            print("[INFO] Flipping the shape based on moment test")
        
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
        
        self.vertices = np.dot(self.vertices, transformation_matrix.T)
        
        self.mesh = Mesh(self.vertices, self.faces)
    
    
        
    def rescale_shape(self):
        """
        _summary_ rescale the shape so that the bounding box is the unit cube
        """
        if self.log:
            print("[INFO] Rescaling the shape so that the bounding box is the unit cube")
        
        bbox = self.mesh.bounding_box()
        dims = [abs(bbox.dim_x()), abs(bbox.dim_y()), abs(bbox.dim_z())]
        m = max(dims)
    
        if m <= 1.0:
            return
        
        scale_factor = 1 / m
        
        if self.log:
            print("[INFO] Scaling factor: " + str(scale_factor))
        
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
        faces_count = self.mesh.face_number()
        vertices_count = self.mesh.vertex_number()
        faces_ratio = self.mesh.face_matrix().shape[1]  # TODO: check this, I think it's wrong

        faces_type = 'triangles' if faces_ratio == 3 else 'quads' if faces_ratio == 4 else 'mix'
        bounding_box = self.mesh.bounding_box()
        axis_aligned_bounding_box = [abs(bounding_box.dim_x()), abs(bounding_box.dim_y()), abs(bounding_box.dim_z()),
                                    abs(bounding_box.diagonal())]

        return [faces_count, vertices_count, faces_type, axis_aligned_bounding_box]

    #---------------------------------------#        
    @staticmethod
    def sign(x):
        if x == 0:
            return 0
        return 1 if x > 0 else -1
    
    @staticmethod
    def get_triangle_area(triangle):
        return 0.5 * np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))
    
    
    # ----------------- 5. Feature extraction from shape ---------------------#
    def get_elongation(self):
        """
        _summary_ compute the elongation of the shape
        """
        bbox = self.mesh.bounding_box()
        [dim_x, dim_y, dim_z] = [bbox.dim_x(), bbox.dim_y(), bbox.dim_z()]
        return max(dim_x, dim_y, dim_z) / min(dim_x, dim_y, dim_z)
    
    def get_curvature(self):
        """
        _summary_ compute the curvature of the shape
        """
        return 1.0
    
    # TODO: check this and add the other features
    