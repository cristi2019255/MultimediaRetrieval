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

class Shape:
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces
    
    def __init__(self, file_name: str):
        self.ms = MeshSet()
        self.ms.load_new_mesh(file_name)
        self.file_name = file_name
        self.mesh = self.ms.current_mesh()
        self.vertices = self.mesh.vertex_matrix()
        self.faces = self.mesh.face_matrix()
    
    def save_mesh(self, file_name: str = None):
        if file_name is None:
            file_name = self.file_name
            if file_name is None:
                raise ValueError("No file name provided")
        
        self.ms.add_mesh(self.mesh)
        self.ms.save_current_mesh(file_name)
    
    def render(self):
        render([self.file_name])
    
    
    def get_barycenter(self):
        """
        _summary_: Computing the barycenter of a shape
        
        TODO: get the correct area for each point
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
        
        print("[INFO] Rescaling the shape so that the bounding box is the unit cube")
        
        bbox = self.mesh.bounding_box()
        [dim_x, dim_y, dim_z] = [bbox.dim_x(), bbox.dim_y(), bbox.dim_z()]
        scale_factor = 1 / max(dim_x, dim_y, dim_z)
        
        for vertex in self.vertices:
            vertex = vertex * scale_factor   

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
        
        self.save_mesh()
    

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
        axis_aligned_bounding_box = [bounding_box.dim_x(), bounding_box.dim_y(), bounding_box.dim_z(),
                                    bounding_box.diagonal()]

        return [faces_count, vertices_count, faces_type, axis_aligned_bounding_box]

    #---------------------------------------#        
    @staticmethod
    def sign(x):
        if x == 0:
            return 0
        return 1 if x > 0 else -1
    
    @staticmethod
    def get_triangle_area(triangle) -> float:
        return 0.5 * np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))