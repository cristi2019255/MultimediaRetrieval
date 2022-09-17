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

import os
from pymeshlab import MeshSet

def off2ply(filename = "./LabeledDB_new/Airplane/61.off"):
    meshes = MeshSet()
    meshes.load_new_mesh(filename)
    filename = filename.replace('.off', '.ply')
    meshes.save_current_mesh(filename)
    return filename

def convert_to_ply(directory = "./LabeledDB_new"):
    for r, d, f in os.walk(directory):                
        for file in f:            
            if ('.off' in file):
                off2ply(os.path.join(r, file))

def scan_files(directory = "./LabeledDB_new"):    
    files = {}
    for r, d, f in os.walk(directory):                
        for file in f:
            if ('.ply' in file):
                dir = r.split('/')[-1]
                if not dir in files:
                    files[dir] = [os.path.join(r, file)]
                else:                
                    files[dir].append(os.path.join(r, file))        
    return files


def get_features(filename = "./LabeledDB_new/Airplane/61.off"):
    meshes = MeshSet()
    meshes.load_new_mesh(filename)
    mesh = meshes.current_mesh()
    
    faces_count = mesh.face_number()
    vertices_count = mesh.vertex_number()
    faces_type = 'triangles' if mesh.face_number() == 3 else 'quads' # to do rewritte this function
    axis_aligned_bounding_box = 'something' # to do rewritte this function
    
    return [faces_count, vertices_count, faces_type, axis_aligned_bounding_box]