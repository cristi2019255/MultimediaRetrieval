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
import time
from pymeshlab import MeshSet

from utils.Logger import Logger


def off2ply(filename="./LabeledDB_new/Airplane/61.off"):
    """_summary_ Converting an .off file to .ply file

    Args:
        filename (str, optional): Defaults to "./LabeledDB_new/Airplane/61.off".

    Returns:
        str : new file name with .ply extension
    """
    if filename.endswith('.ply'):
        return filename
    
    meshes = MeshSet()
    meshes.load_new_mesh(filename)
    if filename.endswith('.off') or filename.endswith('.obj'):
        new_filename = filename[:-4] + '.ply'
    else:
        return filename
    meshes.save_current_mesh(new_filename)
    return filename


def convert_to_ply(directory="./LabeledDB_new"):
    """_summary_ Converts all files in a directory with .off extension to .ply files

    Args:
        directory (str, optional): Defaults to "./LabeledDB_new".
    """
    for r, d, f in os.walk(directory):
        for file in f:
            off2ply(os.path.join(r, file))


def scan_files(directory="./LabeledDB_new", limit=None):
    """_summary_ Returns a list of all files in a directory with .ply extension

    Args:
        directory (str, optional): Defaults to "./LabeledDB_new".

    Returns:
        [str]: file names
    """
    files = {}
    for r, d, f in os.walk(directory):
        if "test" in r:
            continue
        for file in f:    
            if ('.ply' in file):
                dir = r.split('/')[-1]
                if not dir in files:
                    files[dir] = [os.path.join(r, file)]
                else:
                    files[dir].append(os.path.join(r, file))
    
    if limit is None:
        return files
    else:
        k = list(files.keys())[0]
        return {k: files[k][:limit]}


def get_features(filename="./LabeledDB_new/Airplane/61.off"):
    """_summary_ Computing basic features for a shape

    Args:
        filename (str, optional): The file where the shape is stored. Defaults to "./LabeledDB_new/Airplane/61.off".

    Returns:
        [int, int, str, [int, int, int, int]]: faces_count, vertices_count, faces_type, axis_aligned_bounding_box
    """
    meshes = MeshSet()
    meshes.load_new_mesh(filename)
    mesh = meshes.current_mesh()

    faces_count = mesh.face_number()
    vertices_count = mesh.vertex_number()
    faces_ratio = mesh.face_matrix().shape[1]  # TODO: check this, I think it's wrong

    faces_type = 'triangles' if faces_ratio == 3 else 'quads' if faces_ratio == 4 else 'mix'
    bounding_box = mesh.bounding_box()
    axis_aligned_bounding_box = [bounding_box.dim_x(), bounding_box.dim_y(), bounding_box.dim_z(),
                                 bounding_box.diagonal()]

    return [faces_count, vertices_count, faces_type, axis_aligned_bounding_box]

def track_progress(function):
    start = time.time()
    function()
    end = time.time()
    logger = Logger(active=True)
    logger.success(f"{function.__name__} finished in {end - start} seconds")