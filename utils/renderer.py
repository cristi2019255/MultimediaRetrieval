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

import pymeshlab
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d


def render_file(filename="./LabeledDB_new/Airplane/61.off"):
    meshes = pymeshlab.MeshSet()
    meshes.load_new_mesh(filename)
    display_mesh(meshes.current_mesh(), filename)


def display_mesh(mesh, filename):
    print('Displaying mesh')
    v = np.array(mesh.vertex_matrix())    
    f = np.array(mesh.face_matrix())
    vertex_count = mesh.vertex_number()
    faces_count = mesh.face_number()
    
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    pc = art3d.Poly3DCollection(v[f], edgecolor="black")
    ax.add_collection(pc)
    
    plt.axis('off')
    plt.title(filename)
    plt.show()