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

import pyvista

from utils.tools import get_features, off2ply


def render(filenames=["./LabeledDB_new/Airplane/61.ply"]):
    # setting up the plotter    
    plotter = pyvista.Plotter(shape=(2, len(filenames)), row_weights=[4, 1])  # instantiate the plotter

    # adding the meshes to the plotter
    for index, filename in enumerate(filenames):

        if ('.off' in filename):
            filename = off2ply(filename)

        mesh = pyvista.read(filename)
        plotter.subplot(0, index)
        plotter.camera.zoom(0.35)
        plotter.show_grid()
        plotter.add_mesh(mesh, color="tan", show_edges=True, smooth_shading=True)
        plotter.add_title(filename, font_size=10)
        plotter.add_axes(interactive=True)
        plotter.add_bounding_box()
        plotter.subplot(1, index)

        [n_faces, n_vertices, faces_type, axis_aligned_bounding_box] = get_features(filename)
        data = f"Number of faces: {n_faces} \nNumber of vertices: {n_vertices} \nFaces type: {faces_type} \nAxis aligned bounding box: \n{axis_aligned_bounding_box}"

        plotter.add_text(text=data, font_size=14, color="black", position="upper_left", name="text")

    plotter.show()  # show the rendering window


def test_render():
    filenames = []
    for i in range(1, 5):
        j = 60 + i
        filenames.append(f"./LabeledDB_new/Airplane/{j}.off")
    print(filenames)
    render(filenames)
