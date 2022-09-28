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

# create a new pyvista theme
# inherit from the default
my_theme = pyvista.themes.DefaultTheme()

# change mesh color
my_theme.color = "silver"

# change background color to white
my_theme.background = 'white'

# change the fonts
my_theme.font.color = 'black'
pyvista.global_theme.font.family = 'times'

# load the new theme
pyvista.global_theme.load_theme(my_theme)


def render(filenames=["./LabeledDB_new/Airplane/61.ply"], show_features=True):
    """_summary_ Rendering shapes and optionally their features from a list of files

    Args:
        filenames (list[str], optional): Defaults to ["./LabeledDB_new/Airplane/61.ply"].
        show_features (Bool, optional): Defaults to True
    """

    if show_features:
        rows = 2
        row_weights = [4, 1]
    else:
        rows = 1
        row_weights = [1]

    # setting up the plotter
    plotter = pyvista.Plotter(shape=(rows, len(filenames)), row_weights=row_weights)  # instantiate the plotter

    # adding the meshes to the plotter
    for index, filename in enumerate(filenames):

        if ('.off' in filename):
            filename = off2ply(filename)

        mesh = pyvista.read(filename)
        plotter.subplot(0, index)
        plotter.camera.zoom(0.35)
        plotter.show_grid()
        plotter.add_mesh(mesh, show_edges=True, smooth_shading=True)
        plotter.add_title(filename, font_size=10)
        plotter.add_axes(interactive=True)
        plotter.add_bounding_box()

        if show_features:
            plotter.subplot(1, index)
            [n_faces, n_vertices, faces_type, axis_aligned_bounding_box] = get_features(filename)
            axis_aligned_bounding_box = list(map(lambda x: round(x, 2), axis_aligned_bounding_box))
            data = f"Number of faces: {n_faces} \nNumber of vertices: {n_vertices} \n" \
                   f"Faces type: {faces_type} \nAxis aligned bounding box: \n{axis_aligned_bounding_box}"

            plotter.add_text(text=data, font_size=14, position="upper_left", name="text")

    plotter.show()  # show the rendering window


def test_render():
    filenames = []
    for i in range(1, 5):
        j = 60 + i
        filenames.append(f"./LabeledDB_new/Airplane/{j}.off")
    print(filenames)
    render(filenames)


def test_render_report():
    filenames = ["./LabeledDB_new/Airplane/61.ply", "./LabeledDB_new/Octopus/121.ply"]
    render(filenames, show_features=False)
    render(filenames, show_features=True)
