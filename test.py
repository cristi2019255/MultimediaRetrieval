# Here we test everything before implementing any code
# Like, checking the libraries API, etc.

from pymeshlab import MeshSet, Percentage
from shape import Shape
from utils.renderer import render

NR_DESIRED_VERTICES = 1000


def test():
    # test for sub sampling, if too many vertices, it will be reduced

    file_name = "./LabeledDB_new/Airplane/61.off"
    ms = MeshSet()
    ms.load_new_mesh(file_name)
    render([file_name])

    # kinda works, but further investigation is needed
    current_nr_vertices = ms.current_mesh().vertex_number()
    percentage = Percentage((current_nr_vertices / NR_DESIRED_VERTICES))
    ms.apply_filter("generate_resampled_uniform_mesh", cellsize=percentage)  # kind of a subsampling

    new_file_name = "./test_data/61_resampled.ply"

    ms.save_current_mesh(new_file_name)
    render([new_file_name])


def test2():
    # test for super sampling, if too few vertices, it will be increased

    file_name = "./LabeledDB_new/Airplane/61.off"
    ms = MeshSet()
    ms.load_new_mesh(file_name)
    render([file_name])

    # does work, but don't know how to regenerate faces of object, further investigation is needed
    ms.apply_filter("generate_resampled_uniform_mesh", cellsize=Percentage(0.7))  # kind of a subsampling

    # mesh = ms.current_mesh()
    # faces_indices = mesh.polygonal_face_list()

    # ms.apply_filter("generate_sampling_stratified_triangle", samplenum = 10000) # kind of a super sampling
    # mesh = ms.current_mesh()
    # mesh = Mesh(mesh.vertex_matrix(), faces_indices)
    # ms.apply_filter('generate_surface_reconstruction_vcg')
    # ms.add_mesh(mesh)

    new_file_name = "test_data/61_resampled.ply"

    ms.save_current_mesh(new_file_name)
    render([new_file_name])


def test_shape():
    shape = Shape("./LabeledDB_new/Airplane/61.ply")
    shape.normalize()
    shape.save_mesh("./test_data/61_normalized.ply")
    render(["./LabeledDB_new/Airplane/61.ply", "./61_normalized.ply"])


def test_subsampling():
    file_name = "./LabeledDB_new/Airplane/61.off"
    ms = MeshSet()
    ms.load_new_mesh(file_name)
    render([file_name])

    # https://pymeshlab.readthedocs.io/en/latest/filter_list.html?highlight=Quadratic%20Edge%20Collapse%20Detection#meshing_decimation_quadric_edge_collapse
    # https://support.shapeways.com/hc/en-us/articles/360022742294-Polygon-reduction-with-MeshLab
    # 1: calculate the mean number of faces in the distribution for NR_DESIRED_VERTICES
    # for testing, 1 = 1000
    # 2: need to pick an appropriate quality threshold (qualitythr), right now it's just a number pulled out of my ass

    # This would be done before normalisation, so we dont need to preserve boundaries, normal, etc

    ms.apply_filter("meshing_decimation_quadric_edge_collapse", targetfacenum=NR_DESIRED_VERTICES, qualitythr=0.9)

    new_file_name = "./test_data/61_subsampled.ply"

    ms.save_current_mesh(new_file_name)
    render([new_file_name])


# test_subsampling()


def test_supersampling():
    file_name = "./LabeledDB_new/Airplane/61.off"
    ms = MeshSet()
    ms.load_new_mesh(file_name)
    render([file_name])

    # https://pymeshlab.readthedocs.io/en/latest/filter_list.html?highlight=Remeshing%2C%20Simplification%20and%20Reconstruction#meshing_surface_subdivision_butterfly

    # all filters for Subdivision Surfaces below could be used, PyMeshLab has implementations for all of them
    # email sent out to ask which one is most appropriate to use
    # https://www.universal-robots.com/media/1818206/12.png

    ms.apply_filter("meshing_surface_subdivision_butterfly", iterations=3)

    new_file_name = "./test_data/61_supersampled.ply"

    ms.save_current_mesh(new_file_name)
    render([new_file_name])


# test_supersampling()

"""
theoretical pipeline:
1. Choose an appropriate number of faces [d]; Can be done via either the mean or median in the original data.
2. Check the number of faces of the model [f]
    If f = x:
        pass
    If f > x:
        Subsample so f = x
    If f < x:
        Supersample until f > x
        Subsample so f = x

As a result, all models would have the same number of faces. 

I asked the prof. if it would be a good way of doing things. Will update accordingly. 
"""
