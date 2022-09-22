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
    percentage = Percentage((current_nr_vertices / NR_DESIRED_VERTICES ))
    ms.apply_filter("generate_resampled_uniform_mesh", cellsize = percentage) # kind of a subsampling
    
    new_file_name = "./61_resampled.ply"
    
    ms.save_current_mesh(new_file_name)
    render([new_file_name])

def test2():
    # test for super sampling, if too few vertices, it will be increased
    
    file_name = "./LabeledDB_new/Airplane/61.off"
    ms = MeshSet()
    ms.load_new_mesh(file_name)
    render([file_name])
    
    # does work, but don't know how to regenerate faces of object, further investigation is needed
    ms.apply_filter("generate_resampled_uniform_mesh", cellsize = Percentage(0.7)) # kind of a subsampling
    
    #mesh = ms.current_mesh()
    #faces_indices = mesh.polygonal_face_list()
    
    #ms.apply_filter("generate_sampling_stratified_triangle", samplenum = 10000) # kind of a super sampling
    #mesh = ms.current_mesh()
    #mesh = Mesh(mesh.vertex_matrix(), faces_indices)
    #ms.apply_filter('generate_surface_reconstruction_vcg')
    #ms.add_mesh(mesh)
    
    new_file_name = "./61_resampled.ply"
    
    ms.save_current_mesh(new_file_name)
    render([new_file_name])

def test_shape():
    shape = Shape("./LabeledDB_new/Airplane/61.ply")
    shape.render()
    shape.normalize()
    shape.render()
    
test_shape()