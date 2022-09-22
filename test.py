# Here we test everything before implementing any code
# Like, checking the libraries API, etc.

from pymeshlab import MeshSet, Percentage
from utils.renderer import render

def test():
    
    # test for sub sampling, if too many vertices, it will be reduced
    
    file_name = "./LabeledDB_new/Airplane/61.off"
    ms = MeshSet()
    ms.load_new_mesh(file_name)
    render([file_name])
    
    # kinda works, but further investigation is needed
    ms.apply_filter("generate_resampled_uniform_mesh", mergeclosevert = True, cellsize = Percentage(4)) # kind of a subsampling
    
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
    ms.apply_filter("generate_sampling_stratified_triangle", samplenum = 10000) # kind of a super sampling
    ms.apply_filter("compute_mls_projection_apss")
    #ms.apply_filter('generate_surface_reconstruction_vcg')
    
    new_file_name = "./61_resampled.ply"
    
    ms.save_current_mesh(new_file_name)
    render([new_file_name])

test2()