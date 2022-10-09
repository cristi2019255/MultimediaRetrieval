# Here we test everything before implementing any code
# Like, checking the libraries API, etc.

from Shape import Shape
from utils.renderer import render

def test_shape_normalization():
    """
    _summary_ test the shape normalization
    """
    shape = Shape("./data/LabeledDB_new/train/Airplane/61.ply", log=True)
    shape.normalize()
    shape.save_mesh("./test_data/61_normalized.ply")
    render(["./data/LabeledDB_new/train/Airplane/61.ply", "./test_data/61_normalized.ply"])


def test_subsampling():
    """_summary_ testing shape sub sampling
       Sub sampling is done by using the Quadric Edge Collapse Decimation filter
       Sub sampling to a fixed number of faces
    """

    shape = Shape("./data/LabeledDB_new/train/Airplane/61.off")

    shape.resample(target_faces=2000)
    shape.save_mesh("./test_data/61_subsampled.ply")
    render(["./data/LabeledDB_new/train/Airplane/61.ply", "./test_data/61_subsampled.ply"])


def test_supersampling():
    """_summary_ testing shape super sampling
        Super sampling is done by using the Subdivision Surfaces filter
        Super sampling a certain amount of iterations until the number of faces is greater than the desired number of faces
        When greater than the desired number of faces, sub sample to the desired number of faces 
    """
    shape = Shape("./data/LabeledDB_new/train/Bird/253.ply")

    shape.resample(target_faces=21000)
    shape.save_mesh("./test_data/supersampled.ply")
    render(["./data/LabeledDB_new/train/Bird/253.ply", "./test_data/supersampled.ply"])

def test_volume(): 
    shape = Shape("./data/LabeledDB_new/train/Airplane/61.ply")  
    print(shape.get_convex_hull_measures())
    print(shape.get_volume())
    print(shape.get_surface_area())    

def test_super_sampling():
    filename = "data/PRINCETON/train/plant/m1044.ply"
    #filename = "data/PRINCETON/train/plant/m1044.ply"
    render([filename])
    shape = Shape(filename, log=True)
    shape.resample(target_faces=5000)
    shape.save_mesh("./test_data/supersampled.ply")
    render([filename, "./test_data/supersampled.ply"])


def test_convex_hull():
    filename = "preprocessed/PRINCETON/train/furniture/m861.ply"
    shape = Shape(filename, log=True)
    print(shape.get_convex_hull_measures())
    

test_convex_hull()    
#render(["data/PRINCETON/train/furniture/m855.ply"])
# test_shape_normalization()
# test_subsampling()
# test_supersampling()

#test_render_report()
#test_super_sampling()