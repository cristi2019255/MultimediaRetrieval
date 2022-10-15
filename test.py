# Here we test everything before implementing any code
# Like, checking the libraries API, etc.

from Shape import Shape
from utils.renderer import render, render_shape_with_features
from pymeshlab import MeshSet
import matplotlib.pyplot as plt
import math
import numpy as np

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
    

def create_sphere(file_name = './test_data/sphere.ply', radius = 1):
    ms = MeshSet()
    ms.create_sphere(radius = radius, subdiv = 4)
    ms.save_current_mesh(file_name)    
    render([file_name])

def create_torus(file_name = './test_data/torus.ply'):
    ms = MeshSet()
    ms.create_torus(hradius = 1, vradius = 0.5, hsubdiv = 24, vsubdiv = 24)
    ms.save_current_mesh(file_name)
    render([file_name])

def create_cylinder(file_name = './test_data/cylinder.ply'):
    ms = MeshSet()
    ms.create_cone(r0 = 1, r1 = 1, h = 5, subdiv = 50)
    ms.save_current_mesh(file_name)
    render([file_name])

def test_shape_features(file_name = "./test_data/torus.ply"):
    shape = Shape(file_name=file_name, log=True)
    volume = shape.get_volume()
    diameter = shape.get_diameter()
    surface_area = shape.get_surface_area()
    compactness = shape.get_compactness()
    eccentricity = shape.get_eccentricity()
    bbox_volume = shape.get_bbox_volume()
    chull_volume, chull_surface_area = shape.get_convex_hull_measures()
    render_shape_with_features(file_name, volume = volume, diameter = diameter, 
                               surface_area = surface_area, compactness = compactness, 
                               eccentricity = eccentricity, bbox_volume = bbox_volume, 
                               chull_volume = chull_volume, chull_surface_area = chull_surface_area)



def show_shape_hist_features(file_name = "./test_data/sphere.ply", type = "D4"):
    shape = Shape(file_name=file_name, log=True)
    dict = {"A3": shape.get_A3, "D1": shape.get_D1, "D2": shape.get_D2, "D3": shape.get_D3, "D4": shape.get_D4}
    upper_bounds = {
            "A3": math.pi,
            "D1": math.sqrt(3),
            "D2": math.sqrt(3),
            "D3": (math.sqrt(3) / 2) ** (1/2),
            "D4": (1 / 3) ** (1/3)
    }
    
    upper_bound_x = upper_bounds[type]
    data = dict[type]()
    plt.figure(figsize=(10, 10))
    plt.clf()
    plt.title(f"Signature for {file_name} for feature {type}")
        
    file_name = file_name.replace(".ply", ".png")
    x = np.linspace(0, upper_bound_x, len(data))
    plt.plot(x, data)
    plt.xlim(0, upper_bound_x)
    
    #plt.savefig(file_name)
    plt.show()
    plt.close()
    
#test_convex_hull()    
#render(["data/PRINCETON/train/furniture/m855.ply"])
# test_shape_normalization()
# test_subsampling()
# test_supersampling()

#test_render_report()
#test_super_sampling()
#create_sphere()
#create_torus()
#create_cylinder()
#test_shape_features("./test_data/torus.ply")
#show_shape_hist_features()