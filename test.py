# Here we test everything before implementing any code
# Like, checking the libraries API, etc.

from Shape import Shape
from utils.renderer import render, test_render_report

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


def test_shape_normalization():
    """
    _summary_ test the shape normalization
    """
    shape = Shape("./LabeledDB_new/Airplane/61.ply", log=True)
    shape.normalize()
    shape.save_mesh("./test_data/61_normalized.ply")
    render(["./LabeledDB_new/Airplane/61.ply", "./test_data/61_normalized.ply"])


def test_subsampling():
    """_summary_ testing shape sub sampling
       Sub sampling is done by using the Quadric Edge Collapse Decimation filter
       Sub sampling to a fixed number of faces
    """

    shape = Shape("./LabeledDB_new/Airplane/61.off")

    shape.resample(target_faces=2000)
    shape.save_mesh("./test_data/61_subsampled.ply")
    render(["./LabeledDB_new/Airplane/61.ply", "./test_data/61_subsampled.ply"])


def test_supersampling():
    """_summary_ testing shape super sampling
        Super sampling is done by using the Subdivision Surfaces filter
        Super sampling a certain amount of iterations until the number of faces is greater than the desired number of faces
        When greater than the desired number of faces, sub sample to the desired number of faces 
    """
    shape = Shape("./LabeledDB_new/Bird/253.ply")

    shape.resample(target_faces=21000)
    shape.save_mesh("./test_data/supersampled.ply")
    render(["./LabeledDB_new/Bird/253.ply", "./test_data/supersampled.ply"])


# render(["./LabeledDB_new/Airplane/61.ply", "./preprocessed/LabeledDB_new/Airplane/61.ply"])
# test_shape_normalization()
# test_subsampling()
# test_supersampling()

test_render_report()
