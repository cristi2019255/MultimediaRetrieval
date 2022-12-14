from utils.FeatureExtractor import FeatureExtractor
from utils.Preprocessor import Preprocessor
from utils.QueryHandler import QueryHandler

from utils.Shape import Shape
from utils.renderer import render
from utils.tools import track_progress

def main():
    #track_progress(preprocess_data)  # uncomment to preprocess data
    #track_progress(compute_statistics)  # uncomment to compute statistics
    #track_progress(extract_features) # uncomment to extract features
    track_progress(run_query) # uncomment to run query
    #track_progress(extract_feature) # uncomment to extract feature
    #track_progress(compute_statistics_feature_extraction) # uncomment to compute statistics for feature extraction
    
def preprocess_data():
    # this is a costly operation, so it is recommended to run it only once
    preprocessor = Preprocessor(log=True)
    preprocessor.db.prepare_db(limit=None)
    preprocessor.preprocess()

def compute_statistics():
    # this is a costly operation, so it is recommended to run it only once
    preprocessor = Preprocessor(log = True)
    preprocessor.compute_class_distribution()
    preprocessor.compute_statistics(type="before")
    preprocessor.compute_statistics(type="after")
    
def compute_statistics_feature_extraction():
    feature_extractor = FeatureExtractor(log=True)
    #feature_extractor.compute_statistics(type="A3", limit = 20)
    feature_extractor.compute_statistics(type="D1", limit = 20)
    #feature_extractor.compute_statistics(type="D2", limit = 20)
    #feature_extractor.compute_statistics(type="D3", limit = 20)
    #feature_extractor.compute_statistics(type="D4", limit = 20)
    #feature_extractor.compute_statistics(type="volume", limit = 20)
    
def extract_features():
    # this is a costly operation, so it is recommended to run it only once
    feature_extractor = FeatureExtractor(log=True)
    feature_extractor.extract_features()

def extract_feature(feature = 'D1'):
    feature_extractor = FeatureExtractor(log=True)
    feature_extractor.extract_feature(feature = feature)

def run_query(filename="preprocessed/LabeledDB_new/train/Airplane/61.ply"):
    query = QueryHandler(log=True)

    filenames, distances = query.find_similar_shapes(filename=filename, k = 5, normalization_type="minmax")

    search_shape = Shape(filename)
    search_shape.render()

    
    print('Similar shapes with distances: ' + str(distances))

    render(filenames)

#------------------------------------------------------------
main()
