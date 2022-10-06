from utils.FeatureExtractor import FeatureExtractor
from utils.Preprocessor import Preprocessor
from utils.QueryHandler import QueryHandler

from Shape import Shape
from utils.tools import track_progress


def main():
    # convert_to_ply(directory="./data/PRINCETON/train")
    track_progress(preprocess_data)  # uncomment to preprocess data
    # track_progress(extract_features) # uncomment to extract features
    # track_progress(run_query) # uncomment to run query


def preprocess_data():
    # preparing the database
    # this is a costly operation, so it is recommended to run it only once

    preprocessor = Preprocessor(log=True)
    preprocessor.db.prepare_db(limit=None)
    preprocessor.preprocess()


def extract_features():
    feature_extractor = FeatureExtractor(log=True)
    feature_extractor.extract_features()


def run_query(filename="./preprocessed/LabeledDB_new/Airplane/61.ply"):
    query = QueryHandler(log=True)
    query.fetch_shape(filename)
    similar_shapes_data = query.find_similar_shapes(n = 7, distance_measure = 'Cosine Distance', normalization = 'Standard Score Normalization')


    print('Original shape:')

    search_shape = Shape(filename)
    search_shape.render()

    for similar_shape_data in similar_shapes_data:

        shape_id, filename_similar_shape, distance = similar_shape_data

        print('Similar shape with distance: ' + str(distance))

        similar_shape = Shape(filename_similar_shape)
        similar_shape.render()


if __name__ == '__main__':
    main()
