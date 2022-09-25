

from distutils.errors import PreprocessError
from utils.FeatureExtractor import FeatureExtractor
from utils.tools import track_progress


def main():
    #track_progress(preprocess_data)
    track_progress(extract_features)

def preprocess_data():
    # preparing the database
    # this is a costly operation, so it is recommended to run it only once
    
    preprocessor = PreprocessError(log = True)
    preprocessor.db.prepare_db(limit=None)
    preprocessor.preprocess()    

def extract_features():
    feature_extractor = FeatureExtractor(log = True)
    feature_extractor.extract_features()
    
if __name__ == '__main__':
    main()
    