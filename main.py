

from distutils.errors import PreprocessError
import time
from utils.FeatureExtractor import FeatureExtractor

from termcolor import colored

def main():
    #track_progress(preprocess_data)
    track_progress(extract_features)

def track_progress(function):
    start = time.time()
    function()
    end = time.time()
    print(colored(f"[INFO] Total time for {function.__name__}:  {end - start} seconds", "green"))


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
    