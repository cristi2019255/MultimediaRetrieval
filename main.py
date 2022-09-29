

from utils.FeatureExtractor import FeatureExtractor
from utils.Preprocessor import Preprocessor
from utils.QueryHandler import QueryHandler
from utils.tools import track_progress


def main():
    track_progress(preprocess_data) # uncomment to preprocess data
    #track_progress(extract_features) # uncomment to extract features
    #track_progress(run_query) # uncomment to run query
    
def preprocess_data():
    # preparing the database
    # this is a costly operation, so it is recommended to run it only once
    
    preprocessor = Preprocessor(log = True)
    preprocessor.db.prepare_db(limit=None)
    preprocessor.preprocess()    

def extract_features():
    feature_extractor = FeatureExtractor(log = True)
    feature_extractor.extract_features()

def run_query(filename = "./LabeledDB_new/Airplane/61.off"):
    query = QueryHandler(log = True)
    query.fetch_shape(filename)

if __name__ == '__main__':
    main()
    