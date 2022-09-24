
from utils.feature_extraction import FeatureExtractor
from utils.preprocesing import Prepocesor


def main():
    #render()   
    preprocessor = Prepocesor(log = True)
    feature_extractor = FeatureExtractor(log = True)
    
    # preparing the database
    # this is a costly operation, so it is recommended to run it only once
    
    #preprocessor.db.prepare_db()
     
    # preprocessing the shapes
    #preprocessor.preprocess()

    # extracting features
    #feature_extractor.extract_features()

if __name__ == '__main__':
    main()
    