
from utils.renderer import render
from utils.feature_extraction import FeatureExtractor

def main():
    #render()   
    #preprocessor = Prepocesor(log = True)
    
    # preparing the database
    # this is a costly operation, so it is recommended to run it only once
    
    #preprocessor.db.prepare_db()
     
    # preprocessing the shapes
    #preprocessor.preprocess()

    feature_extractor = FeatureExtractor(log = True)
    
    # extracting features
    feature_extractor.extract_features()

    render(['./preprocessed/LabeledDB_new/Airplane/61.ply'])
    
if __name__ == '__main__':
    main()
    