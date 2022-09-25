

from utils.preprocesing import Preprocessor

def main():
    #render()   
    preprocessor = Preprocessor(log = True)
    
    # preparing the database
    # this is a costly operation, so it is recommended to run it only once
    
    preprocessor.db.prepare_db(limit=5)
     
    # preprocessing the shapes
    preprocessor.preprocess()

    #feature_extractor = FeatureExtractor(log = True)
    
    # extracting features
    #feature_extractor.extract_features()

    #render(['./preprocessed/LabeledDB_new/Bust/307.ply', './preprocessed/LabeledDB_new/Bust/308.ply'])
    
if __name__ == '__main__':
    main()
    