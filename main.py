
from utils.preprocesing import Prepocesor


def main():
    #render()   
    preprocessor = Prepocesor(log = True)
    
    
    # preparing the database
    # this is a costly operation, so it is recommended to run it only once
    
    #preprocessor.db.prepare_db() 
    
    
    preprocessor.preprocess()

if __name__ == '__main__':
    main()
    