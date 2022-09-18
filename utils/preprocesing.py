# Copyright 2022 Cristian Grosu
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from utils.tools import *
from utils.db_tools import *
from utils.renderer import render
from utils.statistics import get_average_shape, plot_histogram

def prepare_database(conn):    
    
    # creating the database and the tables    
    create_db(conn)  
    
    # changing the connection to the new database
    conn = get_db_connection(os.getenv('DB_NEW_NAME'))
      
    create_table(conn)
    
    # inserting data in the database
    files = scan_files()    
    insert_data(conn, files)
        
    
def resample_outliers(conn):
    pass
    

def preprocess():
    
    # getting database connection
    db_connection = get_db_connection()
    
    # preparing the database
    # this is a costly operation, so it is recommended to run it only once
    #prepare_database(db_connection)
    
    db_connection = get_db_connection(os.getenv('DB_NEW_NAME'))
    
    # getting statistics about the database
    filename = get_average_shape(db_connection, by = "vertices_count")
    render([filename])
    
    filename = get_average_shape(db_connection, by = "faces_count")
    render([filename])
    
    # plotting histograms
    plot_histogram(db_connection, by = "vertices_count")
    plot_histogram(db_connection, by = "faces_count")
    plot_histogram(db_connection, by = "class")
    
    # resampling outliers
    resample_outliers(db_connection)
    
    # Closing the connection
    db_connection.close()