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

import psycopg2
import tools
import os
from dotenv import load_dotenv

def get_db_connection(db_name = None):       
    # loading the environment variables
    load_dotenv()
    print("Connecting to PostgreSQL database...")        
    try:
        conn = psycopg2.connect(    
            database = db_name if db_name else os.getenv('POSTGRES_DB'),            
            user = os.getenv('POSTGRES_USER'),
            password = os.getenv('POSTGRES_PASSWORD'),
            host = os.getenv('POSTGRES_HOST'),
            port = os.getenv('POSTGRES_PORT')
        )
        conn.autocommit = True
        return conn
    except Exception as e:
        print(e)
        return e
    
def create_db(connection):    
    # Creating a cursor object
    cursor = connection.cursor()
  
    # query to create a database         
    db_name = os.getenv('DB_NEW_NAME')
    sql = f''' CREATE database {db_name}; ''';    
    
    try:    
        cursor.execute(sql)
        print("Database has been created successfully !!");
    except Exception as e:
        print('Database already exists!')        

    

def create_table(connection):
    # Creating a cursor object
    cursor = connection.cursor()
  
    # query to create a table
    sql = '''CREATE TABLE IF NOT EXISTS shapes (
            id SERIAL PRIMARY KEY NOT NULL,
            class VARCHAR(50) NOT NULL,
            faces_count INT NOT NULL,
            vertices_count INT NOT NULL,
            faces_type VARCHAR(50) NOT NULL,            
            axis_aligned_bounding_box VARCHAR(50) NOT NULL,
            file_name VARCHAR(50) NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT NOW()
            );'''
    try:        
        cursor.execute(sql)
        print("Table has been created successfully !!");
    except Exception as e:
        print("Table already exists!")

def insert_data(connection, files):
    # Creating a cursor object
    cursor = connection.cursor()
    for key, value in files.items():
        for file in value:
            print(file)
            [faces_count, vertices_count, faces_type, axis_aligned_bounding_box] = tools.get_features(file)
            sql = f'''INSERT INTO shapes (class, faces_count, vertices_count, faces_type, axis_aligned_bounding_box, file_name) 
                VALUES ('{key}', '{faces_count}', '{vertices_count}', '{faces_type}', '{axis_aligned_bounding_box}', '{file}');'''
            try:        
                cursor.execute(sql)
                print("Data has been inserted successfully !!");
            except Exception as e:
                print("Data already exists!")