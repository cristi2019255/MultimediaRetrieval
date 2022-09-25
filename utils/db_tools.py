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
import os
from dotenv import load_dotenv
from Shape import Shape
from utils.tools import scan_files
import numpy as np


class Database:
    def __init__(self, log:bool = False):
        self.log = log
        self.connection = self.get_db_connection()
        self.cursor = self.connection.cursor()
    
    
    def get_db_connection(self, db_name=None):
        # loading the environment variables
        load_dotenv()
        
        database = db_name if db_name else os.getenv('POSTGRES_DB'),
        
        if self.log:
            print("[INFO] Connecting to PostgreSQL database...", database)
        try:
            conn = psycopg2.connect(
                database=db_name if db_name else os.getenv('POSTGRES_DB'),
                user=os.getenv('POSTGRES_USER'),
                password=os.getenv('POSTGRES_PASSWORD'),
                host=os.getenv('POSTGRES_HOST'),
                port=os.getenv('POSTGRES_PORT')
            )
            conn.autocommit = True
            return conn
        except Exception as e:
            if self.log:
                print(f"[ERROR] {e}")
            return e

    def create_shapes_table(self):
        sql = '''CREATE TABLE IF NOT EXISTS shapes (
            id SERIAL PRIMARY KEY NOT NULL,
            file_name VARCHAR(300) NOT NULL,
            class VARCHAR(50) NOT NULL,
            faces_count INT NOT NULL,
            vertices_count INT NOT NULL,
            faces_type VARCHAR(50) NOT NULL,            
            bounding_box_dim_x FLOAT NOT NULL,
            bounding_box_dim_y FLOAT NOT NULL,
            bounding_box_dim_z FLOAT NOT NULL,
            bounding_box_diagonal FLOAT NOT NULL, 
            features_id INT NOT NULL DEFAULT 0,           
            created_at TIMESTAMP NOT NULL DEFAULT NOW()
            );'''
        self.execute_query(sql, 'create')

    def create_features_table(self):
        """_summary_ Create features table in the database
        
        Notations:
            A3: angle between 3 random vertices
            D1: distance between barycenter and random vertex
            D2: distance between 2 random vertices
            D3: square root of area of triangle given by 3 random vertices
            D4: cube root of volume of tetrahedron formed by 4 random vertices
        """
        sql = '''CREATE TABLE IF NOT EXISTS features (
            id SERIAL PRIMARY KEY NOT NULL,
            surface_area FLOAT NOT NULL DEFAULT 0,
            compactness FLOAT NOT NULL DEFAULT 0,
            bbox_volume FLOAT NOT NULL DEFAULT 0,
            diameter FLOAT NOT NULL DEFAULT 0,
            eccentricity FLOAT NOT NULL DEFAULT 0,
            A3 FLOAT[], 
            D1 FLOAT[],
            D2 FLOAT[],
            D3 FLOAT[],
            D4 FLOAT[],
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            shape_id INT NOT NULL,
            CONSTRAINT shape_id FOREIGN KEY (id) REFERENCES shapes (id)
            ); '''
        self.execute_query(sql, 'create')
    
    def get_table_data(self, table = 'shapes', columns = None):
        if columns is None:
            sql = f'''SELECT * FROM {table} '''
        else:
            sql = f'''SELECT {', '.join(columns)} FROM {table} '''
        self.cursor.execute(sql)
        rows = self.cursor.fetchall()
        return rows
       
    def insert_shape_data(self, files):
        for key, value in files.items():
            for file in value:
                
                if self.log:
                    print("[INFO] Inserting data for file: " + file)
                
                shape = Shape(file)
                [faces_count, vertices_count, faces_type, axis_aligned_bounding_box] = shape.get_features()
                [dim_x, dim_y, dim_z, diagonal] = axis_aligned_bounding_box
                sql = f'''INSERT INTO shapes (class, faces_count, vertices_count, faces_type, file_name, bounding_box_dim_x, bounding_box_dim_y, bounding_box_dim_z, bounding_box_diagonal) 
                    VALUES ('{key}', {faces_count}, {vertices_count}, '{faces_type}', '{file}', {dim_x}, {dim_y}, {dim_z}, {diagonal});'''
                self.execute_query(sql, "insert")

    def update_shape_data(self, shape:Shape, original_filename: str):
        """_summary_ Updating the data in the database after preprocessing

        Args:
            shape (Shape): _description_ The shape object
            original_filename (str): _description_ The original filename
        """
        
        [faces_count, vertices_count, faces_type, axis_aligned_bounding_box] = shape.get_features()
        [dim_x, dim_y, dim_z, diagonal] = axis_aligned_bounding_box
        filename = shape.file_name
        
        sql = f'''UPDATE shapes SET 
                faces_count = {faces_count}, 
                file_name = '{filename}',
                vertices_count = {vertices_count}, 
                faces_type = '{faces_type}',
                bounding_box_dim_x = {dim_x},
                bounding_box_dim_y = {dim_y},
                bounding_box_dim_z = {dim_z},
                bounding_box_diagonal = {diagonal}
                WHERE file_name = '{original_filename}';'''
        
        self.execute_query(sql, "update")

    def insert_features_data(self, shape:Shape, shape_id:int):
        features = {'A3': shape.get_A3(), 'D1': shape.get_D1(), 'D2': shape.get_D2(), 'D3': shape.get_D3(), 'D4': shape.get_D4()}
        A3 = "'" + str(features['A3']).replace('[', '{').replace(']', '}') + "'"
        D1 = "'" + str(features['D1']).replace('[', '{').replace(']', '}') + "'"
        D2 = "'" + str(features['D2']).replace('[', '{').replace(']', '}') + "'"
        D3 = "'" + str(features['D3']).replace('[', '{').replace(']', '}') + "'"
        D4 = "'" + str(features['D4']).replace('[', '{').replace(']', '}') + "'"
        
        
        sql = f'''INSERT INTO features (shape_id, surface_area, compactness, bbox_volume, diameter, eccentricity, A3, D1, D2, D3, D4) 
                VALUES ({shape_id}, 
                        {shape.get_surface_area()}, 
                        {shape.get_compactness()},
                        {shape.get_bbox_volume()},
                        {shape.get_diameter()},
                        {shape.get_eccentricity()}, 
                        {A3},
                        {D1},
                        {D2},
                        {D3},
                        {D4}
                    );'''
        
        get_feature_id_sql = f'''SELECT id FROM features WHERE shape_id = {shape_id};'''
        
        self.execute_query(sql, "insert")
        self.execute_query(get_feature_id_sql, "select")
        return self.cursor.fetchone()[0]
        
        
    def update_shape_feature_id(self, shape_id:int, feature_id:int):
        sql = f'''UPDATE shapes SET features_id = {feature_id} WHERE id = {shape_id};'''
        self.execute_query(sql, "update")
        
    def execute_query(self, sql, type = "insert"):
        try:
            self.cursor.execute(sql)
            if self.log:
                print(f"[INFO] Data has been {type}ed successfully !!");
        except Exception as e:
            message = {
                'insert': "[ERROR] Data already exists!",
                'update': "[ERROR] Data does not exist!",
                'create': "[ERROR] Table already exists!",
            }
            if self.log:
                print(message[type])
                print("[ERROR] " + sql)
                print(f"[ERROR] {e}")
    
    def close(self):
        if self.log:
            print("[INFO] Closing the connection to the database...")
        self.connection.close()
        self.cursor.close()

    def get_column_data(self, by="vertices_count", table="shapes"):
        try:
            self.cursor.execute("SELECT {0} FROM {1}".format(by, table))
            rows = self.cursor.fetchall()
            data = [row[0] for row in rows]
            return data
        except Exception as e:
            if self.log:
                print(f"[ERROR] {e}")
            return None

    def get_average(self, by="vertices_count", table = "sahpes"):
        try:
            data = self.get_column_data(by, table)
            avg = np.mean(data)
            
            #avg_id = np.argmin(abs(np.array(
            #    data) - avg))  # TODO: change, not the best way to get the closest value assuming id is the same as index
            #self.cursor.execute("SELECT  * FROM shapes WHERE id = {0}".format(avg_id))
            #avg_shape = self.cursor.fetchone()
            
            if self.log:
                print(f"[INFO] The average by {by} is {avg}")
                #print(f"[INFO] The average shape by {by} is: {avg_shape} ")
            return avg

        except Exception as e:
            if self.log:
                print(f"[ERROR] {e}")
            return None

    def prepare_db(self, limit:int = None):
        self.create_shapes_table()
        files = scan_files(limit = limit)
        self.insert_shape_data(files)
