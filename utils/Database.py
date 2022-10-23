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
from utils.Logger import Logger
from utils.tools import scan_files
import numpy as np

BACKUP_FOLDER = "database_backup"


class Database:
    def __init__(self, log:bool = False):
        self.logger = Logger(active=log)
        self.connection = self.get_db_connection()
        self.cursor = self.connection.cursor()
    
    def __del__ (self):
        self.close()
    
    def get_db_connection(self, db_name=None):
        # loading the environment variables
        load_dotenv()
        
        database = db_name if db_name else os.getenv('POSTGRES_DB'),
        
        self.logger.log(f"Connecting to PostgreSQL database... {database}")
        
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
            self.logger.log(f"Error while connecting to PostgreSQL database: {e}")
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
            ratio_bbox_volume FLOAT NOT NULL DEFAULT 0,
            volume FLOAT NOT NULL DEFAULT 1,
            ratio_ch_volume FLOAT NOT NULL DEFAULT 0, 
            ratio_ch_area FLOAT NOT NULL DEFAULT 0,
            diameter FLOAT NOT NULL DEFAULT 0,
            eccentricity FLOAT NOT NULL DEFAULT 0,
            A3 FLOAT[], 
            D1 FLOAT[],
            D2 FLOAT[],
            D3 FLOAT[],
            D4 FLOAT[],
            shape_id INT NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT NOW()
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
                
                self.logger.log(f"Inserting data for file: {file}")
                
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
        
        surface_area = shape.get_surface_area()
        compactness = shape.get_compactness()
        bbox_volume = shape.get_bbox_volume()
        volume = shape.get_volume()
        diameter = shape.get_diameter()
        eccentricity = shape.get_eccentricity()
        volume_convex_hull, surface_area_convex_hull = shape.get_convex_hull_measures()
        ratio_volume = volume / volume_convex_hull
        ratio_surface_area = surface_area / surface_area_convex_hull
        ratio_bbox_volume = volume / bbox_volume 
        
        sql = f'''INSERT INTO features (shape_id, surface_area, compactness, ratio_bbox_volume, volume, ratio_ch_volume, ratio_ch_area, diameter, eccentricity, A3, D1, D2, D3, D4) 
                VALUES ({shape_id}, 
                        {surface_area}, 
                        {compactness},
                        {ratio_bbox_volume},
                        {volume},
                        {ratio_volume},
                        {ratio_surface_area},
                        {diameter},
                        {eccentricity}, 
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

    def insert_feature_data(self, shape:Shape, shape_id:int, feature:str):
        FEATURES = {
                    'A3': shape.get_A3, 
                    'D1': shape.get_D1, 
                    'D2': shape.get_D2, 
                    'D3': shape.get_D3, 
                    'D4': shape.get_D4, 
                    'surface_area': shape.get_surface_area, 
                    'compactness': shape.get_compactness, 
                    'volume': shape.get_volume, 
                    'diameter': shape.get_diameter,  
                    'eccentricity': shape.get_eccentricity,
                    }
        if feature not in FEATURES.keys():
            self.logger.error(f"Feature {feature} not found")
            return
            
        feature_data = FEATURES[feature]()
        feature_data = "'" + str(feature_data).replace('[', '{').replace(']', '}') + "'"
        
        sql = f'''UPDATE features SET {feature} = {feature_data} WHERE shape_id = {shape_id};'''
        self.execute_query(sql, "update")
    
    
    def delete_shape(self, shape_file_name):
        sql = f'''DELETE FROM shapes WHERE file_name = '{shape_file_name}';'''
        self.execute_query(sql, "delete")
        
    def update_shape_feature_id(self, shape_id:int, feature_id:int):
        sql = f'''UPDATE shapes SET features_id = {feature_id} WHERE id = {shape_id};'''
        self.execute_query(sql, "update")
        
    def execute_query(self, sql, type = "insert", log = True):
        try:
            self.cursor.execute(sql)
            if log:
                if type == "delete":
                    self.logger.error(f"Data has been {type}ed successfully !!")
                else:
                    self.logger.log(f"Data has been {type}ed successfully !!")
            
        except Exception as e:
            message = {
                'insert': "Data already exists!",
                'update': "Data does not exist!",
                'create': "Table already exists!",
            }
            self.logger.error(message[type])
            self.logger.error(sql)
            self.logger.error(e)
            
    def close(self):
        self.logger.log("Closing the connection to the database...")
        self.connection.close()
        self.cursor.close()

    def get_column_data(self, by="vertices_count", table="shapes"):
        try:
            self.cursor.execute("SELECT {0} FROM {1}".format(by, table))
            rows = self.cursor.fetchall()
            data = [row[0] for row in rows]
            return data
        except Exception as e:
            self.logger.error(e)
            return None

    def get_average(self, by="vertices_count", table = "shapes"):
        try:
            data = self.get_column_data(by, table)
            avg = np.mean(data)
            
            self.logger.log(f"Average {by} is {avg}")
            
            return avg

        except Exception as e:
            self.logger.error(e)
            return None

    def prepare_db(self, limit:int = None):
        self.create_shapes_table()
        
        files = scan_files(directory="data/PRINCETON/train" ,limit = limit)
        self.insert_shape_data(files)
        
        files = scan_files(directory= "data/LabeledDB_new/train", limit = limit)
        self.insert_shape_data(files)
        
    def create_backup(self):
        """_summary_ Creating a backup for the database

        """
        os.makedirs(BACKUP_FOLDER, exist_ok=True)
        tables_to_save = ["shapes", "features"]
        for table in tables_to_save:
            rows = self.get_table_data(table=table)
            with open(os.path.join(BACKUP_FOLDER, table + ".csv"), "w") as file:
                for row in rows:
                    string = str(list(row)) + "\n"
                    file.write(string)
    
    def restore_from_backup(self):
        """_summary_ Restoring database from the backup
        """
        if not os.path.exists(BACKUP_FOLDER):
            self.logger.error("Backup folder does not exist!!")
            return
        
        self.logger.log("Restoring database from the backup...")
        
        self.logger.log("Deleting all data from the database...")
        self.cursor.execute("DROP TABLE shapes;")
        self.cursor.execute("DROP TABLE features;")
        
        self.logger.log("Creating tables...")
        self.create_features_table()
        self.create_shapes_table()
        self.logger.log("Tables created successfully!")
        
        self.logger.log("Restoring data...")
        
        try:
            for r,d,f in os.walk(BACKUP_FOLDER):
                for file in f:
                    table_name = file.replace(".csv", "")
                    self.logger.log(f"Restoring data for table {table_name}...")
                    with open(os.path.join(r, file), "r") as table_file:
                        lines = table_file.readlines()
                        for line in lines:
                            line = line.split("datetime")[0] # discard the last column with created_at timestamp
                            line = str(line)[1:-2] # remove the brackets                        
                            sanitized_line = line.replace("[", "'{").replace("]", "}'") # insert nested arrays
                            sql = f"INSERT INTO {table_name} VALUES ({sanitized_line})"
                            self.execute_query(sql, "insert", log = False)
            
            self.logger.log("Database restored from backup")
        except Exception as e:
            self.logger.error(e)
            self.logger.error("Failed to restore database from backup")            