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
from shape import Shape
from utils.tools import scan_files
import numpy as np


class Database:
    def __init__(self):
        self.connection = self.get_db_connection()
        self.cursor = self.connection.cursor()
        self.create_db()

    def get_db_connection(self, db_name=None):
        # loading the environment variables
        load_dotenv()
        print("Connecting to PostgreSQL database...")
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
            print(e)
            return e

    def create_db(self):
        db_name = os.getenv('DB_NEW_NAME')
        sql = f''' CREATE database {db_name}; ''';
        try:
            self.cursor.execute(sql)
            print("Database has been created successfully !!");
        except Exception as e:
            print('Database already exists!')

        # changing the connection to the new database
        self.connection = self.get_db_connection(os.getenv('DB_NEW_NAME'))

    def create_shapes_table(self):
        sql = '''CREATE TABLE IF NOT EXISTS shapes (
            id SERIAL PRIMARY KEY NOT NULL,
            file_name VARCHAR(50) NOT NULL,
            class VARCHAR(50) NOT NULL,
            faces_count INT NOT NULL,
            vertices_count INT NOT NULL,
            faces_type VARCHAR(50) NOT NULL,            
            bounding_box_dim_x FLOAT NOT NULL,
            bounding_box_dim_y FLOAT NOT NULL,
            bounding_box_dim_z FLOAT NOT NULL,
            bounding_box_diagonal FLOAT NOT NULL,            
            created_at TIMESTAMP NOT NULL DEFAULT NOW()
            );'''
        try:
            self.cursor.execute(sql)
            print("Table has been created successfully !!");
        except Exception as e:
            print("Table already exists!")

    def insert_data(self, files):
        for key, value in files.items():
            for file in value:
                print(file)
                shape = Shape(file)
                [faces_count, vertices_count, faces_type, axis_aligned_bounding_box] = shape.get_features(file)
                [dim_x, dim_y, dim_z, diagonal] = axis_aligned_bounding_box
                sql = f'''INSERT INTO shapes (class, faces_count, vertices_count, faces_type, file_name, bounding_box_dim_x, bounding_box_dim_y, bounding_box_dim_z, bounding_box_diagonal) 
                    VALUES ('{key}', {faces_count}, {vertices_count}, '{faces_type}', '{file}', {dim_x}, {dim_y}, {dim_z}, {diagonal});'''
                try:
                    self.cursor.execute(sql)
                    print("Data has been inserted successfully !!");
                except Exception as e:
                    print("Data already exists!")

    def update_data(self, filename):
        shape = Shape(filename)
        [faces_count, vertices_count, faces_type, axis_aligned_bounding_box] = shape.get_features(filename)
        [dim_x, dim_y, dim_z, diagonal] = axis_aligned_bounding_box
        sql = f'''UPDATE SET 
                faces_count = {0}, 
                vertices_count = {1}, 
                faces_type = {2},
                bounding_box_dim_x = {3},
                bounding_box_dim_y = {4},
                bounding_box_dim_z = {5},
                bounding_box_diagonal = {6},
                FROM shapes WHERE file_name = '{filename}';'''.format(
            faces_count,
            vertices_count,
            faces_type,
            dim_x,
            dim_y,
            dim_z,
            diagonal
        )
        try:
            self.cursor.execute(sql)
            print("Data has been updated successfully !!");
        except Exception as e:
            print("Data already exists!")

    def close(self):
        self.connection.close()
        self.cursor.close()

    def get_column_data(self, by="vertices_count"):
        try:
            self.cursor.execute("SELECT {0} FROM shapes".format(by))
            rows = self.cursor.fetchall()
            data = [row[0] for row in rows]
            return data
        except Exception as e:
            print(e)
            return None

    def get_average_shape(self, by="vertices_count"):
        try:
            data = self.get_column_data(by)
            avg = np.mean(data)
            print(f"The average by {by} is {avg}")
            avg_id = np.argmin(abs(np.array(
                data) - avg))  # TODO: change, not the best way to get the closest value assuming id is the same as index
            self.cursor.execute("SELECT  * FROM shapes WHERE id = {0}".format(avg_id))
            avg_shape = self.cursor.fetchone()
            print(f"The average shape by {by} is: {avg_shape} ")
            return avg_shape
        except Exception as e:
            print(e)
            return None

    def prepare_db(self):
        self.create_shapes_table()
        files = scan_files()
        self.insert_data(files)
