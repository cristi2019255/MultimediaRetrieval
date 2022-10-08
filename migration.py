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
from utils.Database import Database

def migrate_db_to_cloud(table = "shapes"):
    """_summary_ Migrates the local database to the cloud
    """
    
    cloud = Database()
    if table == "shapes":
        cloud.create_shapes_table()
        table_schema = "(id, file_name, class, faces_count, vertices_count, faces_type, bounding_box_dim_x, bounding_box_dim_y, bounding_box_dim_z, bounding_box_diagonal, features_id)"
    elif table == "features":
        cloud.create_features_table()
        table_schema = "(id, surface_area, compactness, ratio_bbox_volume, volume, ratio_ch_volume, ratio_ch_area, diameter, eccentricity, A3, D1, D2, D3, D4, shape_id)"
    else:
        return
    
    # get local database
    conn = psycopg2.connect(
                database="postgres",
                user="postgres",
                password="root",
                host="localhost",
                port="5432"
            )
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM {table}")
    rows = cur.fetchall()
    
    # creating the insert query for the cloud database
    sql = f""" INSERT INTO {table} {table_schema} VALUES """
    for row in rows:
        sql += str(row[:-1]).replace("[", "(").replace("]", ")") + ", "
    sql = sql[:-2] + ";"
    print(sql[:1000])
    
    # insert data into cloud database
    cloud.cursor.execute(sql)