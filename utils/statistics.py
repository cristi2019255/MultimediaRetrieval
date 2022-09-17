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

# (a) Find out what is the average shape in the database (in terms of vertex and face counts);
# (b) If there are significant outliers from this average (e.g. shapes having many, or few, vertices or cells).
# The best way to do this is to show a histogram counting how many shapes are in the database for every range of the property of interest 
# (e.g., number of vertices, number of faces, shape class).


import numpy as np
import matplotlib.pyplot as plt

def get_data(connection, by = "vertices_count"):    
    cur = connection.cursor()
    try:
        cur.execute("SELECT  id,{0} FROM shapes".format(by))
        rows = cur.fetchall()            
        data = [row[1] for row in rows]        
        return data
    
    except Exception as e:
        print(e)
        return None

def get_average_shape(connection, by = "vertices_count"):    
    cur = connection.cursor()
    try:
        data = get_data(connection, by)    
        avg = np.mean(data)        
        print(f"The average by {by} is {avg}")        
        avg_id = np.argmin(abs(np.array(data) - avg))
        cur.execute("SELECT  * FROM shapes WHERE id = {0}".format(avg_id))
        avg_shape = cur.fetchone()
        print(f"The average shape by {by} is: {avg_shape} ")
        return avg_shape[1]
    except Exception as e:
        print(e)
        return None

def plot_histogram(connection, by = "vertices_count"):
    data = get_data(connection, by)
    plt.hist(data)
    plt.title(f"Histogram of {by}")    
    plt.show()

