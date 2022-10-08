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

# (a) Find out what is the average shape in the database (in terms of vertex and face counts); (b) If there are
# significant outliers from this average (e.g. shapes having many, or few, vertices or cells). The best way to do
# this is to show a histogram counting how many shapes are in the database for every range of the property of
# interest (e.g., number of vertices, number of faces, shape class).


import matplotlib.pyplot as plt

plt.style.use(['science', 'no-latex'])  # style from SciencePlots


def plot_histogram(data, title="Histogram of vertex counts", bins = 20):
    plt.figure(figsize=(8, 6), dpi=80)
    plt.clf()
    plt.hist(data, bins=bins, edgecolor='black', rwidth=0.85, color='#607c8e')
    plt.title(title)

    
    plt.xticks(rotation=60, fontsize=10, va='top', ha='center')

    file_name = title.replace(" ", "_") + ".png"
    plt.savefig(f"./report/histograms/{file_name}")
    plt.close()
