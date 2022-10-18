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

TITLE = "Multimedia Retrieval System"
WINDOW_SIZE = (1000, 600)
BACKGROUND_COLOR = "#252526"
BUTTON_PRIMARY_COLOR = "#007acc"
TEXT_COLOR = "#ffffff"
RIGHTS_MESSAGE = "© 2022 Cristian Grosu. All rights reserved."
RIGHTS_MESSAGE_2 = "Made by Cristian Grosu, Marc Fluiter and Dmitar Angelov for Utrecht University in 2022"


import PySimpleGUI as sg
import os
from utils.QueryHandler import QueryHandler
from utils.renderer import render, render_shape_features

class GUI:
    def __init__(self):
        self.window = self.build()
        self.query = QueryHandler(log=True)

    def _get_layout(self):        
        data_dir = os.path.join(os.getcwd(), "data", "PRINCETON", "train", "animal")
        file_list_column = [
            [
                sg.Text(size= (15, 1), text = "3D Shapes Folder", background_color=BACKGROUND_COLOR),
                sg.In(size=(22, 1), enable_events=True, key="-FOLDER-"),
                sg.FolderBrowse(button_text="Browse folder", button_color=(TEXT_COLOR, BUTTON_PRIMARY_COLOR), initial_folder=data_dir, size = (22,1)),
            ],
            [
                sg.Checkbox("Show shape features", key="-SHOW FEATURES-", background_color=BACKGROUND_COLOR, text_color=TEXT_COLOR, default=False),
            ],
            [
               sg.Text("Choose a shape from list: ", background_color=BACKGROUND_COLOR),
            ],
            [   
               sg.Text(size=(50, 1), key="-TOUT-", background_color=BACKGROUND_COLOR),            
            ],
            [
                sg.Listbox(
                    values=[], enable_events=True, size=(60, 20), key="-FILE LIST-"
                )
            ],
            [
                sg.Button("Retrieve similar shapes", button_color=(TEXT_COLOR, BUTTON_PRIMARY_COLOR), size = (59,1), key = "-RETRIEVE BTN-"),
            ]
        ]
        
        retrieval_column = [
            [
                sg.Text("How many shapes would you want to retrieve?", background_color=BACKGROUND_COLOR),
                sg.Combo(
                    values=[i for i in range(1, 6)],
                    default_value=3,
                    size=(5, 1),
                    key="-RETRIEVAL NUMBER-",
                ),
            ],
            [
                sg.Text("Which distance function to use for scalars? ", background_color=BACKGROUND_COLOR, size = (40,1)),
                sg.Combo(
                    values=["L1", "L2", "Linf", "Cosine"],
                    default_value="Cosine",
                    size = (20, 1),
                    key = "-SCALAR DISTANCE-",
                )
            ],
            [
                sg.Text("Which distance function to use for histograms? ", background_color=BACKGROUND_COLOR, size = (40,1)),
                sg.Combo(
                    values=["Earth Mover Distance", "Other"],
                    default_value="Earth Mover Distance",
                    size = (20, 1),
                    key = "-HISTOGRAMS DISTANCE-",
                )
            ],
            [
                sg.Text("Which type of normalization to use for scalars? ", background_color=BACKGROUND_COLOR, size = (40,1)),
                sg.Combo(
                    values=["minmax", "z-score"],
                    default_value="minmax",
                    size = (20, 1),
                    key = "-NORMALIZATION TYPE-",
                )
            ],
            [
                sg.Text("Indicate how to weight the scalars and histograms (A3, D1, D2, D3, D4): ", background_color=BACKGROUND_COLOR, size = (40,1)),
                sg.In(size=(22, 1), enable_events=False, key="-GLOBAL WEIGHTS-", default_text="2,1,1,1,1,1"),
                sg.Text("", background_color=BACKGROUND_COLOR, text_color="red", key="-ERROR GLOBAL WEIGHTS-")
            ],
            [
                sg.Text("Indicate how to weight the scalars in the distance function: ", background_color=BACKGROUND_COLOR, size = (40,1)),
                sg.In(size=(22, 2), enable_events=False, key="-WEIGHTS-", default_text="1,1,1,1,1,1,1,1"),
                sg.Text("The scalars: (surface_area, compactness, ratio_bbox_volume, volume, ratio_ch_volume, ratio_ch_area, diameter, eccentricity) ", background_color=BACKGROUND_COLOR, size = (40,1)),
                sg.Text("", background_color=BACKGROUND_COLOR, text_color="red", key="-ERROR SCALAR WEIGHTS-")
            ],
            [
                sg.Checkbox("Show retrieved shapes features", key="-SHOW FEATURES RESPONSE-", background_color=BACKGROUND_COLOR, text_color=TEXT_COLOR, default=False),
            ],
            [sg.Text("Choose a shape from list: ", background_color=BACKGROUND_COLOR)],
            [
               sg.Column([
                    [sg.Text("Shapes list: ", background_color=BACKGROUND_COLOR)],
                    [sg.Listbox(values=[], enable_events=True, size=(25, 20), horizontal_scroll=True, no_scrollbar=True, key="-RETRIEVAL LIST-")],   
                ], background_color=BACKGROUND_COLOR),
                sg.VSeparator(),
                sg.Column([
                    [sg.Text("Distance from original: ", background_color=BACKGROUND_COLOR)],
                    [sg.Listbox(values=[], enable_events=True, size=(25, 20), no_scrollbar = True, key="-DISTANCE LIST-")],   
                ], background_color=BACKGROUND_COLOR),    
            ]
        ]
        
        
        layout = [
            [
                sg.Column(file_list_column, background_color=BACKGROUND_COLOR),
                sg.VSeparator(),
                sg.Column(retrieval_column, background_color=BACKGROUND_COLOR),
            ],
            [
                sg.Text( RIGHTS_MESSAGE, background_color=BACKGROUND_COLOR),
            ],
            [
                sg.Text( RIGHTS_MESSAGE_2, background_color=BACKGROUND_COLOR),
            ]
        ]

        return layout
    
    def build(self):
        layout = self._get_layout()
        window = sg.Window(TITLE, layout, size=WINDOW_SIZE, background_color=BACKGROUND_COLOR)
        return window
        
    def start(self):
        while True:
            event, values = self.window.read()
            
            if event == "Exit" or event == sg.WIN_CLOSED:
                break
        
            self.handle_event(event, values)
        
        self.stop()
        
    def handle_event(self, event, values):        
        # Folder name was filled in, make a list of files in the folder
        EVENTS = {
            "-FOLDER-": self.handle_select_folder_event,
            "-FILE LIST-": self.handle_file_list_event,
            "-RETRIEVE BTN-": self.handle_retrieve_event,
            "-RETRIEVAL LIST-": self.handle_retrieval_list_event,
        }
        
        EVENTS[event](event, values)

    def handle_select_folder_event(self, event, values):
        folder = values["-FOLDER-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [ f for f in file_list
                   if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith((".ply"))
                ]
        self.window["-FILE LIST-"].update(fnames)
    
    def handle_file_list_event(self, event, values):
            try:
                filename = os.path.join(
                    values["-FOLDER-"], values["-FILE LIST-"][0]
                )
                self.window["-TOUT-"].update(filename)
                if values["-SHOW FEATURES-"]:
                    filename = filename.replace(os.path.abspath("."), "").replace("data", "preprocessed")[1:] # remove the first slash
                    features = self.query.fetch_features(filename)
                    render_shape_features(filename, features)
                else:  
                    render([filename])
            except:
                pass
        
    def handle_retrieve_event(self, event, values):
        try:
            filename = os.path.join(values["-FOLDER-"], values["-FILE LIST-"][0])
            shapes_nr = values["-RETRIEVAL NUMBER-"]
            distance_scalar = values["-SCALAR DISTANCE-"]
            distance_histograms = values["-HISTOGRAMS DISTANCE-"]
            normalization_type = values["-NORMALIZATION TYPE-"]
            global_weights = values["-GLOBAL WEIGHTS-"].split(",")
            scalar_weights = values["-SCALAR WEIGHTS-"].split(",")
            
            if len(global_weights) != 6 or len(global_weights) != 2:
                self.window["-ERROR GLOBAL WEIGHTS-"].update("Global weights should be a list of either 2 or 6 elements")
                return
            
            try:
                global_weights = list(map(lambda x: float(x), global_weights))
            except Exception:
                self.window["-ERROR SCALAR WEIGHTS-"].update("Global weights should be float numbers")
                return 
            
            if  len(scalar_weights) != 8 or len(scalar_weights) != 1:
                self.window["-ERROR SCALAR WEIGHTS-"].update("Scalar weights should be a list of either 1 or 8 elements")
                return 
            
            try:
                scalar_weights = list(map(lambda x: float(x), scalar_weights))
            except Exception:
                self.window["-ERROR SCALAR WEIGHTS-"].update("Scalar weights should be float numbers")
                return 
            
            filename = filename.replace(os.getcwd(), "").replace("data", "preprocessed")[1:] # remove first slash
                        
            similar_shapes_data = self.query.find_similar_shapes(filename = filename,
                                                               target_nr_shape_to_return=shapes_nr,
                                                               distance_measure_scalars=distance_scalar,
                                                               distance_measure_histograms=distance_histograms,
                                                               normalization_type=normalization_type,
                                                               global_weights=global_weights,
                                                               scalar_weights=scalar_weights
                                                               )
            
            distances = list(map(lambda x: x[1], similar_shapes_data))
            filenames = list(map(lambda x: x[0], similar_shapes_data))
            
            self.window["-RETRIEVAL LIST-"].update(filenames)
            self.window["-DISTANCE LIST-"].update(distances)
            
            
            render(filenames)
        except:
            pass
        
    def handle_retrieval_list_event(self, event, values):
        try: 
            if values["-SHOW FEATURES RESPONSE-"]:
                features = self.query.fetch_features(values["-RETRIEVAL LIST-"][0])
                render_shape_features(values["-RETRIEVAL LIST-"][0], features)
            else:
                render(values["-RETRIEVAL LIST-"])
        except:
            pass
        
    def stop(self):
        self.window.close()

#GUI().start()