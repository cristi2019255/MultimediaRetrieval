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
WINDOW_SIZE = (1150, 750)
BACKGROUND_COLOR = "#252526"
BUTTON_PRIMARY_COLOR = "#007acc"
TEXT_COLOR = "#ffffff"
RIGHTS_MESSAGE = "© 2022 Cristian Grosu. All rights reserved."
RIGHTS_MESSAGE_2 = "Made by Cristian Grosu, Marc Fluiter and Dmitar Angelov for Utrecht University in 2022"


import PySimpleGUI as sg
import os

from tsne import load_tsne_and_labels
from utils.Logger import Logger
from utils.QueryHandler import QueryHandler
from utils.renderer import render, render_shape_features
from PIL import Image

class GUI:
    def __init__(self):
        self.window = self.build()
        self.query = QueryHandler(log=True)
        self.logger = Logger()
        self.tsne_data = load_tsne_and_labels()   

    def _get_layout(self):        
        data_dir = os.path.join(os.getcwd(), "data", "PRINCETON", "train", "animal")
        histogram_distances = ["Earth Mover", "Kulback-Leibler"]
        scalar_distances = ["Cosine", "L1", "L2", "Linf", "Mahalanobis"]
        
        file_list_column = [
            [
                sg.Text(size= (15, 1), text = "3D Shapes Folder", background_color=BACKGROUND_COLOR),
                sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
                sg.FolderBrowse(button_text="Browse folder", button_color=(TEXT_COLOR, BUTTON_PRIMARY_COLOR), initial_folder=data_dir, size = (22,1)),
            ],
            [
                sg.Checkbox("Show shape features", key="-SHOW FEATURES-", background_color=BACKGROUND_COLOR, text_color=TEXT_COLOR, default=False),
            ],
            [
               sg.Text("Choose a shape from list: ", background_color=BACKGROUND_COLOR),
            ],
            [   
               sg.Text(key="-TOUT-", background_color=BACKGROUND_COLOR, size=(70, 1)),            
            ],
            [
                sg.Listbox(
                    values=[], enable_events=True, size=(70, 30), key="-FILE LIST-"
                )
            ],
            [
                sg.Button("Retrieve similar shapes", button_color=(TEXT_COLOR, BUTTON_PRIMARY_COLOR), size = (69,1), key = "-RETRIEVE BTN-"),
            ],
            [
                sg.Button("Show the database in 2D (t-sne)", button_color=(TEXT_COLOR, BUTTON_PRIMARY_COLOR), size = (69,1), key = "-TSNE BTN-"),
            ],
            [
                sg.Text("", size=(10,1), background_color=BACKGROUND_COLOR)    
            ],
            [
                sg.Text( RIGHTS_MESSAGE, background_color=BACKGROUND_COLOR),
            ],
            [
                sg.Text( RIGHTS_MESSAGE_2, background_color=BACKGROUND_COLOR),
            ]
        ]
        
        retrieval_column = [
            [
                sg.Text("How many shapes would you want to retrieve?", background_color=BACKGROUND_COLOR, size=(60,1)),
                sg.Combo(
                    values=[i for i in range(1, 6)] + ["threshold based retrieval"],
                    default_value=4,
                    size=(20, 1),
                    key="-RETRIEVAL NUMBER-",
                    enable_events=True,
                ),
            ],
            [
                sg.Checkbox("Advanced search", 
                            key="-ADVANCED OPTIONS-", 
                            background_color=BACKGROUND_COLOR, 
                            text_color=TEXT_COLOR, 
                            default=False, 
                            enable_events=True),
            ],
            [
              sg.Text("Indicate the distance threshold", 
                      background_color=BACKGROUND_COLOR, 
                      size=(60,1), 
                      visible=False, 
                      key="-THRESHOLD TEXT-"),
              sg.In(default_text="0.0001", 
                    size=(20,1),
                    visible=False,
                    key="-THRESHOLD-"),
            ],
            [
                sg.Text("Which distance function to use for scalars? ", 
                        background_color=BACKGROUND_COLOR, 
                        size = (60,1), 
                        visible=False, 
                        key = "-SCALAR DISTANCE TEXT-"
                        ),
                sg.Combo(
                    values=scalar_distances,
                    default_value=scalar_distances[0],
                    size = (20, 1),
                    visible=False,
                    key = "-SCALAR DISTANCE-",
                    enable_events=True,
                )
            ],
            [
                sg.Text("Which distance function to use for A3 histograms? ", 
                        background_color=BACKGROUND_COLOR, 
                        size = (60,1), 
                        visible=False, 
                        key = "-HISTOGRAM A3 DISTANCE TEXT-"
                        ),
                sg.Combo(
                    values=histogram_distances,
                    default_value=histogram_distances[0],
                    size = (20, 1),
                    visible=False,
                    key = "-HISTOGRAM A3 DISTANCE-",
                )
            ],
            [
                sg.Text(
                        "Which distance function to use for D1 histograms? ", 
                        background_color=BACKGROUND_COLOR, 
                        size = (60,1),
                        visible=False, 
                        key = "-HISTOGRAM D1 DISTANCE TEXT-"
                        ),
                sg.Combo(
                    values=histogram_distances,
                    default_value=histogram_distances[0],
                    size = (20, 1),
                    visible=False,
                    key = "-HISTOGRAM D1 DISTANCE-",
                )
            ],
            [
                sg.Text(
                        "Which distance function to use for D2 histograms? ", 
                        background_color=BACKGROUND_COLOR, 
                        size = (60,1),
                        visible=False, 
                        key = "-HISTOGRAM D2 DISTANCE TEXT-"
                        ),
                sg.Combo(
                    values=histogram_distances,
                    default_value=histogram_distances[0],
                    size = (20, 1),
                    visible=False,
                    key = "-HISTOGRAM D2 DISTANCE-",
                )
            ],
            [
                sg.Text("Which distance function to use for D3 histograms? ", 
                        background_color=BACKGROUND_COLOR, 
                        size = (60,1),
                        visible=False, 
                        key = "-HISTOGRAM D3 DISTANCE TEXT-"
                        ),
                sg.Combo(
                    values=histogram_distances,
                    default_value=histogram_distances[0],
                    size = (20, 1),
                    visible=False,
                    key = "-HISTOGRAM D3 DISTANCE-",
                )
            ],
            [
                sg.Text("Which distance function to use for D4 histograms? ",
                        background_color=BACKGROUND_COLOR, 
                        size = (60,1),
                        visible=False, 
                        key = "-HISTOGRAM D4 DISTANCE TEXT-"
                        ),
                sg.Combo(
                    values=histogram_distances,
                    default_value=histogram_distances[0],
                    size = (20, 1),
                    visible=False,
                    key = "-HISTOGRAM D4 DISTANCE-",
                )
            ],
            [
                sg.Text("Which type of normalization to use for scalars? ", 
                        background_color=BACKGROUND_COLOR, 
                        size = (60,1), 
                        visible=False,
                        key="-NORMALIZATION TYPE TEXT-"),
                sg.Combo(
                    values=["minmax", "z-score"],
                    default_value="z-score",
                    size = (20, 1),
                    visible=False,
                    key = "-NORMALIZATION TYPE-",
                )
            ],
            [
                sg.Text("Indicate how to weight the scalars and histograms (A3, D1, D2, D3, D4): ", 
                        background_color=BACKGROUND_COLOR, 
                        size = (60,1),
                        visible=False, 
                        key = "-GLOBAL WEIGHTS TEXT-"
                        ),
                sg.In(default_text="3,1,1,1,1,1", 
                      size=(22, 1), 
                      enable_events=False, 
                      visible=False,
                      key="-GLOBAL WEIGHTS-"),
            ],
            [
                sg.Text("", 
                        background_color=BACKGROUND_COLOR, 
                        text_color="red", 
                        visible=False,
                        key="-ERROR GLOBAL WEIGHTS-")
            ],
            [
                sg.Text("Indicate how to weight the scalars in the distance function: ", 
                        background_color=BACKGROUND_COLOR, 
                        size = (60,1), 
                        visible=False,
                        key="-SCALAR WEIGHTS TEXT-"
                        ),
                sg.In(default_text="1,1,1,1,1,1,1,1", 
                      size=(22, 2), 
                      enable_events=False, 
                      visible=False,
                      key="-SCALAR WEIGHTS-"),
            ],
            [
                sg.Text("The scalars: (surface_area, compactness, ratio_bbox_volume,", 
                        background_color=BACKGROUND_COLOR, 
                        size = (80,1), 
                        visible=False,
                        key="-SCALAR TEXT1-"),
            ],
            [
              sg.Text(" volume, ratio_ch_volume, ratio_ch_area, diameter, eccentricity)",
                      background_color=BACKGROUND_COLOR, 
                      size = (80,1), 
                      visible=False,
                      key="-SCALAR TEXT2-"),  
            ],
            [
                sg.Text("", 
                        background_color=BACKGROUND_COLOR, 
                        text_color="red",
                        visible=False,
                        key="-ERROR SCALAR WEIGHTS-")
            ],
            
            # ---------------------------------------------------------------------------------------------------
            [
                sg.Checkbox("Show retrieved shapes features", key="-SHOW FEATURES RESPONSE-", background_color=BACKGROUND_COLOR, text_color=TEXT_COLOR, default=False),
            ],
            [sg.Text("Choose a shape from list: ", background_color=BACKGROUND_COLOR)],
            [
               sg.Column([
                    [sg.Text("Shapes list: ", background_color=BACKGROUND_COLOR)],
                    [sg.Listbox(values=[], enable_events=True, size=(50, 10), horizontal_scroll=True, key="-RETRIEVAL LIST-")],   
                ], background_color=BACKGROUND_COLOR),
                sg.VSeparator(),
                sg.Column([
                    [sg.Text("Distance from original: ", background_color=BACKGROUND_COLOR)],
                    [sg.Listbox(values=[], enable_events=True, size=(25, 10), key="-DISTANCE LIST-")],   
                ], background_color=BACKGROUND_COLOR),    
            ]
        ]
        
        
        layout = [
            [
                sg.Column(file_list_column, background_color=BACKGROUND_COLOR),
                sg.VSeparator(),
                sg.Column(retrieval_column, background_color=BACKGROUND_COLOR),
            ],
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
            "-RETRIEVAL NUMBER-": self.handle_retrieval_number_event,
            "-SCALAR DISTANCE-": self.handle_scalar_distance_event,
            "-ADVANCED OPTIONS-": self.handle_advanced_options_event,
            "-TSNE BTN-": self.handle_tsne_event,
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
                    features = self.query.fetch_shape_features(filename)
                    render_shape_features(filename, features)
                else:  
                    render([filename])
            except Exception as e:
                self.logger.error("Error while loading the shape" + str(e))
    
    def handle_tsne_event(self, event, values):
        path = os.path.join("report", "tsne.png")
        if not os.path.exists(path):
            return
        img = Image.open(path)
        img.show()
        
    def handle_retrieval_number_event(self, event, values):
        retrieval = values["-RETRIEVAL NUMBER-"]        
        if retrieval == "threshold based retrieval":
            self.window["-THRESHOLD TEXT-"].update(visible=True)
            self.window["-THRESHOLD-"].update(visible=True)
        else:
            self.window["-THRESHOLD TEXT-"].update(visible=False)
            self.window["-THRESHOLD-"].update(visible=False)
    
    def handle_scalar_distance_event(self, event, values):
        visible = False if values["-SCALAR DISTANCE-"] == "Mahalanobis" else True        
        elements = ["-SCALAR WEIGHTS TEXT-", "-SCALAR WEIGHTS-", "-ERROR SCALAR WEIGHTS-", "-NORMALIZATION TYPE TEXT-", "-NORMALIZATION TYPE-", "-SCALAR TEXT1-", "-SCALAR TEXT2-"]
        self.switch_visibility(elements, visible)    
        
    def handle_advanced_options_event(self, event, values):
        elements = [
            "-SCALAR DISTANCE TEXT-", 
            "-SCALAR DISTANCE-", 
            "-SCALAR WEIGHTS TEXT-", 
            "-SCALAR WEIGHTS-",
            "-GLOBAL WEIGHTS TEXT-", 
            "-GLOBAL WEIGHTS-", 
            "-ERROR SCALAR WEIGHTS-",
            "-NORMALIZATION TYPE TEXT-",
            "-NORMALIZATION TYPE-", 
            "-SCALAR TEXT1-", 
            "-SCALAR TEXT2-", 
            "-HISTOGRAM A3 DISTANCE TEXT-",
            "-HISTOGRAM D1 DISTANCE TEXT-",
            "-HISTOGRAM D2 DISTANCE TEXT-",
            "-HISTOGRAM D3 DISTANCE TEXT-",
            "-HISTOGRAM D4 DISTANCE TEXT-",
            "-HISTOGRAM A3 DISTANCE-",
            "-HISTOGRAM D1 DISTANCE-",
            "-HISTOGRAM D2 DISTANCE-",
            "-HISTOGRAM D3 DISTANCE-",
            "-HISTOGRAM D4 DISTANCE-",
        ]
        self.switch_visibility(elements, values["-ADVANCED OPTIONS-"])
    
    def switch_visibility(self, elements, visible):
        for x in elements:
            self.window[x].update(visible=visible)
        self.window.refresh()
        
    def handle_retrieve_event(self, event, values):
        advanced = values["-ADVANCED OPTIONS-"]
        if advanced:
            self.handle_advanced_retrieve_event(event, values)
        else:
            self.handle_indexed_retrieve_event(event, values)
    
    def handle_indexed_retrieve_event(self, event, values):
        # here will be the indexed retrieval based on ann index
        try:
            filename = os.path.join(values["-FOLDER-"], values["-FILE LIST-"][0])
            filename = filename.replace(os.getcwd(), "").replace("data", "preprocessed")[1:] # remove first slash
                
            self.set_loading_state_retrieval()    
            shapes_nr = values["-RETRIEVAL NUMBER-"] 
            
            threshold = None
            if shapes_nr == "threshold based retrieval":
                threshold = float(values["-THRESHOLD-"])
            
            filenames, distances = self.query.get_similar_shapes_indexed(filename, shapes_nr, threshold)
            
            self.update_retrieved_shapes(filenames, distances)
        except Exception as e:
            self.logger.error("Error while retrieving the shape" + str(e))
            return
        
        
        
        
    def handle_advanced_retrieve_event(self, event, values):
        try:
            # ------------------ Getting the data from the GUI ------------------
            filename = os.path.join(values["-FOLDER-"], values["-FILE LIST-"][0])
            shapes_nr = values["-RETRIEVAL NUMBER-"]
            distance_scalar = values["-SCALAR DISTANCE-"]
            distance_histogram_A3 = values["-HISTOGRAM A3 DISTANCE-"]
            distance_histogram_D1 = values["-HISTOGRAM D1 DISTANCE-"]
            distance_histogram_D2 = values["-HISTOGRAM D2 DISTANCE-"]
            distance_histogram_D3 = values["-HISTOGRAM D3 DISTANCE-"]
            distance_histogram_D4 = values["-HISTOGRAM D4 DISTANCE-"]
            normalization_type = values["-NORMALIZATION TYPE-"]
            global_weights = values["-GLOBAL WEIGHTS-"].split(",")
            scalar_weights = values["-SCALAR WEIGHTS-"].split(",")
            # ---------------------------------------------------------------------
            
            
            # ------------------ Validation of the data from the GUI ------------------
            threshold_based_retrieval = False
            threshold = 0
            if shapes_nr == "threshold based retrieval":
                threshold = float(values["-THRESHOLD-"])
                threshold_based_retrieval = True
                
            if len(global_weights) != 6 and len(global_weights) != 2:
                self.window["-ERROR GLOBAL WEIGHTS-"].update("Global weights should be a list of either 2 or 6 elements")
                return
            
            try:
                global_weights = list(map(lambda x: float(x), global_weights))
            except Exception:
                self.window["-ERROR GLOBAL WEIGHTS-"].update("Global weights should be float numbers")
                return 
            
            self.window["-ERROR GLOBAL WEIGHTS-"].update("")            
            self.window.refresh()
            
            if  len(scalar_weights) != 8 and len(scalar_weights) != 1:
                self.window["-ERROR SCALAR WEIGHTS-"].update("Scalar weights should be a list of either 1 or 8 elements")
                return 
            
            try:
                scalar_weights = list(map(lambda x: float(x), scalar_weights))
            except Exception:
                self.window["-ERROR SCALAR WEIGHTS-"].update("Scalar weights should be float numbers")
                return 
            self.window["-ERROR SCALAR WEIGHTS-"].update("")
            self.window.refresh()
            
            # ---------------------------------------------------------------------
            
            # ------------------ Updating the GUI if the data is valid ------------------
            
            filename = filename.replace(os.getcwd(), "").replace("data", "preprocessed")[1:] # remove first slash
            self.set_loading_state_retrieval()
            # ----------------------------------------------------------------------------------------------------
            
            # ------------------ Retrieving the shapes ------------------
            filenames, distances = self.query.find_similar_shapes(filename = filename,
                                                               k = shapes_nr,
                                                               threshold_based_retrieval=threshold_based_retrieval,
                                                               threshold=threshold,
                                                               distance_measure_scalars=distance_scalar,
                                                               distance_measure_histogram_A3=distance_histogram_A3,
                                                               distance_measure_histogram_D1=distance_histogram_D1,
                                                               distance_measure_histogram_D2=distance_histogram_D2,
                                                               distance_measure_histogram_D3=distance_histogram_D3,
                                                               distance_measure_histogram_D4=distance_histogram_D4,
                                                               normalization_type=normalization_type,
                                                               global_weights=global_weights,
                                                               scalar_weights=scalar_weights
                                                               )
            
            distances = list(map(lambda x: round(x, 10), distances))
            # ---------------------------------------------------------------------
            self.update_retrieved_shapes(filenames, distances)
            
        except Exception as e:
            self.logger.error(e)
    
    def set_loading_state_retrieval(self):
        self.window["-RETRIEVAL LIST-"].update(["Loading..."])
        self.window["-DISTANCE LIST-"].update(["Loading..."])
        self.window.refresh()
    
    def update_retrieved_shapes(self, filenames, distances):
        # ------------------ Updating the GUI with the retrieved shapes ------------------
        if filenames != []:
            self.window["-RETRIEVAL LIST-"].update(filenames)
            self.window["-DISTANCE LIST-"].update(distances)
            if len(filenames) > 5:
                filenames = filenames[:5]
            render(filenames)
        else:
            self.window["-RETRIEVAL LIST-"].update(["No similar shapes found"])
            self.window["-DISTANCE LIST-"].update(["No similar shapes found"])
        # --------------------------------------------------------------------------------
        
    def handle_retrieval_list_event(self, event, values):
        try: 
            if values["-SHOW FEATURES RESPONSE-"]:
                features = self.query.fetch_shape_features(values["-RETRIEVAL LIST-"][0])
                render_shape_features(values["-RETRIEVAL LIST-"][0], features)
            else:
                render(values["-RETRIEVAL LIST-"])
        except Exception as e:
            self.logger.error("Error while loading the shape" + str(e))
        
    def stop(self):
        self.window.close()