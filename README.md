<!--
 Copyright 2022 Cristian Grosu
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

# Multimedia Retrieval assignment

## Description

A multimedia retrieval system for 3D shapes. Project for Utrecht University in 2022 done by Grosu Cristian, Angelov Dmitar and Fluiter Marc, all rights reserved.

## Geting started

In order to run the application follow the next 2 simple steps:

1) In order to get the project started first run the following command: `pip install -r requirements.txt`
2) Now you can run `python3 mainGUI.py` to start the application

Enjoy!

For the technical report please follow this link `https://www.overleaf.com/1781678451tvmkmtkhcrbg`

## Cloud database

The extracted features are stored in a cloud database. Please be informed that we are sharing a database connection via the
`.env` file, the database have some limitations we set in order to prevent abuses from future users. The database will become unavailable in 2023.

Database dashboard in Hekoru: `https://data.heroku.com/datastores/fd3a9a51-2a81-42fd-8891-093915414ce3#`
See `.env` for more details.

## TO update the Database locally

Run `docker-compose up` and update the database then run `docker-compose down` to shut down the container
