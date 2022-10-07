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

A multimedia retrieval pipeline for 3D shapes.

## Geting started

In order to run the application follow the next 3 simple steps:

1) In order to get the project started first run the following command: `pip install -r requirements.txt`
2) In order to create the database run `docker-compose up`
3) Now you can run `python3 main.py` to start the application

Enjoy!

For the technical report please follow this link `https://www.overleaf.com/1781678451tvmkmtkhcrbg`

# Cloud database

In latter stages when we will have all the features extracted we will upload the database to the cloud,
so that we can use it for future steps

Database dashboard in Hekoru: `https://data.heroku.com/datastores/fd3a9a51-2a81-42fd-8891-093915414ce3#`
See `.env` for more details.

## TO update the Database locally

Run `docker-compose up` and update the database then run `docker-compose down` to shut down the container

## TODO

For Step 2:

    Problems:
        A lot of shapes from princeton data set have problems with super sampling, for now we just ignore them, however we would probably like to solve this problem and not ignore them
        Histograms before resampling and after resampling differ because of ignoring shapes when doing resampling

For Step 3:
    Test if the features are correctly implemented
    Testing througth histograms on each feature and class apartenance
    Volume of the shape seems to be correctly implemented

    Problems:
        Running time

For Step 4:
    ...
