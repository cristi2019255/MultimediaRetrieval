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

...

## Geting started

In order to get the project started first run the following command: `pip install -r requirements.txt`
In order to run up the database run `docker-compose up`

For the technical report I've set up a latex file in overleaf, please follow this link `https://www.overleaf.com/1781678451tvmkmtkhcrbg`

## TODO

Download the benchmark dataset from princeton and put it here (the website was down)

For Step 2:

Problems:
Re-scaling (giving bounding box of dimension 2 in any direction), not sure if it is working correct
Sub/Supersampling don't like how super sampling is implemented now,furthermore when super sampling
for some shapes the filter does not work, should we just discard them? (See report folder, histograms)
