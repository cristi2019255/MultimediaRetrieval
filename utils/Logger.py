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

from termcolor import colored

class Logger:
    def __init__(self, active=True):
        self.active = active
    
    def log(self, message):
        if self.active:
            print(f"[INFO] {message}")
    
    def warn(self, message):
        if self.active:
            print(colored(f"[WARNING] {message}", "yellow"))
    
    def error(self, message):
        if self.active:
            print(colored(f"[ERROR] {message}", "red"))
    
    def debug(self, message):
        if self.active:
            print(colored(f"[DEBUG] {message}","blue"))
            
    def success(self, message):
        if self.active:
            print(colored(f"[SUCCESS] {message}", "green"))