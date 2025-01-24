#!/bin/bash

# start local ollama server in background
ollama serve &

# install the requirements
pip install -r requirements.txt

# run the api.py script
# python app.py
