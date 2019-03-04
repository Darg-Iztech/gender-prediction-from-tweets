#!/bin/sh

cp parameters_en.py parameters.py
python main.py
python modelDeleter.py

cp parameters_es.py parameters.py
python main.py
python modelDeleter.py

cp parameters_ar.py parameters.py
python main.py
python modelDeleter.py


