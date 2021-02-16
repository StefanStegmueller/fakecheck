#!/bin/bash

docker run -p 8888:8888 -e JUPYTER_ENABLE_LAB=yes -v $1:/home/jovyan/work jupyter/fakecheck start-notebook.sh --NotebookApp.password='sha1:5a0e54ce83e1:9fc3c47be9ec303e4a14a7a5a4c01dbdeff6a16e' 
