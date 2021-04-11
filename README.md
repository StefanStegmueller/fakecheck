To build the jupyter notebook docker image:

```
docker build --rm -t jupyter/fakecheck .
```

To run the docker containter execute *start-docker.sh*.
Give your working directory as an argument.
The jupyter lab password is: **secret**

Access jupyter lab with the URL: *127.0.0.1:8888*.

# fakecheck

## Getting Started

### Docker Environment
Docker allows an easy execution of the provided scripts by creating a virtual environment that installs all the needed dependencies. This folder contains a dockerfile that installs a Jupyter notebook environment.

#### Setup
First, install Docker as described in the documentation: https://docs.docker.com/install/

Then, build from the root folder of the repository the docker container. Run:
```
docker build --rm -t jupyter/fakecheck .
```

This builds the Jupyter container and assigns the name *jupyter/fakecheck* to it. 

Then start the container. On Linux, run:
```
docker run -p 8888:8888 -p 6006:6006 -e JUPYTER_ENABLE_LAB=yes -v "$PWD":/home/jovyan/work jupyter/fakecheck start-notebook.sh --NotebookApp.password='sha1:5a0e54ce83e1:9fc3c47be9ec303e4a14a7a5a4c01dbdeff6a16e'  
```

On Windows, run:
```
docker run -p 8888:8888 -p 6006:6006 -e JUPYTER_ENABLE_LAB=yes -v "%cd%":/home/jovyan/work jupyter/fakecheck start-notebook.sh --NotebookApp.password='sha1:5a0e54ce83e1:9fc3c47be9ec303e4a14a7a5a4c01dbdeff6a16e'  

```

The $PWD (%cd% on Windows) gets the current path and mounts it in the container to the folder /home/jovyan/work.
The jupyter lab password is: **secret**

Access jupyter lab with the URL: *127.0.0.1:8888*.

