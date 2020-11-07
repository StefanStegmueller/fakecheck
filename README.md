# fakecheck

## Getting Started

### Jupyter Notebook Environment

Install nix package manager as described here: https://nixos.org/download.html#nix-quick-install

Then run the following command to start the Juypter server:
```
nix-shell --command "jupyter lab" 
```

Connect to http://localhost:8888 to access the Jupyter lab. 

### Docker Environment
Docker allows an easy execution of the provided scripts by creating a virtual environment that installs all the needed dependencies. This folder contains a dockerfile that installs a Python 3.8 environment, TensorFlow, Keras and more.

#### Setup
First, install Docker as described in the documentation: https://docs.docker.com/install/

Then, build from the root folder of the repository the docker container. Run:
```
docker build . -t fakecheck 
```

This builds the Python 3.8 container and assigns the name *fakecheck* to it. 

Then start the container. On Linux, run:
```
docker run -it -v "$PWD":/src/ fakecheck bash 
```

On Windows, run:
```
docker run -it -v "%cd%":/src/ fakecheck bash
```

The $PWD (%cd% on Windows) gets the current path and mounts it in the container to the folder /src. You can also pass other paths on your hosting system that should be used in the container:
```
docker run -it -v /my/path/on/host/system:/src/ fakecheck bash
```
