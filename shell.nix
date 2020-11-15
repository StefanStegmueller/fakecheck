# usage nix-shell --command "jupyter lab"

let
  jupyter = import (builtins.fetchGit {
    url = https://github.com/tweag/jupyterWith;
    rev = "";
  }) {};

  iPython = jupyter.kernels.iPythonWith {
    name = "python";
    packages = p: with p; [ numpy
                            matplotlib
                            gensim 
                            nltk
                            spacy
                            tensorflow
                            tensorflow-tensorboard
                            Keras
                            scikitlearn
                          ];
  };

  jupyterEnvironment =
    jupyter.jupyterlabWith {
      kernels = [ iPython ];
    };
in
  jupyterEnvironment.env
