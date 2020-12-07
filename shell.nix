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
                            tensorflow-build_2
                            tensorflow-tensorboard
                            Keras
                            scikitlearn
                            pydot
                            graphviz
                          ];
  };

  jupyterEnvironment =
    jupyter.jupyterlabWith {
      kernels = [ iPython ];
    };
in
  jupyterEnvironment.env
