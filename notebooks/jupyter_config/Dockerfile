# use to build image: docker build --rm -t jupyter/fakecheck .

FROM jupyter/tensorflow-notebook:016833b15ceb

# Install from requirements.txt file
COPY --chown=${NB_UID}:${NB_GID} requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER



