# use to build image: docker build --rm -t jupyter/fakecheck .

FROM jupyter/base-notebook

# Install from requirements.txt file
COPY --chown=${NB_UID}:${NB_GID} requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

RUN python3 -m spacy download de_core_news_sm



