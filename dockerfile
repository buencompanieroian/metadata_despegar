FROM continuumio/miniconda3

ADD eval_test_datasets.py /

# load in the environment.yml file
ADD python_3_enviroment.yml /

# create the environment
RUN conda env create -f python_3_enviroment.yml
ENV PATH /opt/conda/envs/py3env/bin:$PATH
RUN /bin/bash -c "source activate py3env"

CMD [ "python", "./eval_test_datasets.py" ]

