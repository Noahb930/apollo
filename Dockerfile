FROM continuumio/miniconda3

ADD ./environment.yml /tmp/environment.yml
RUN conda env create f /tmp/environment.yml
RUN conda activate apollo