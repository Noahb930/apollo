FROM continuumio/miniconda3
WORKDIR /app
COPY environment.yml /app/environment.yml
RUN conda env create -f environment.yml
COPY . /app/
RUN conda activate apollo
ENV PATH /opt/conda/envs/apollo/bin:$PATH