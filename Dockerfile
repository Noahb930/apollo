FROM continuumio/miniconda3
WORKDIR /app
COPY environment.yml /app/environment.yml
COPY requirements.txt /app/requirements.txt
COPY apollo.ipynb /app/apollo.ipynb
COPY apollo /app/apollo
COPY model /app/model
RUN conda env create -f environment.yml python=3.5
ENV PATH /opt/conda/envs/apollo/bin:$PATH
RUN apt-get update && apt-get install -y g++
RUN pip install -r requirements.txt
#CMD voila —-port=$PORT —-no-browser apollo.ipynb