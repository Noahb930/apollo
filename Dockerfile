FROM continuumio/miniconda3
WORKDIR /app
COPY environment.yml /app/environment.yml
COPY requirements.txt /app/requirements.txt
COPY apollo.ipynb /app/apollo.ipynb
COPY apollo /app/apollo
RUN conda env create -f environment.yml python=3.5
RUN pip install -r requirements.txt
ENV PATH /opt/conda/envs/apollo/bin:$PATH
CMD voila —-port=$PORT —-no-browser apollo.ipynb