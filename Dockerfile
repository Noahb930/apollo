FROM continuumio/miniconda3
WORKDIR /app
COPY environment.yml /app/environment.yml
COPY requirements.txt /app/requirements.txt
COPY apollo.ipynb /app/apollo.ipynb
COPY apollo /app/apollo
RUN conda env create -f environment.yml
RUN pip install -r requirements.txt
# RUN echo "source activate apollo" > ~/.bashrc
ENV PATH /opt/conda/envs/apollo/bin:$PATH