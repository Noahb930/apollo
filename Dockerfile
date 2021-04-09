FROM heroku/miniconda

# Install conda dependencies
RUN conda install pytorch torchvision torchaudio cpuonly -c pytorch -c conda-forge
RUN conda install rdkit -c conda-forge
RUN conda install jupyterlab ipywidgets voila -c conda-forge

# Install pip dependencies from requirements.txt
ADD ./requirements.txt /tmp/requirements.txt
RUN pip install -qr /tmp/requirements.txt