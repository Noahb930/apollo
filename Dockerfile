FROM heroku/miniconda

# Add our code
ADD ./webapp /opt/webapp/
WORKDIR /opt/webapp

# Install conda dependencies fromrequirements.txt
RUN conda install pytorch torchvision torchaudio cpuonly -c pytorch -c conda-forge
RUN conda install rdkit -c conda-forge
RUN conda install jupyterlab ipywidgets voila -c conda-forge

# Install pip dependencies fromrequirements.txt
ADD ./requirements.txt /tmp/requirements.txt
RUN pip install -qr /tmp/requirements.txt