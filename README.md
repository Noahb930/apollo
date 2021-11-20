# Apollo
## Background
* The advent of Graph Convolutional Networks has allowed for machine learning models to propagate information across the inherent structure of a mollecule.
* This has massive implications for the future of machine learning based drug discovery platforms for property prediction and candidate identification.
* Unfortunately, such models traditionally require massive amounts of data not available to the typical lab.
* However, Apollo was used to demonstrate that such models can be successfully trained on a dataset that is multiple times as small, if the problem is treated as an information retreival task.
* Apollo works by combining a **[Self-Attention Graph Pooling](https://arxiv.org/pdf/1904.08082.pdf)** network architecture with the **[LambdaRank](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/lambdarank.pdf)** loss function.
* The implementation in this repository will allow anyone to train a Graph Attention Network with their specifications and dataset and no additional code
* You can learn more about the project in [this article](https://www.freethink.com/technology/drug-discovery) by Freethink and [this video](https://www.youtube.com/watch?v=q6hb7lxSglg) by the Regeneron Science Talent Search,

## Installation
There are two ways to install the needed dependencies
### Installation with Conda
```bash
conda env create -f environment.yml
conda activate apollo
pip install -r requirements.txt
```
### Installation with Docker
```bash
docker build -t .
```
## Launch
Launch the interactive notebook by running: 
```bash
voila apollo.ipynb
```
