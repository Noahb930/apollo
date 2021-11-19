# Apollo
## Background
* Apollo works by combining a **[Self-Attention Graph Pooling](https://arxiv.org/pdf/1904.08082.pdf) ** network architecture with the **[LambdaRank](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/lambdarank.pdf)** loss function.

## Installation
There are two ways to install the needed dependencies
### Installation with Conda
```bash
conda env create -f environment.yml`
conda activate apollo`
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

Y=Bottom + (Top-Bottom)/(1+10^((LogEC50 - LogX * HillSlope))