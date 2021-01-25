<div align="center">
  <img src="https://www.kxy.ai/theme/images/logos/logo.svg"><br>
</div>

-----------------

# KXY: A Powerful Serverless Analysis Toolkit That Takes *Trial And Error* Out of Machine Learning Projects
[![License](https://img.shields.io/badge/license-AGPLv3%2B-blue)](https://github.com/kxytechnologies/kxy-python/blob/master/LICENSE)
[![PyPI Latest Release](https://img.shields.io/pypi/v/kxy.svg)](https://www.kxy.ai/)
[![Downloads](https://pepy.tech/badge/kxy)](https://www.kxy.ai/)


## Documentation
https://www.kxy.ai/reference/


## Installation
From PyPi:
```Bash
pip install kxy
```
From GitHub:
```Bash
git clone https://github.com/kxytechnologies/kxy-python.git & cd ./kxy-python & pip install .
```
## Authentication
All heavy-duty computations are run on our serverless infrastructure and require an API key. To configure the package with your API key, run 
```Bash
kxy configure
```
and follow the instructions. To get an API key you need an account; you can sign up for a free trial [here](https://www.kxy.ai/signup/). You'll then be automatically given an API key which you can find [here](https://www.kxy.ai/portal/profile/identity/).


## Docker
The Docker image [kxytechnologies/kxy](https://hub.docker.com/repository/docker/kxytechnologies/kxy) has been built for your convenience, and comes with anaconda, auto-sklearn, and the kxy package.

To start a Jupyter Notebook server from a sandboxed Docker environment, run
```Bash
docker run -i -t -p 5555:8888 kxytechnologies/kxy:latest /bin/bash -c "kxy configure <YOUR API KEY> && /opt/conda/bin/jupyter notebook --notebook-dir=/opt/notebooks --ip='*' --port=8888 --no-browser --allow-root --NotebookApp.token=''"
```
where you should replace `<YOUR API KEY>` with your API key and navigate to [http://localhost:5555](http://localhost:5555) in your browser. This docker environment comes with [all examples available on the documentation website](https://www.kxy.ai/reference/latest/examples/).

To start a Jupyter Notebook server from an existing directory of notebooks, run
```Bash
docker run -i -t --mount src=</path/to/your/local/dir>,target=/opt/notebooks,type=bind -p 5555:8888 kxytechnologies/kxy:latest /bin/bash -c "kxy configure <YOUR API KEY> && /opt/conda/bin/jupyter notebook --notebook-dir=/opt/notebooks --ip='*' --port=8888 --no-browser --allow-root --NotebookApp.token=''"
```
where you should replace `</path/to/your/local/dir>` with the path to your local notebook folder and navigate to [http://localhost:5555](http://localhost:5555) in your browser.


## Applications

### Higher ROI Machine Learning Projects

The `kxy` package utilizes information theory to takes *trial and error* out of machine learning projects. 

From the get-go, the **achievable performance analysis** of the `kxy` package tells data scientists whether their datasets are sufficiently informative to achieve a performance (e.g. <img src="https://render.githubusercontent.com/render/math?math=R^2">, maximum log-likelihood, and classification error) to their liking in a classification or regression problem, and if so what is the best performance that can be achieved using said datasets. *No need to train tens of models to know what performance can be achieved*.

The **model-free variable selection analysis** provided by the `kxy` package allows data scientists to train smaller models, faster, cheaper, and to achieve a higher performance than throwing all inputs in a big model or proceeding by trial-and-error.

Once a model has been trained, the `kxy` **improvability analysis** quantifies the extent to which the trained model can be improved without resorting to additional features. This allows data scientists to focus their modeling efforts on high ROI initiatives. *No need to implement tens of fancy models on specialized hardware to see whether a trained model can be improved*.

When a classification or regression model has successfully extracted all the value in using the features to predict the label, the `kxy` **dataset valuation analysis** allows data scientists to quickly quantify the performance increase (e.g. <img src="https://render.githubusercontent.com/render/math?math=R^2">, maximum log-likelihood, and classification error) that a new dataset may bring about. *No need to train or retrain tens of models with the new datasets to see whether the production model can be improved*.


### Model Audit

From **understanding** the marginal contribution of each variable towards the decision made by **a black-box regression or classification model**, to **detecting bias** in your trained classification and regression model, the `kxy` toolkit allows data scientists and decision markers to fully **audit complex machine learning models**.


### Modern Financial Machine Learning

From **non-Gaussian** and **memory-robust** risk analysis, to **alternative datasets valuation** the `kxy` toolkit propels quants from the age of Gaussian distributions/linear regression/LASSO/Ridge/Random Forest into the age of modern machine learning, rigorously and cost-effectively.
