<div align="center">
  <img src="https://www.kxy.ai/theme/images/logos/logo.svg"><br>
</div>

-----------------

# KXY: A Seemless API to 10x The Productivity of Machine Learning Engineers
[![License](https://img.shields.io/badge/license-GPLv3%2B-blue)](https://github.com/kxytechnologies/kxy-python/blob/master/LICENSE)
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

KXY is free for academic use. 


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

## Other Programming Language
We plan to release friendly API client in more programming language. 

In the meantime, you can directly issue requests to our [RESTFul API](https://www.kxy.ai/reference/latest/api/index.html) using your favorite programming language. 

