<div align="center">
  <img src="https://docs.kxysolutions.com/_static/logo.svg"><br>
</div>

-----------------

# KXY: Powerful Serverless Pre-Learning and Post-Learning Analysis Toolkit
[![License](https://img.shields.io/badge/license-AGPLv3%2B-blue)](https://github.com/kxytechnologies/kxy-python/blob/master/LICENSE)
[![PyPI Latest Release](https://img.shields.io/pypi/v/kxy.svg)](https://docs.kxysolutions.com/)
[![Downloads](https://img.shields.io/pypi/dm/kxy.svg)](https://docs.kxysolutions.com/)


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
and follow the instructions. To request a demo and get a trial API key, email demo@kxy.ai.

## Applications

**Performance Bound:** The `kxy` package is the first and only toolkit that provides data scientists with an information-theoretical estimate of the best performance (e.g. R² or classification error) that can be achieved on a regression or classification problem using a specific set of inputs. This allows data scientists to stop wasting time trying to improve a model that cannot be improved, or trying to train models that cannot perform to an acceptable standard.

**Model-Free Variable Selection:** Moreover, the `kxy` package uses information theory to empower data scientists to automatically discern informative inputs from inputs that are useless for solving the regression or classification problem at hand. By computing the marginal utility of each input, our input importance analyses discerns redundant inputs from complementary inputs. Our model-free approach to variable selection allows data scientists to train smaller models, faster, cheaper, and that achieve higher performance than throwing all inputs in a big model. *We support both continuous and categorical inputs for regression and classification problems*.

**Model Audit:** From understanding the marginal contribution of each variable towards the decision made by your trained regression or classification model, to detecting bias in your trained classification and regression model, the :code:`kxy` toolkit allows data scientists and decision markers to fully audit complex machine learning models.

**Model Improvement v Dataset Acquisition Prioritization:** Finally, the `kxy` toolkit can be used to guide data scientists and decision makers strike the right balance between improving production models using existing datasets and acquiring complementary datasets. Our analysis will tell you if there is still juice that your production model hasn't extracted from your datasets yet. When you are considering acquiring a new dataset, the toolkit will estimate its expected marginal impact on performance.

**Infinite Scalability:** Computations are mostly run on our infrastructure using summary statistics of your data. Scalability is near infinite thanks to AWS serverless computing, with analysis of hundreds of inputs taking minutes, if not seconds.


## Documentation
https://docs.kxysolutions.com/
