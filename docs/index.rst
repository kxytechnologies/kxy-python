.. meta::
   :description: Documentation of the KXY Lean AutoML platform.
   :keywords: AutoML, Lean AutoML, KXY AutoML, Pre-Learning, Post-Learning
   :http-equiv=content-language: en



A Powerful Serverless Analysis Toolkit That Takes *Trial And Error* Out of Machine Learning Projects
====================================================================================================
.. image:: https://img.shields.io/badge/license-GPLv3%2B-blue
   :alt: License
   :target: https://www.gnu.org/licenses/agpl-3.0.en.html
.. image:: https://img.shields.io/pypi/v/kxy.svg
   :alt: PyPI Latest Release
   :target: https://pypi.org/project/kxy/
.. image:: https://pepy.tech/badge/kxy
   :alt: Downloads
   :target: https://github.com/kxytechnologies/kxy-python/


==============
Get An API Key
==============
To get an API key, simply open an account with us `here <https://www.kxy.ai/portal>`_. As soon as you have an account, you may retrieve your API key `here <https://www.kxy.ai/portal/profile/identity/>`_.


=================================================================
Boost Your The Productivity Of Your ML Teams Tenfold With Lean ML
=================================================================
The :code:`kxy` package utilizes information theory to take *trial and error* out of machine learning projects. 

-------------------
Project Feasibility
-------------------
From the get-go, the **data valuation** analysis of the :code:`kxy` package tells data scientists whether their datasets are sufficiently informative to achieve a performance (e.g. :math:`R^2`, RMSE, maximum log-likelihood, and classification error) to their liking in a classification or regression problem, and if so what is the best performance that can be achieved using said datasets. *Only spend time and compute resources on a project once you know it can yield the desired business impact*.

----------------------------------------
Automatic (Model-Free) Feature Selection
----------------------------------------
The **model-free variable selection** analysis provided by the :code:`kxy` package allows data scientists to train smaller models, faster, cheaper, and to achieve a higher performance than throwing all inputs in a big model or proceeding by trial-and-error.


---------------------------------------
Production Model Improvability Analyses
---------------------------------------
**Data-Driven Improvability:** Once a model has been trained, the :code:`kxy` *model-driven improvability* analysis quantifies the extent to which the trained model can be improved without resorting to additional features. This allows data scientists to focus their modeling efforts on high ROI initiatives. *Only throw the might of your ML team and platform at improving the fit of your production model when you know it can be improved. Never again will you spend weeks, if not months, and thousands of dollars in cloud compute, implementing the latest models on specialized hardware to improve your production model, only to find out its fit cannot be improved*.

**Model-Driven Improvability:** Once the fit of a production model is optimal (i.e. it has successfully extracted all the value in using a given set features to predict the label), the :code:`kxy` *data-driven improvability* allows data scientists to quickly quantify the performance increase (e.g. :math:`R^2`, RMSE, maximum log-likelihood, and classification error) that a new dataset may bring about. *Only retrain models with additional features when you know they can bring about a meaningful performance boost*.


------------------------------------------------------
Reducing Time and Resources Spent on Overfitted Models
------------------------------------------------------
We provide callbacks in the major Python machine learning libraries that will terminate training when the running best performance seems unrealistic (i.e. far exceeds the theoretical-best achievable). Our callbacks allow saving time and compute resources on models that we can reliably determine will overfit once fully trained, well before training ends. This is a cost-effective alternative to cross-validation.





.. toctree::
	:hidden:
	:caption: QUICKSTART

	latest/quickstart/getting_started


.. toctree::
	:hidden:
	:caption: EXAMPLES

	latest/examples/index


.. toctree::
	:hidden:
	:caption: THEORETICAL FOUNDATION

	latest/theoretical_foundation/memoryless/index

	latest/theoretical_foundation/memoryful/index


.. toctree::
	:hidden:
	:caption: PYTHON CODE DOCUMENTATION

	latest/data_valuation/index

	latest/variable_selection/index

	latest/learning/index

	latest/model_explanation/index

	latest/model_improvability/index


.. toctree::
	:hidden:
	:caption: MISCELLANEOUS

	latest/data_transfer/index

	latest/pandas/index


.. toctree::
	:hidden:
	:caption: OTHER LANGUAGES

	latest/api/index


.. toctree::
	:hidden:
	:caption: INDEX

	latest/index/index
