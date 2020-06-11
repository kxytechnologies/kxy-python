.. meta::
   :description: The Python API to KXY, the first and only AutoML platform for pre-learning and post-learning
   :keywords: AutoML, Pre-Learning, Post-Learning, KXY API, KXY Technologies



A Powerful Serverless Pre-Learning & Post-Learning Analysis Toolkit
===================================================================
.. image:: https://img.shields.io/badge/license-AGPLv3%2B-blue
   :alt: License
   :target: https://www.gnu.org/licenses/agpl-3.0.en.html
.. image:: https://img.shields.io/pypi/v/kxy.svg
   :alt: PyPI Latest Release
   :target: https://pypi.org/project/kxy/
.. image:: https://img.shields.io/pypi/dm/kxy.svg
   :alt: Downloads
   :target: https://github.com/kxytechnologies/kxy-python/



**Performance Bound:** The :code:`kxy` package is the first and only toolkit that provides data scientists with an information-theoretical estimate of the best performance (e.g. :math:`R^2` or classification error) that can be achieved on a regression or classification problem using a specific set of inputs. This allows data scientists to stop wasting time trying to improve a model that cannot be improved, or trying to train models that cannot perform to an acceptable standard.

**Model-Free Variable Selection:** Moreover, the :code:`kxy` package uses information theory to empower data scientists to automatically discern informative inputs from inputs that are useless for solving the regression or classification problem at hand. By computing the marginal utility of each input, our input importance analyses discerns redundant inputs from complementary inputs. Our model-free approach to variable selection allows data scientists to train smaller models, faster, cheaper, and that achieve higher performance than throwing all inputs in a big model. *We support both continuous and categorical inputs for regression and classification problems*.

**Model Audit:** From understanding the marginal contribution of each variable towards the decision made by your trained regression or classification model, to detecting bias in your trained classification and regression model, the :code:`kxy` toolkit allows data scientists and decision markers to fully audit complex machine learning models.

**Model Improvement v Dataset Acquisition Prioritization:** Finally, the :code:`kxy` toolkit can be used to guide data scientists and decision makers strike the right balance between improving production models using existing datasets and acquiring complementary datasets. Our analysis will tell you if there is still juice that your production model hasn't extracted from your datasets yet. When you are considering acquiring a new dataset, the toolkit will estimate its expected marginal impact on performance.

**Infinite Scalability:** Computations are mostly run on our infrastructure using summary statistics of your data. Scalability is near infinite thanks to AWS serverless computing, with analysis of hundreds of inputs taking minutes, if not seconds.



==============
Request A Demo
==============
To request a demo and get a trial API key, email demo@kxy.ai.




.. toctree::
	:hidden:
	:caption: QUICKSTART

	latest/introduction/getting_started/getting_started


.. toctree::
	:hidden:
	:caption: EXAMPLES

	latest/notebooks/classification/index


.. toctree::
	:hidden:
	:caption: THEORETICAL FOUNDATION

	latest/introduction/memoryless/index

	latest/introduction/time_series/index


.. toctree::
	:hidden:
	:caption: CODE DOCUMENTATION

	latest/estimation/index

	latest/asset_management/index

	latest/classification/index

	latest/regression/index

	latest/pandas/index

	latest/utilities/index

	latest/api/index


.. toctree::
	:hidden:
	:caption: INDEX

	latest/index/index
