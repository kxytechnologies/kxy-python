.. meta::
	:description: Definition of pre-learning and post-learning in supervised learning problems
	:keywords:  Pre-Learning Explained, Post-Learning Explained, Model Audit, Model Explanation
	:http-equiv=content-language: en

.. role:: raw-html(raw)
    :format: html

I - Problem Formulation
=======================

.. admonition:: Summary

	We introduce **pre-learning** and **post-learning** problems, and discuss their importance. 


A supervised learning problem (i.e. regression or classification) aims at reliably learning an association between 
a vector of inputs :math:`x` and a label :math:`y` that is either categorical or real-valued. The association is learned using a training dataset, with the hope that, given a value of the inputs vector never seen before, the associated label can be predicted with high enough accuracy.

:raw-html:`<mark class='kxy-blue'>While the adequacy of the learned association between <i>x</i> and <i>y</i> depends solely on the model used, the overall accuracy achieved is bound by how informative the inputs are about the label.</mark>` If :math:`x` and :math:`y` are unrelated, no model, no matter 
how fancy or deep can infer :math:`y` from :math:`x`, and any attempt to do so would be futile and result in a waste of time and money. 

1 - Pre-Learning
----------------

What Is Pre-Learning?
^^^^^^^^^^^^^^^^^^^^^
A good analogy to understand **pre-learning** is that pre-learning is to supervised learning what exploration is to oil production.

It would never occur to an oil company to build a production well first, and then determine whether the site has oil by trying to extract some from the ground. Setting up an oil production site without exploration would be inefficient and very costly. The `exploration phase <https://en.wikipedia.org/wiki/Hydrocarbon_exploration>`_ ought to come first, and is critical to planning and the overall success of operations. In the exploration phase, inference techniques are used to find sites that are likely to be rich in oil, prior to, and independently from oil extraction, a field known as `exploration geophysics <https://en.wikipedia.org/wiki/Exploration_geophysics>`_.

In a supervised learning setting, **the site is the data used** to predict the business outcome, **the oil is the business value created** through the improvement of decision making, and **the oil extraction is the training of machine learning models**. Starting to train machine learning models on datasets without any expectation on what performance could be achieved, is like setting up an oil extraction site without knowing in advance that the site is rich in oil.

Selecting and training great predictive models only affects the amount of value *extracted* from the inputs, it does not change the amount of value *there is intrinsically* in those inputs. The same way the amount of oil that can be produced at a site is bound by the amount of oil accessible in the ground, the performance of a predictive model is bound by the intrinsic value that can be found in the inputs :math:`x` about the outcome of interest :math:`y`. 

.. admonition:: Definition

	**Pre-learning** is the study and selection of datasets to use to solve a supervised learning problem, prior to, and independently from any modeling.


Why Is Pre-Learning Important?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To solve a supervised learning problem, choosing inputs that are collectively insightful about the outcome of interest has as big an impact on overall performance as, if not bigger than, the machine learning model used to extract such insights.

Additionally, by quantifying the performance that can be achieved in a supervised learning problem, prior to and independently from modeling, the **pre-learning** phase empowers data scientists to know what to aim for, and to focus their efforts and resources accordingly.




2 - Post-Learning
-----------------
Once a set of informative inputs have been selected and a model has been trained, overall accuracy can be improved by either looking for a better supervised learning model, or looking for additional complementary datasets to use. Determining which action would result in the highest ROI is one of the objects of **post-learning**.

Because the learned model did not yield a satisfactory enough predictive accuracy, does not necessarily mean that a more elaborate model could do better using the same datasets. It is very possible that, although it has an unsatisfactory predictive accuracy, the learned model already factors in everything the input datasets can tell us about our label. In such an event, the only possible course of action would be 
to look for additional datasets to use.

Even then, because a new dataset is sufficiently informative about the label to predict does not necessarily mean that it can be used to improve the performance of our trained model. It is important to choose a dataset that is not only informative about the label to predict, 
but in a way that is complementary to datasets used to train the existing model.

Another object of **post-learning** is *model audit*, which entails understanding the decisions made by a trained machine learning model, and detecting any bias it might encode, to name but a couple aims.


.. admonition:: Definition

	**Post-learning** is the study and audit of a trained supervised learning model, as well as courses of action to take to improve its predictive accuracy.



