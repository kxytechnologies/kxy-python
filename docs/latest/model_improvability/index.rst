.. meta::
   :description: How to use the kxy Python package to quantify by how much the performance of a trained supervised learning may be improved in a model-driven fashion (i.e. by simply looking for a better model, and without resorting to additional explanatory variables), or in a data-driven fashion (i.e. how much incremental value a specific new set of explanatory variables may bring about).
   :keywords: Model-Driven Improvability, Data-Driven Improvability, Post-Learning, KXY.
   :http-equiv=content-language: en

   
===================
Model Improvability
===================
Estimation of the amount by which the performance of a trained supervised learning model can be increased, either in a model-driven fashion, or a data-driven fashion.

Model-Driven Improvability
--------------------------

.. autofunction:: kxy.post_learning.improvability.model_driven_improvability


Data-Driven Improvability
-------------------------

.. autofunction:: kxy.post_learning.improvability.data_driven_improvability
