.. meta::
	:description: The theoretical foundation of the KXY Lean AutoML platform.
	:keywords:  Pre-Learning, Post-Learning, Maximum-Entropy Principle, Input Importance, Feature Importance, KXY, Model Explanation, Dataset Valuation, Input Importance, Feature Importance, Model Suboptimality, Model Optimality.
	:http-equiv=content-language: en
***********************
Memoryless Observations
***********************
In this section we provide the intuition for, and the theoretical underpinnings of our solutions to the **pre-learning** and **post-learning** problems in data science life cycles, when input and output variables do not exhibit temporal structures, or their temporal structures are of negligible importance. 

When temporal structures play a primary role in the study at hand, the data scientist could either make the problem Markovian by augmenting the input space with temporal features (i.e. augment input variables with some temporal features, and assume temporal structures play no role other than through selected feeatures) or refer to our :ref:`Time Series` section.



.. toctree::
	:hidden:

	problem_formulation
	quantifying_informativeness
	applications
	estimation
