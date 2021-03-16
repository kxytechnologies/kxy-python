.. meta::
	:description: The theoretical foundation of the KXY Lean AutoML platform.
	:keywords:  Pre-Learning, Post-Learning, Maximum-Entropy Principle, Input Importance, Feature Importance, KXY, Model Explanation, Dataset Valuation, Input Importance, Feature Importance, Model Suboptimality, Model Optimality.
	:http-equiv=content-language: en
***********************
Memoryless Observations
***********************
In this section we provide an in-depth discussion of what makes the KxY platform work. We begin with predictive problems where input and output variables do not exhibit temporal structures, or their temporal structures are of negligible importance. For time series problems, refer to our :ref:`Memoryful Observations` section.

The KxY platform aims at **Democratizing Lean AI**. But what is *lean AI*, you might wonder? 

Our estimate is that *1-in-10* machine learning experiments fail, resulting in tremendous amount of avoidable waste (e.g. productivity, compute power and carbon footprint etc.). *Lean AI* is all about developping machine learning techniques to detect experiments in data science projects, or entire data science projects, that are likely to result in dead-ends and, as such, that should be avoided. Done right, this can increase the productivity of your data science teams tenfold, while slashing costs. 

*We are pioneers in this space, and our works are published in top-tier machine learning conferences.*

Real-life predictive modeling needs are primarily of two types. An organization could be starting a predictive modeling project from scratch, and might be interested in predicting a new business outcome using available data as potential explanatory variables. Alternatively, the organization might be looking to improve a predictive model that was previously trained and released to production. 

We refer to problems arising from attempting to determine whether projects of the former kind (resp. latter kind) would result in a dead-end as **pre-learning** problems (resp. **post-learning** problems).





.. toctree::
	:hidden:

	problem_formulation
	quantifying_informativeness
	applications
	estimation
