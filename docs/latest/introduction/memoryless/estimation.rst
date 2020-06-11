

IV - Model-Free Estimation
==========================

.. admonition:: Summary

 	We discuss estimating previously introduced metrics in a model-free and nonparametric fashion. 

 	We argue for the use of `the principle of maximum entropy <https://en.wikipedia.org/wiki/Principle_of_maximum_entropy>`_ to do so. We discuss formulating the maximum entropy problem in the observation space (the **primal space**), as well as in the **copula-uniform dual space** (or simply the **dual space**) defined as the image of the observation space through `the probability integral transform <https://en.wikipedia.org/wiki/Probability_integral_transform>`_. 

 	We argue that, for the purpose of estimating mutual informations, unless marginals are known in advance to be Gaussian, which would rarely be the case in practice, the maximum entropy problem should be formulated and solved in the dual space.

For the purposes of pre-learning and post-learning, estimation ought to be performed in a model-free fashion in the sense that we should not assume that the :math:`(y, x)` is known. 

Considering that the pre-learning and post-learning phases should allow for any model to possibly be used in the learning phase, we should not be constrained by a choice of generative model. Doing so would rig the achievable performance and suboptimality analyses towards the specific generative model that was posited, when they should allow us to quantify what performance can be achieved by *any* model, and how much additional performance can be generated whithout restriction on which model to use to do so.

To apply the results of section :ref:`III - Applications` we need to be able to estimate the Shannon entropy :math:`h(y)` of a categorical variable :math:`y`, and the (differential, Shannon or mixed) mutual information :math:`I(y, x)` between a continuous or discrete variable :math:`y` and a :math:`d`-dimensional random vector with continuous and/or categorical coordinates. To this end, all we need in addition to an estimator for Shannon entropy is an estimator for the mutual information between two continuous variables, and an estimator for the differential entropy of a continuous scalar random variable. 

To estimate a conditional entropy (resp. mutual information) where conditioning is based on a categorical variable (not a specific value thereof), we recall that it is equal to the average value of the entropy (resp. mutual information) of the conditional distribution(s), weighted by the probability of each outcome.

.. We make it a point never to estimate the entropy of continuous random vector directly. As previously discussed, in matters pertaining to quantifying associations, the copula of a random vector plays a more central role than its marginals. While the copula of a random vector fully (and only) captures associations between its coordinates, its marginals are heavily influenced by how the sample was gathered and invertible feature transformations it might have undergone (e.g. logarithm transformation, standardization, etc.). Intuitively, no invertible transformation applied to coordinates should affect our understanding of associations between them. Such transformations, when increasing, would indeed leave the copula invariant, but would affect the marginals.


1 - Univariate Entropies
------------------------

a) Shannon Entropy
^^^^^^^^^^^^^^^^^^
We estimate the entropy of categorical random variables using the plug-in estimator

.. math::
	:label: ent_ent_est

	\hat{H} = -\sum_{i=1}^q \hat{p}_i \log \hat{p}_i,


where :math:`\hat{p}_i` is the frequency or MLE estimator of the :math:`i`-th category. This is consistent and asymptocially normal estimator. [*]_ 

This estimator applies to both univariate and multivariate discrete distributions, as the latter can be turned into the former.



b) Univariate Differential Entropy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We implement three methods for estimating the differential entropy of a scalar random variable. They are all similar in that they resort to first approximating pdf of the distribution, and then estimating the resulting entropy, but they are different in the approximation used.


M-Spacing Estimator
"""""""""""""""""""
Introduced in [6]_, this method locally estimates the probability density function by approximating :math:`p(x_i)d\epsilon` by the probability that the :math:`m`-th nearest neighbor of the point :math:`x_i` exists within the sphere of radius :math:`\frac{\epsilon}{2}`, that there are :math:`m-1` points strictly inside of the sphere, and exactly :math:`n-m-1` points more than :math:`\frac{\epsilon}{2}` away from the sphere, where :math:`n` is the number of i.i.d. samples.

For :math:`m=1`, the estimator of the entropy of a continuous scalar random variable :math:`x` from n i.i.d samples :math:`x_1, \dots, x_n` using the standard 1-spacing estimator reads:

.. math::

	\hat{h}(x) = - \gamma(1) + \frac{1}{n-1} \sum_{i=1}^{n-1} \log \left[ n \left(x_{(i+1)} - x_{(i)} \right) \right],

where :math:`x_{(i)}` is the i-th smallest sample, and :math:`\gamma` is `the digamma function. <https://en.wikipedia.org/wiki/Digamma_function>`_ See [2]_ and references therein for a review of its statistical properties.


Gaussian Moment-Matching
""""""""""""""""""""""""
This method simply consists of assuming the true data generating distribution is Gaussian, estimating its variance :math:`\hat{\sigma}^2`, and estimating the corresponding entropy as 

.. math::

	\hat{h}(x) = \frac{1}{2} \log \left(2 \pi e \hat{\sigma}^2 \right).



Kernel-Density Estimation
"""""""""""""""""""""""""
Like its name suggests, this method uses `kernel density estimation <https://en.wikipedia.org/wiki/Kernel_density_estimation>`_ to approximate the pdf nonparametrically. We find this method to be the most robust of the three; it performs well with the Gaussian kernel and an appropriate bandwith choice.



2 - Maximum Entropy Multivariate Estimation
-------------------------------------------
What's left for us to provide is an estimator for the mutual information between two continuous random variables. Methods have been proposed for nonparametric estimation of the mutual information between two continuous random variables. Perhaps the most notable work is the application of the :math:`m`-spacing entropy estimator of [6]_ in [7]_. 

We find these :math:`m`-spacing approaches undesirable as they are not coordinate/representation invariant, [*]_ they scale poorly with sample size, and they can result in negative mutual informations. 

Instead, we propose a novel maximum entropy alternative.


a) The Maximum Entropy Principle
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We consider estimating the mutual information :math:`I(a, b)` between two continuous (possibly one-dimensional) random vectors :math:`a` and :math:`b` whose true pdf is :math:`p(a, b)`.

Central to our approach is the choice of a constraint function :math:`\phi` such that the functional 

.. math::
	
	p \to E\left(\phi(a, b)\right) := \int \phi(a, b) p(a, b) da db

measures association between coordinates of :math:`a` and/or coordinates of :math:`b`. As an illustration, the constraint function :math:`\phi_{f,g}(a, b) = \left(f(a)f(a)^T, f(a) g(b)^T, g(b)g(b)^T\right)` represents the autocovariance matrix of :math:`(f(a), g(b))` for any two functions :math:`f` and :math:`g`.

Additionally, we assume that we can form an efficient estimator of :math:`E\left(\phi(a, b)\right)`.

To estimate :math:`I(a, b)`, we estimate :math:`E\left(\phi(a, b)\right)` from data, say :math:`E\left(\phi(a, b)\right) \approx \hat{\alpha}` and we ask ourselves the question: among all generative model for :math:`(a, b)` that satisfy the constraint

.. math::

	E\left(\phi(a, b)\right) = \hat{\alpha},


which model is the *most uncertain about everything else*, or equivalently, which model has the highest entropy?

The result is a generative model, and its associated mutual information, that make no arbitrary assumption on the data generating distribution, and only reflect properties encoded by :math:`\phi` that have been evidenced in the data through :math:`\hat{\alpha}`. As :math:`\phi` gets more and more expressive, the maximum entropy distribution converges to the true data generating distribution, but more importantly, the associated mutual information converges to the true mutual information.

This modeling paradigm, known as `the principle of maximum entropy <https://en.wikipedia.org/wiki/Principle_of_maximum_entropy>`_, was first pioneered by E.T. Jaynes, one of the most celebrated authors in the probabilistic machine learning community, in his seminal works [3]_ and [4]_.

Note that, because we are always estimating our mutual information as the mutual information of a distribution, instead of estimating the three differential entropies separately as in the case of :math:`m`-spacing, our estimated mutual information can never be negative. This approach is also very efficient given that it depends on data solely through :math:`\hat{\alpha}` and, as such, is amenable to caching. The foregoing feature also makes this approach great for privacy.


Applying the above to the estimation the mutual information :math:`I(y; x)` between continuous inputs and a continuous label, as discussed on the previous page, this mutual information is also equal to the mutual information between the respective copula-uniform dual representations :math:`I(u_y, u_x)`. Thus, we can apply the maximum entropy principle in the primal space (i.e.  :math:`(a, b) = (y, x)`) or in the dual space (i.e. :math:`(a, b) = (u_y, u_x)`). Both approaches are implemented in the :code:`kxy` package. We discuss maximum entropy inference in each space below, and then draw the link between the two.


3 - Primal Estimation of Multivariate Copula Entropy
----------------------------------------------------
In the primal space, :math:`(a,b)=(y, x)` and we use as maximum entropy constraints the Pearson autocovariance matrix of :math:`(y, x)`. 

Fortunately, in this case, the maximum entropy problem has closed-form solution, and the maximum entropy distributions happen to be the Gaussian distributions with same autocovariance matrix. Note that the mean does not matter as the differential entropy is invariant by translation.

To estimate the expected constraints from data, we could use the standard unbiased estimator for Pearson's autocovariance matrix. It has great asymptotic properties, but it is not robust to outliers. 

To form a more robust estimator, we note that Gaussian distributions being fully characterized by their first two moments, and correlations being invariant by translation, there is a one-to-one map between Pearson's correlation matrix and Spearman's rank correlation matrix for Gaussian distributions. Thus, we may first estimate the Spearman rank correlation matrix, which is robust to outliers, and then map it back to its Pearson's counterpart.

.. important::

	Strickly speaking, using Pearson's covariance as expected constraint in the primal space cannot reveal nonlinear associations in data. To do so in the primal space, one needs to use another constraint functions (e.g. include skewness and kurtosis terms). However, closed form solutions would not be available, and numerical estimations would be tedious, if at all possible. This is one of the reason why we advise against estimating mutual information in the primal space. For a broader discussion, see section :ref:`5 - Primal v Dual Spaces`.


4 - Dual Estimation of Multivariate Copula Entropy
--------------------------------------------------
In the primal space :math:`(a,b)=(u_y, u_x)`, and we exploit the fact that the mutual information between two continuous random variables is the mutual information between their copula-uniform dual representations.

The two primary requirements guiding the choice of the constraint function :math:`\phi` are:

#. :math:`E\left(\phi(u_y, u_x)\right)` should reflect depedence between coordinates of the copula-uniform dual representations :math:`(u_y, u_x)` or, equivalently, between coordinates of :math:`(y, x)`.
#. :math:`E\left(\phi(u_y, u_x)\right)` should be amenable to efficient and robust estimation from i.i.d. samples of :math:`(y, x)`.


These requirements are satisfied by a plethora of concordance measures, among which Spearman's rank correlation, Kendall's tau, Gini's gamma, Blest's measures, to name but a few. Simply put, concordance measures (Definition 5.1.7 in [1]_) quantify the extent to which two random variables take large (resp. small) values at the same time. 

i) Kendall's Tau
^^^^^^^^^^^^^^^^

An example directly in line with this interpretation is Kendall's tau (or Kendall's rank correlation), defined as 

.. math::

	\tau = \mathbb{P} \left[(x_1-x_2)(y_1-y_2) > 0\right] - \mathbb{P} \left[(x_1-x_2)(y_1-y_2) < 0\right]


where :math:`(x_1, y_1)` and :math:`(x_2, y_2)` are independent draws from the same bivariate distribution with copula-uniform dual representation :math:`(u, v)` and copula :math:`C(u, v)`. It can be expressed in terms of the copula-uniform dual representation as 

.. math::

	\tau = E\left( C(u, v)\right),

and its sample estimate from n i.i.d. draws of :math:`(x, y)` reads

.. math::
	
	\hat{\tau} = \frac{2}{n(n-1)} \sum_{i<j} \text{sgn}(x_i-x_j)\text{sgn}(y_i-y_j).


**Interpretation:** :math:`\mathbb{P} \left[(x_1-x_2)(y_1-y_2) > 0\right]` measures the propensity for two random variables :math:`(x, y)` to be concordant (i.e. increase simultaneously or decrese simultaneously across independent random draws), while :math:`\mathbb{P} \left[(x_1-x_2)(y_1-y_2) < 0\right]` measures their propensity to be discordant (i.e. one decreases while the other increases between random draws). Thus, :math:`\tau \in [-1, 1]` is :math:`0` if and only if the directions of changes of :math:`x` and :math:`y` across independent random draws are unrelated. :math:`\tau=-1` (resp. :math:`\tau=1`) if and only if the directions of changes of :math:`x` and :math:`y` across independent random draws are always opposite (resp. the same). In fact, :math:`\tau` can also be interpreted as the Pearson correlation between the signs of increments of :math:`x` and :math:`y` across two independent draws:

.. math::

	\tau = \mathbb{C}\text{orr}\left(\text{sgn}(x_1-x_2), \text{sgn}(y_1-y_2) \right)

.. note:: 
	Like copulas, Kendall's tau is invariant by any increasing transformation applied to :math:`x` and/or :math:`y`.

Kendall's tau cannot be directly utilized within our framework, as the corresponding :math:`\phi` depends on the copula. That said, it has been shown to be asymptotically equivalent to another measure of concordance, namely Spearman's rho, for which :math:`\phi` is unrelated to the copula. [*]_


ii) Spearman's Rho
^^^^^^^^^^^^^^^^^^
Let us consider the bivariate random variable :math:`(x, y)` with copula-uniform representation :math:`(u, v)`, and n i.i.d. draws thereof :math:`(x_1, y_1), \dots, (x_n, y_n)`. The sample version of the Spearman rank correlation is defined as the Pearson correlation between the rank of :math:`x_i` (among :math:`x_1, \dots, x_n`) and the rank of :math:`y_i` (among :math:`y_1, \dots, y_n`)

.. math::

	\hat{\rho}(x, y) &= \mathbb{C}\text{orr}\left(\text{rg}(x_i), \text{rg}(y_i)\right) \\
					 &= \frac{12}{n^2-1}\left[\left( \frac{1}{n} \sum_{i=1}^n \text{rg}(x_i) \text{rg}(y_i) \right) - \frac{(n+1)^2}{4} \right].

Its population version reads

.. math::
	
		\rho :=  E\left( \phi_\rho(u, v)\right), ~~ \text{with} ~~ \phi_\rho(u,v) :&= 12\left[uv-\frac{1}{4} \right] \\
			  																		&= 3 \left[(u+v-1)^2 - (u-v)^2\right].


.. note::

	As :math:`u` and :math:`v` are both uniformly distributed on :math:`[0, 1]`, :math:`12 E\left(uv-\frac{1}{4}\right)` is in fact the Pearson correlation between :math:`u` and :math:`v`, so that

	.. math:: 

		\rho := \mathbb{C}\text{orr} \left(u, v\right).

	Thus, Spearman's rho is an obvious measure of association in the copula-uniform dual space. Although Pearson's correlation only captures linear association in the copula-uniform dual space, it is worth stressing that, Spearman's rho is in fact invariant by any increasing transformation applied to :math:`x` and/or :math:`y`.


We refer the reader to Chapter 5 in [1]_ for more details on the link between concordance measures and copulas.



iii) Other Rank Statistics
^^^^^^^^^^^^^^^^^^^^^^^^^^
Spearman's rho shed some light on the link between the empirical copula-uniform dual representation :math:`\left(\text{rg}(x_i)/n,  \text{rg}(y_i)/n\right)` and the true copula-uniform dual representation :math:`(u, v)`. Under mild conditions, the empirical copula-uniform dual representation converges in distribution to the true copula-uniform dual representation and, for a given :math:`\phi`, 

.. math::
	:label: phi_est

	\frac{1}{n} \sum_{i=1}^n \phi\left(\frac{\text{rg}(x_i)}{n}, \frac{\text{rg}(y_i)}{n} \right)


is a good estimator of :math:`E\left(\phi(u, v)\right)`. Hence, a larger class of constraint functions :math:`\phi` can be obtained by choosing :math:`\phi` to reflect association in the copula-uniform dual space, and using Equation :eq:`phi_est` as estimator in the primal space.

An example is Gini's gamma, for which 

.. math::

	\phi_\gamma(u,v) := 2 \left(\vert u+v-1 \vert - \vert u-v \vert\right),


and that can be estimated in the primal space as 

.. math::

	\hat{\gamma} = \frac{2}{n} \left[\sum_{i=1}^n \left\vert \frac{\text{rg}(x_i)}{n} + \frac{\text{rg}(y_i)}{n} - 1 \right\vert - \left\vert \frac{\text{rg}(x_i)}{n} - \frac{\text{rg}(y_i)}{n} \right\vert \right].


iv) Implemented Constraints
^^^^^^^^^^^^^^^^^^^^^^^^^^^
At the current time, the API only implements constraints based on Spearman's rank correlation matrices. Note that, although we only discussed bivariate constraint functions above, the extension to the multivariate case is trivial, and would consist of choosing a vector-valued :math:`\phi` with coordinates all pairwise constraints.



v) Beyond Concordance
^^^^^^^^^^^^^^^^^^^^^
A blindspot of concordance measures is that they only capture monotonic associations in data. To illustrate this, let us consider a toy example. We consider a scalar random variable :math:`x` drawn from a distribution whose pdf is symmetric about :math:`0` (for instance a centered Gaussian, or the uniform distribution on :math:`[-1, 1]`), and the random variable :math:`y=x^2`. 

By symmetry, :math:`(x, y)` and :math:`(-x, y)` have the same joint distributions, and therefore the same Spearman rank correlation (or any concordance measure for that matter). Additionally, the Spearman rank correlation (resp. any concordance measure) between :math:`-x` and :math:`y` should be the opposite of the Spearman rank correlation (resp. the concordance measure) between :math:`x` and :math:`y`. Hence the Spearman rank correlation (and any other concordance measure) between :math:`x` and :math:`y` should be :math:`0`. 

This implies that an application of the principle of maximum entropy to :math:`(x, y)` using as empirical evidence their Spearman rank correlation (or any other concordance measure) would suggest that they are statistically independent. 

The foregoing observation neither invalidates the pertinence of the maximum entropy principle, nor does it invalidate the utility of using concordance measures as maximum entropy constraints. It simply stresses the fact that concordance measures can only capture monotonic association in data.

To mitigate this limitation, we use the fact that when :math:`y` and :math:`x` are both continuous,

.. math::

	I\left(y; x, \vert x - \mu \vert \right) &= I\left(y; x\right) + \underbrace{I\left(\vert x - \mu \vert; y  \big\vert x\right)}_{=0} \\
	 										 &= I\left(y; x \right).

We then apply the maximum entropy principle to the left handside using Spearman rank correlations as constraints, which allows us to capture associations that are monotonic in :math:`x` and/or in :math:`\vert x - \mu \vert`, where we choose :math:`\mu` to be the sample mean of :math:`x`.

Going back to our toy example, the Spearman rank correlation between :math:`y` and :math:`|x|` is :math:`1`, and association in our data is fully reflected by the maximum entropy constraints. More generally, :math:`\vert x - \mu \vert` allows us to capture any non-monotonic association that is symmetric about the hyperplane :math:`x=\mu` and monotonic in :math:`\vert x - \mu \vert`.


.. note::

	The current version of the :code:`kxy` package does not yet fully capture all non-monotonic associations; a notable exception is periodic associations. Support for periodic associations will be added in the near future.



5 - Primal v Dual Spaces
------------------------
For the purpose of estimating mutual information, solutions to the maximum entropy problem in the primal and dual spaces are related but very different. To illustrate the difference, let us consider the bivariate case where both :math:`y` and :math:`x` are continuous scalar random variables, with respective copula-uniform dual representations :math:`u` and :math:`v`. 

The dual maximum entropy problem maximizes :math:`h(u, v)` under certain constraints, whereas the primal maximum entropy problem maximizes :math:`h(x, y)` under other constraints. As discussed in section :ref:`c) Entropy Decomposition`, 

.. math::
	
	h(x, y) = h(x) + h(y) + h(u, v).

Clearly, the marginal entropies :math:`h(x)` and :math:`h(y)` play an important role in the primal maximum entropy problem. However, we also know that the mutual information between :math:`x` and :math:`y` does not depend on marginals! In fact, marginals are completely uninteresting in the study of structures in *continuous* random variables, not least because they are representation-specific.

Another way to look at the term :math:`h(x) + h(y)` is as a regularizer that shrinks marginals towards the most unstructured marginal distributions that are consistent with the constraints. For instance, when Pearson's autocovariance is used as constraints in the primal space, the term :math:`h(x) + h(y)` shrinks marginals towards being Gaussian, whereas the term :math:`h(u, v)` avoids excessive structure in the copula. 

In practice however, we usually have absolutely no clue what makes sense as base distribution, if any, towards which we should be shrinking marginals. Marginals depend on how the data were gathered in the first place, and there is no clear '*uninformative*' distribution for marginal distributions that are not bounded. For instance, whether you work with prices or log-prices, volume or log-volumes, sigmoid-normalized data or not, will have a drastic and unexpected effect on marginals and the '*uninformative*' distributions, if any, towards which they should be shrunk. 

Another way to look at this is that to properly work in the primal space, the constraint function should be *so* informative about marginals that they will not be shrunk towards the wrong distribution. Once more, this is counter-intuitive as mutual information between continuous random variables has nothing to do with marginals, and would create unecessary analytical, numerical and computational hurdles.s




.. rubric:: References

.. [1] Nelsen, R.B., 2007. An introduction to copulas. Springer Science & Business Media.

.. [2] Beirlant, J., Dudewicz, E.J., Györfi, L., van der Meulen, E.C., 1997. Nonparametric entropy estimation: an overview. International Journal of Mathematical and Statistical Sciences. 6 (1): 17–40. ISSN 1055-7490. 

.. [3] Jaynes, E.T., 1957. Information theory and statistical mechanics. Physical review, 106(4), p.620.

.. [4] Jaynes, E.T., 1957. Information theory and statistical mechanics. II. Physical review, 108(2), p.171.

.. [5] Sidak, Z., Sen, P.K. and Hajek, J., 1999. Theory of rank tests. Elsevier.

.. [6] Kozachenko, L. F., and Nikolai N. Leonenko. "Sample estimate of the entropy of a random vector." Problemy Peredachi Informatsii 23.2 (1987): 9-16.

.. [7] Kraskov, Alexander, Harald Stögbauer, and Peter Grassberger. "Estimating mutual information." Physical review E 69.6 (2004): 066138.


.. rubric:: Footnotes

.. [*] Hint: Apply the central limit theorem and the delta method.

.. [*] Especially when the :math:`L^\infty` is selected for the nearest neighbor search, choice often made to speed up computations through the use of k-d trees.

.. [*] See [5]_ pages 60 and 61.



