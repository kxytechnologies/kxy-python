.. meta::
   :description: The theoretical foundation of the KXY Lean AutoML platform, for time series.
   :http-equiv=content-language: en

**********************
Memoryful Observations
**********************

The **lean AI** approach we advocated for memoryless problems in section :ref:`Memoryless Observations` presents as much potential when applied to time series forecasting problems. We consider predicting a business outcome :math:`y_t` using past values :math:`y_{t-1}, \dots, y_1` and using present and past values of an exogeneous explanatory vector-valued time series :math:`\{x_t\}`. 

The time series approach to modeling expresses two key points: (i) like random variables, we are uncertain about the value of the phenomenon we are modeling until it is observed, (ii) but unlike random variables, the phenomenon of interest may exhibit some memory in that observations drawn at different times may be related. 


I - From Memoryful to Memoryless
--------------------------------
In practice, we do not have the luxury of being able to replay time so as to gather multiple samples of a phenomenon corresponding to the same time, which would be the equivalent of having multiple draws from the same random variable in the memoryless setting. We need to learn from a single finite-length path :math:`\{(y_1, x_1), \dots, (y_T, x_T) \}`. Consequently, instead of using all past values :math:`(y_{t-1}, x_{t-1}), \dots, (y_1, x_1)` to predict :math:`y_t`, we may have to settle for a shorter time window :math:`(y_{t-1}, x_{t-1}), \dots, (y_{t-q}, x_{t-q})` in the interest of forming low-sample estimates, where the window size :math:`q` can be as large as allowed by our sample size :math:`T`.

The natural question that arises is, can we simply define :math:`Y_i=y_t` and :math:`X_i=\left(x_t, y_{t-1}, x_{t-1}, \dots, y_{t-q}, x_{t-q}\right)`, and apply all the results developed in section :ref:`Memoryless Observations` to the dataset :math:`(Y_i, X_i)_{i \in [1, T]}`? 

The answer is yes, but we need to be cautious! The main difference with the memoryless setting is that :math:`\left(Y_i, X_i \right)` are not necessarily i.i.d. However, so long as the time series :math:`\{z_t\} = \{y_t, x_t\}` is assumed to be `stationary ergodic <https://en.wikipedia.org/wiki/Stationary_ergodic_process>`_, all population metrics we previously introduced are well-defined, make as much sense as in the memoryless case, and the associated sample estimates remain consistent.

More generally, when :math:`\{z_t\}` can be assumed to be trend-stationary and ergodic (i.e. :math:`\{y_t-f(t), x_t-g(t)\}` is stationary ergodic for some deterministic functions :math:`f, g`), we do not need to remove trends explicitly. We may simply add time as an explanatory variable, and apply results from the :ref:`Memoryless Observations` section to :math:`(Y_i, X_i^\prime)_{i \in [1, T]}`, with :math:`X_i^\prime = \left(t, x_t, y_{t-1}, x_{t-1}, \dots, y_{t-q}, x_{t-q}\right)`.


In the event (trend-)stationarity is too unrealistic an assumption, the time series may be assumed locally-stationary. In other words, we do not assume (trend-adjusted) marginals to be invariant by any translation, but we consider that the magnitude of a change of marginals resulting from a translation depends on the norm of the translation vector. The smaller the norm of the translation vector, the smaller the magnitude of the changes to the marginals.

Here too, results from the memoryless section apply, but with two caveats. First, we may not use as large a sample size :math:`T` as we want. :math:`T` has to be large enough that we may achieve low-variance estimates, yet small enough that the path used for training only contains the prevailing *local dynamics*. Second, all estimates from the memoryless section should only be considered valid in a limited time window following the last training timestamp [[*]_], and should be regenerated with new data on a rolling basis.


II - Choosing the Window Size
-----------------------------
It is important to note that :math:`q` is only a function of the length :math:`T` of the path we use for training. It is not necessarily chosen so that all lags are relevant. For a given choice of :math:`q`, we will have :math:`m=T-q` distinct samples and an even smaller `effective sample size <https://en.wikipedia.org/wiki/Effective_sample_size>`_ :math:`n<m`, [[*]_] which in turn affects the variability of the estimated solution to the maximum-entropy copula problem [[1]_]. 

As a rule of thumb, we need about 200 i.i.d. samples per input dimension to solve the maximum-entropy dual problem. Assuming a conservative ratio :math:`n/m`, we may estimate the minimum sample size :math:`T_j` required for :math:`q=j`, and choose :math:`q` to be the largest :math:`j` for which :math:`T>T_j`.

Once :math:`q` is chosen, section :ref:`2 - Variable Selection Analysis` can be used to determine which lags are actually insightful and should be included in your predictive model.






.. rubric:: References

.. [1] Kom Samo, Y.-L., Inductive Mutual Information Estimation: A Convex Maximum-Entropy Copula Approach. Proceedings of the 24th International Conference on Artificial Intelligence and Statistics (AISTATS) 2021, San Diego, California, USA. PMLR: Volume 130. 


.. rubric:: Footnotes

.. [*] Of size not exceeding :math:`T`.
.. [*] Given the time series memory.








