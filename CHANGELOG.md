

# Change Log
## v.1.4.8 Changes

* Froze the versions of all python packages in the docker file.


## v.1.4.7 Changes

Changes related to optimizing Principal Feature Selection.

* Made it easy to change PFS' default learning parameters.
* Changed PFS' default learning parameters (learning rate is now 0.005 and epsilon 1e-04)
* Adding a seed parameter to PFS' fit for reproducibility.

To globally change the learning rate to 0.003, change Adam's epsilon to 1e-5, and the number of epochs to 25, do

```Python
from kxy.misc.tf import set_default_parameter
set_default_parameter('lr', 0.003)
set_default_parameter('epsilon', 1e-5)
set_default_parameter('epochs', 25)
```

To change the number epochs for a single iteration of PFS, use the `epochs` argument of the `fit` method of your `PFS` object. The `fit` method now also has a `seed` parameter you may use to make the PFS implementation deterministic.

Example:
```Python
from kxy.pfs import PFS
selector = PFS()
selector.fit(x, y, epochs=25, seed=123)
```

Alternatively, you may also use the `kxy.misc.tf.set_seed` method to make PFS deterministic.


## v.1.4.6 Changes

Minor PFS improvements.

* Adding more (robust) mutual information loss functions.
* Exposing the learned total mutual information between principal features and target as an attribute of PFS.
* Exposing the number of epochs as a parameter of PFS' fit.