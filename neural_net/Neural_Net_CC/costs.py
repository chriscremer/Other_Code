


import numpy as np

from activations import *

class CrossEntropyCost:
    @staticmethod
    def cost(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).
        """
        return np.nan_to_num(np.sum(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        return (a-y)



class QuadraticCost:

    @staticmethod
    def cost(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.

        Quadratic is useful when the output of neurons aren't sigmoid ie betw 0 and 1
        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a-y) * Linear_Activation.function_prime_vec(z)
