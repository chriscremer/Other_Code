
from __future__ import absolute_import
# from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm

# from black_box_svi import black_box_variational_inference
from autograd.optimizers import adam


from ball_sequence import make_ball_gif


sequence, action_list = (make_ball_gif())

print sequence.shape



