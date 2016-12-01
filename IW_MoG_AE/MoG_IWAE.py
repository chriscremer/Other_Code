



import numpy as np
import tensorflow as tf
import random
import math
from os.path import expanduser
home = expanduser("~")
import time
import pickle

from MoG_VAE import MoG_VAE

class MoG_IWAE(MoG_VAE):


    def weight_by_cluster(self, log_w, normalized_weights):

        samps_per_cluster = self.n_particles / self.n_clusters

        weights= []
        for cluster in range(self.n_clusters):

            # [spc, B], this is log_w for the samples of this cluster
            samples_from_this_cluster = tf.slice(log_w, [cluster*samps_per_cluster, 0], [samps_per_cluster, self.batch_size])

            #log mean exp
            max_ = tf.reduce_max(samples_from_this_cluster, reduction_indices=0) #over samples
            logmeanexp = tf.log(tf.reduce_mean(tf.exp(samples_from_this_cluster - max_), reduction_indices=0)) + max_
            # avg_of_clust = tf.reduce_mean(samples_from_this_cluster, reduction_indices=0) #over smaples so its [B]
            weights.append(logmeanexp)
            
        # [C,B]
        weights = tf.pack(weights)
        # [B,C]
        weights = tf.transpose(weights, [1,0])
        # [B,C]
        log_w = weights + tf.log(normalized_weights)

        return log_w





