


import numpy as np

import random
import pickle

from costs import *
from activations import *




class Network():

    def __init__(self, layer_sizes, activations, cost=CrossEntropyCost, regularization='l2', weights_n_biases=None):
        """The list ``layers`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.default_weight_initializer`` (see docstring for that
        method).

        """
        self.num_layers = len(layer_sizes)
        self.layers = layer_sizes

        self.cost=cost
        self.activation_functions=activations
        self.regularization=regularization

        if weights_n_biases == None:
            self.default_weight_initializer()
            self.best_prev_acc = -1
        else:
            self.weights = weights_n_biases[0]
            self.biases = weights_n_biases[1]
            self.best_prev_acc = weights_n_biases[2]


    def default_weight_initializer(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        numpy randn returns an array of dimentions (x,y)
        so it has columns for the nodes that the weight starts at and rows for the nodes they connect to 

        """
        self.biases = [np.random.randn(y, 1) for y in self.layers[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) 
                        for x, y in zip(self.layers[:-1], self.layers[1:])]


    def backprop(self, x, y):
        """Return a tuple ``(bias_gradients, weight_gradients)`` representing the
        gradient for the cost function C_x.  ``bias_gradients`` and
        ``weight_gradients`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""

        bias_gradients = [np.zeros(b.shape) for b in self.biases]
        weight_gradients = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        # we dont want to use the feedforward function here becasue we will need the z vecotrs later when calculating the error
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w, act_func in zip(self.biases, self.weights, self.activation_functions):

            z = np.dot(w, activation)+b
            zs.append(z)
            #activation = sigmoid_vec(z)
            activation = act_func.function_vec(z)
            activations.append(activation)
            
        # backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        bias_gradients[-1] = delta
        weight_gradients[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            #spv = sigmoid_prime_vec(z)
            spv = self.activation_functions[-l].function_prime_vec(z)
            
            delta = np.dot(self.weights[-l+1].transpose(), delta) * spv
            bias_gradients[-l] = delta
            weight_gradients[-l] = np.dot(delta, activations[-l-1].transpose())

        return (bias_gradients, weight_gradients)



    def update_mini_batch(self, mini_batch, learn_rate, lmbda, n):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.

        The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``learn_rate`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.

        """
        bias_gradients = [np.zeros(b.shape) for b in self.biases]
        weight_gradients = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:

            delta_bias_gradients, delta_weight_gradients = self.backprop(x, y)
            bias_gradients = [nb+dnb for nb, dnb in zip(bias_gradients, delta_bias_gradients)]
            weight_gradients = [nw+dnw for nw, dnw in zip(weight_gradients, delta_weight_gradients)]
  
        self.biases = [b - (learn_rate/len(mini_batch))*nb 
                       for b, nb in zip(self.biases, bias_gradients)]


        #L2 Regularization
        if self.regularization == 'l2':
        #new_w = w - LR(nw/batch_size + w*lmbda/n)
        #self.weights = [(1-learn_rate*(lmbda/n))*w-(learn_rate/len(mini_batch))*nw 
        #                for w, nw in zip(self.weights, weight_gradients)] 
        #this way is more interpretable  
            self.weights = [w - learn_rate*(  nw/len(mini_batch) + (lmbda/n)*w) 
                            for w, nw in zip(self.weights, weight_gradients)]
       


        #L1 Regularization (Clipping)
        elif self.regularization == 'l1':
            weights_temp = []
            for w, nw in zip(self.weights, weight_gradients):

                w_temp = w - learn_rate*(nw/len(mini_batch))

                for i in range(len(w_temp)):
                    for j in range(len(w_temp[0])):

                        if w_temp[i][j] > 0:

                            w_temp[i][j]  = max([0, w_temp[i][j]  - (lmbda/n)*learn_rate])

                        elif w_temp[i][j]  <= 0:

                            w_temp[i][j]  = min([0, w_temp[i][j]  + (lmbda/n)*learn_rate])

                weights_temp.append(w_temp)

            self.weights = weights_temp
        






    def feedforward(self, a):
        """Return the output of the network if ``a`` is input.
            Very useful to draw the arrays to see how this works
            Sigmoid funciton needs to be vectorized because the dot product of the weight matrix and the a vector is a vector
        """
        #for b, w in zip(self.biases, self.weights):
        for b, w, act_func in zip(self.biases, self.weights, self.activation_functions):
            #a = sigmoid_vec(np.dot(w, a)+b)
            a = act_func.function_vec(np.dot(w, a)+b)
        return a

    def total_cost(self, data, lmbda):
            """Return the total cost for the data set ``data``. 
            """
            cost = 0.0
            for x, y in data:
                a = self.feedforward(x)
                cost += self.cost.cost(a, y)/len(data)

            if self.regularization == 'l2':
                cost += 0.5*(lmbda/len(data))*sum(
                    np.linalg.norm(w)**2 for w in self.weights)
            elif self.regularization == 'l1':
                cost += (lmbda/len(data))*sum(
                    np.linalg.norm(w)**1 for w in self.weights)

            return cost

    def accuracy(self, data):

        boundary=0.5
        accuracy = []
        for sample in data:
            output = self.feedforward(sample[0])
            #print output
            #print 'output ' + str(output) + ' expected ' + str(sample[1])
            if (output[1] > boundary and sample[1][1] > boundary) or (output[1] < boundary and sample[1][1] < boundary):
                accuracy.append(1)
            else:
                accuracy.append(0)

        return np.mean(accuracy)


    def SGD(self, training_data, epochs, mini_batch_size, learn_rate, 
            lmbda = 0.0, 
            evaluation_data=None, 
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False, 
            monitor_training_accuracy=False):
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.
        """
        n = len(training_data)

        #for recording best stats
        best_eval_cost = -1
        best_acc = -1
        best_weights = []
        best_biases = []

        #for early stopping
        numb_non_dec_epochs = 0

        for j in xrange(epochs):

            print 'Start epoch ' + str(j)

            random.shuffle(training_data)

            #make a list of the mini-batches
            mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]

            #run each mini-batches
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learn_rate, lmbda, len(training_data))

            #report the stats of this epoch
            training_cost= -1.0
            training_accuracy= -1.0
            evaluation_cost= -1.0
            evaluation_accuracy= -1.0
            if monitor_training_cost:
                training_cost = self.total_cost(training_data, lmbda)
            if monitor_training_accuracy:
                training_accuracy = self.accuracy(training_data)
            if monitor_evaluation_cost:
                evaluation_cost = self.total_cost(evaluation_data, lmbda)
            if monitor_evaluation_accuracy:
                evaluation_accuracy = self.accuracy(evaluation_data)

            print ('Epoch ' + str(j) + 
                    '|Train cost ' + str(training_cost) + 
                    '|Train acc ' + str(training_accuracy) + 
                    '|Eval cost ' + str(evaluation_cost) + 
                    '|Eval acc ' + str(evaluation_accuracy)
                    )

            if j %50 == 0:
                for layer1 in self.weights:
                    print layer1
                for sample in evaluation_data[:10]:
                    output = self.feedforward(sample[0])
                    print 'output ' + str(output[1]) + ' expected ' + str(sample[1][1])

            #save best model
            #if eval cost improved and accuracy didnt go down
            if (evaluation_accuracy >= self.best_prev_acc or self.best_prev_acc == -1) and (evaluation_cost < best_eval_cost or best_eval_cost == -1):
                best_eval_cost = evaluation_cost
                best_acc = evaluation_accuracy
                best_weights = self.weights
                best_biases = self.biases
                numb_non_dec_epochs = 0
                print 'Updated best'
            else:
                numb_non_dec_epochs += 1
                if numb_non_dec_epochs > 2:
                    print 'Early Stop'
                    weights_n_biases = [best_weights, best_biases, best_acc]
                    pickle.dump(weights_n_biases, open( "saved/w_n_b.p", "wb" ) )
                    break


        print 'Saving'
        weights_n_biases = [best_weights, best_biases, best_acc]
        pickle.dump(weights_n_biases, open( "saved/w_n_b.p", "wb" ) )
        print 'Training Complete'

        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy





