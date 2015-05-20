


import numpy as np
import csv
import random
import pprint



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
        return (a-y) * sigmoid_prime_vec(z)


class Network():

    def __init__(self, sizes, cost=CrossEntropyCost):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.default_weight_initializer`` (see docstring for that
        method).

        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost=cost


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
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) 
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]


    def feedforward(self, a):
        """Return the output of the network if ``a`` is input.

            Very useful to draw the arrays to see how this works

            Sigmoid funciton needs to be vectorized because the dot product of the weight matrix and the a vector is a vector
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid_vec(np.dot(w, a)+b)
        return a

    def backprop(self, x, y):
        """Return a tuple ``(bias_gradients, weight_gradients)`` representing the
        gradient for the cost function C_x.  ``bias_gradients`` and
        ``weight_gradients`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""

        #print 'backprop start'

        bias_gradients = [np.zeros(b.shape) for b in self.biases]
        weight_gradients = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        # we dont want to use the feedforward function here becasue we will need the z vecotrs later when calculating the error

        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):

            #print 'w ' + str(w.shape) 
            #print 'a ' + str(activation.shape) 


            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid_vec(z)
            activations.append(activation)
            
        # backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        bias_gradients[-1] = delta
        weight_gradients[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            spv = sigmoid_prime_vec(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * spv
            bias_gradients[-l] = delta
            #print 'qqqqqqqqq'
            #print delta
            #print  activations[-l-1].transpose()
            #print 'qqqqqqqqqqq'
            weight_gradients[-l] = np.dot(delta, activations[-l-1].transpose())

        #print 'backprop end'

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
            #print 'x ' + str(x)
            #print 'y ' + str(y)
            delta_bias_gradients, delta_weight_gradients = self.backprop(x, y)
            bias_gradients = [nb+dnb for nb, dnb in zip(bias_gradients, delta_bias_gradients)]
            weight_gradients = [nw+dnw for nw, dnw in zip(weight_gradients, delta_weight_gradients)]

        #whats going on here???    
        self.weights = [(1-learn_rate*(lmbda/n))*w-(learn_rate/len(mini_batch))*nw 
                        for w, nw in zip(self.weights, weight_gradients)]
        self.biases = [b-(learn_rate/len(mini_batch))*nb 
                       for b, nb in zip(self.biases, bias_gradients)]

    def total_cost(self, data, lmbda):
            """Return the total cost for the data set ``data``.  The flag
            ``convert`` should be set to False if the data set is the
            training data (the usual case), and to True if the data set is
            the validation or test data.  See comments on the similar (but
            reversed) convention for the ``accuracy`` method, above.
            """
            cost = 0.0
            for x, y in data:
                a = self.feedforward(x)
                #print 'got ' + str(a)
                #print 'expected ' + str(y)


                cost += self.cost.cost(a, y)/len(data)
            cost += 0.5*(lmbda/len(data))*sum(
                np.linalg.norm(w)**2 for w in self.weights)

            return cost


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

        The
        other non-optional parameters are self-explanatory, as is the
        regularization parameter ``lmbda``. 

        The method also accepts
        ``evaluation_data``, usually either the validation or test
        data.  We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four
        lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.

        """
        n = len(training_data)

        if evaluation_data: n_data = len(evaluation_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []

        for j in xrange(epochs):

            print 'Start epoch ' + str(j)

            for sample in training_data:
                if training_data[0][0].shape == (0,1):
                    print 'IS EMPTY 1'

            random.shuffle(training_data)

            for sample in training_data:
                if training_data[0][0].shape == (0,1):
                    print 'IS EMPTY 2'


            #make a list of the mini-batches
            mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]

            for batch in mini_batches:
                if batch[0][0].shape == (0,1):
                    print 'YES THERE IS AN EMPTY'

            #pprint.pprint(mini_batches)

            for mini_batch in mini_batches:
                #print 'mini_batch'
                #print mini_batch
                self.update_mini_batch(mini_batch, learn_rate, lmbda, len(training_data))

            print "Epoch %s training complete" % j
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print "Cost on training data: {0}".format(cost)
                #stringtoprint = "Cost on training data: {0}".format(cost)
                # with open('workfile.txt', 'a') as f2:
                #     f2.write("Cost on training data: {0}".format(cost))
                #     f2.write('\n')
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print "Accuracy on training data: {0} / {1}".format(accuracy, n)
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print "Cost on evaluation data: {0}".format(cost)
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print "Accuracy on evaluation data: {0} / {1}".format(
                    self.accuracy(evaluation_data), n_data)
            print
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

sigmoid_vec = np.vectorize(sigmoid)

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

sigmoid_prime_vec = np.vectorize(sigmoid_prime)



