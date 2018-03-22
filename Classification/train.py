




import numpy as np





#Load data
print 'Loading data'
with open(home+'/Documents/MNIST_data/mnist.pkl','rb') as f:
    mnist_data = pickle.load(f)

train_x = mnist_data[0][0]
train_y = mnist_data[0][1]
valid_x = mnist_data[1][0]
valid_y = mnist_data[1][1]
test_x = mnist_data[2][0]
test_y = mnist_data[2][1]


print (train_x.shape)
print (train_y.shape)
print (valid_x.shape)
print (valid_y.shape)


fsadfsa


#Load model







#Train model
random_seed=1
rs=npr.RandomState(random_seed)
n_datapoints = len(train_y)
arr = np.arange(n_datapoints)

if path_to_load_variables == '':
    self.sess.run(self.init_vars)

else:
    #Load variables
    self.saver.restore(self.sess, path_to_load_variables)
    print 'loaded variables ' + path_to_load_variables

#start = time.time()
for epoch in range(epochs):

    #shuffle the data
    rs.shuffle(arr)
    train_x = train_x[arr]
    train_y = train_y[arr]

    data_index = 0
    for step in range(n_datapoints/batch_size):

        #Make batch
        batch = []
        batch_y = []
        while len(batch) != batch_size:
            batch.append(train_x[data_index]) 
            one_hot=np.zeros(10)
            one_hot[train_y[data_index]]=1.
            batch_y.append(one_hot)
            data_index +=1

        # Fit training using batch data
        _ = self.sess.run((self.optimizer), feed_dict={self.x: batch, self.y: batch_y, 
                                                self.batch_size: batch_size})

        # Display logs per epoch step
        if step % display_step == 0:

            cost,pred = self.sess.run((self.cost,self.prediction), feed_dict={self.x: batch, self.y: batch_y, 
                                                self.batch_size: batch_size})
            # cost = -cost #because I want to see the NLL
            print "Epoch", str(epoch+1)+'/'+str(epochs), 'Step:%04d' % (step+1) +'/'+ str(n_datapoints/batch_size), "elbo=", "{:.6f}".format(float(cost))#,logpy,logpW,logqW #, 'time', time.time() - start
            # print 'target'
            print ["{:.2f}".format(float(x)) for x in batch_y[0]] 
            # print 'prediciton'
            # print pred.shape
            print ["{:.2f}".format(float(x)) for x in pred[0]] 
            print

if path_to_save_variables != '':
    self.saver.save(self.sess, path_to_save_variables)
    print 'Saved variables to ' + path_to_save_variables





