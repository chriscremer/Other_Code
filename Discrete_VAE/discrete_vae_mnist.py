

# code modified from here: https://github.com/ericjang/gumbel-softmax/blob/master/gumbel_softmax_vae_v2.ipynb


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

slim=tf.contrib.slim
Bernoulli = tf.contrib.distributions.Bernoulli
OneHotCategorical = tf.contrib.distributions.OneHotCategorical
RelaxedOneHotCategorical = tf.contrib.distributions.RelaxedOneHotCategorical



# black-on-white MNIST (harder to learn than white-on-black MNIST)
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)






batch_size=100
tau0=1.0 # initial temperature
K=10 # number of classes
# N=200//K # number of categorical distributions
N=1 # number of categorical distributions

straight_through=False # if True, use Straight-through Gumbel-Softmax
kl_type='relaxed' # choose between ('relaxed', 'categorical')
learn_temp=False






x=tf.placeholder(tf.float32, shape=(batch_size,784), name='x')
net = tf.cast(tf.random_uniform(tf.shape(x)) < x, x.dtype) # dynamic binarization
net = slim.stack(net,slim.fully_connected,[512,256])
logits_y = tf.reshape(slim.fully_connected(net,K*N,activation_fn=None),[-1,N,K])
tau = tf.Variable(tau0,name="temperature",trainable=learn_temp)
q_y = RelaxedOneHotCategorical(tau,logits_y)
y = q_y.sample()
if straight_through:
  y_hard = tf.cast(tf.one_hot(tf.argmax(y,-1),K), y.dtype)
  y = tf.stop_gradient(y_hard - y) + y
net = slim.flatten(y)
net = slim.stack(net,slim.fully_connected,[256,512])
logits_x = slim.fully_connected(net,784,activation_fn=None)
p_x = Bernoulli(logits=logits_x)
x_mean = p_x.mean()



recons = tf.reduce_sum(p_x.log_prob(x),1)
logits_py = tf.ones_like(logits_y) * 1./K

if kl_type=='categorical' or straight_through:
  # Analytical KL with Categorical prior
  p_cat_y = OneHotCategorical(logits=logits_py)
  q_cat_y = OneHotCategorical(logits=logits_y)
  KL_qp = tf.contrib.distributions.kl(q_cat_y, p_cat_y)
else:
  # Monte Carlo KL with Relaxed prior
  p_y = RelaxedOneHotCategorical(tau,logits=logits_py)
  KL_qp = q_y.log_prob(y) - p_y.log_prob(y)



KL = tf.reduce_sum(KL_qp,1)
mean_recons = tf.reduce_mean(recons)
mean_KL = tf.reduce_mean(KL)
loss = -tf.reduce_mean(recons-KL)






train_op=tf.train.AdamOptimizer(learning_rate=3e-4).minimize(loss)




#for sampling prior
y_ = np.zeros((batch_size, K))
for j in range(10):
  y_.T[j][j*10:10*(j+1)] = 1.
y_ = np.reshape(y_, [batch_size, N, K])





data = []
with tf.train.MonitoredSession() as sess:
  for i in range(1,50000):
  # for i in range(1,1004):

    batch = mnist.train.next_batch(batch_size)
    res = sess.run([train_op, loss, tau, mean_recons, mean_KL], {x : batch[0]})
    if i % 100 == 1:
      data.append([i] + res[1:])
    if i % 1000 == 1:
      print('Step %d, Loss: %0.3f' % (i,res[1]))

    if i % 5000 == 1:






      # end training - do an eval
      batch = mnist.test.next_batch(batch_size)
      np_x = sess.run(x_mean, {x : batch[0]})

      data1 = np.array(data).T

      plt.clf()

      f,axarr=plt.subplots(1,4,figsize=(18,6))
      axarr[0].plot(data1[0],data1[1])
      axarr[0].set_title('Loss')

      # axarr[1].plot(data[0],data[2])
      # axarr[1].set_title('Temperature')

      # axarr[2].plot(data[0],data[3])
      # axarr[2].set_title('Recons')

      # axarr[3].plot(data[0],data[4])
      # axarr[3].set_title('KL')


      # axarr[1].plot(data1[0],data1[3])
      # axarr[1].set_title('Recons')

      # axarr[2].plot(data1[0],data1[4])
      # axarr[2].set_title('KL')



      prior_x_ = sess.run(x_mean, {y : y_})

      axarr[1].set_title('Prior Samples')
      # print (batch[0].shape)
      tmp = np.reshape(prior_x_,(-1,280,28)) # (10,280,28)
      # print (tmp.shape)
      img = np.hstack([tmp[i] for i in range(10)])
      # print (img.shape)
      # axarr[1].imshow(img.T)
      axarr[1].imshow(img)






      axarr[2].set_title('Test set')
      # print (batch[0].shape)
      tmp = np.reshape(batch[0],(-1,280,28)) # (10,280,28)
      # print (tmp.shape)
      img = np.hstack([tmp[i] for i in range(10)])
      # print (img.shape)
      axarr[2].imshow(img)




      axarr[3].set_title('Recons')

      tmp = np.reshape(np_x,(-1,280,28)) # (10,280,28)
      img = np.hstack([tmp[i] for i in range(10)])
      axarr[3].imshow(img)

      plt.show()
      plt.grid('off')











