



# code modified from here: https://github.com/ericjang/gumbel-softmax/blob/master/gumbel_softmax_vae_v2.ipynb


# adding another latent layer

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

K2=10
N2=1

straight_through=False # if True, use Straight-through Gumbel-Softmax
kl_type='relaxed' # choose between ('relaxed', 'categorical')
learn_temp=False
tau = tf.Variable(tau0,name="temperature",trainable=learn_temp)


# INFERENCE
x=tf.placeholder(tf.float32, shape=(batch_size,784), name='x')
net = tf.cast(tf.random_uniform(tf.shape(x)) < x, x.dtype) # dynamic binarization
net = slim.stack(net,slim.fully_connected,[512,256])

# q(c|x)
logits_c = tf.reshape(slim.fully_connected(net,K*N,activation_fn=None),[-1,N,K])
q_c = RelaxedOneHotCategorical(tau,logits_c)
c = q_c.sample()  #[B,N,K] 
q_c_eval = q_c.log_prob(c) #[B,NK]
c_flat = slim.flatten(c)  #[B,NK]

net2 = slim.fully_connected(c_flat,100)

# q(z|x,c)
logits_z = tf.reshape(slim.fully_connected(tf.concat([net,net2],axis=1),K2*N2,activation_fn=None),[-1,N2,K2])
q_z = RelaxedOneHotCategorical(tau,logits_z)
z = q_z.sample() #[B,N2,K2] 
q_z_eval = q_z.log_prob(z) #[B,N2K2]
z_flat = slim.flatten(z)  #[B,N2*K2] 

# logits_y = tf.reshape(slim.fully_connected(net,K*N,activation_fn=None),[-1,N,K])
# q_y = RelaxedOneHotCategorical(tau,logits_y)
# y = q_y.sample()

# if straight_through:
#   y_hard = tf.cast(tf.one_hot(tf.argmax(y,-1),K), y.dtype)
#   y = tf.stop_gradient(y_hard - y) + y
# net = slim.flatten(y)





# DECODER

# p(x|z,c)
net4 = slim.stack(tf.concat([c_flat,z_flat],axis=1),slim.fully_connected,[256,512])
logits_x = slim.fully_connected(net4,784,activation_fn=None)
p_x = Bernoulli(logits=logits_x)
p_x_eval = p_x.log_prob(x)

 # tf.reduce_sum(p_x.log_prob(x),1)

# p(c)
logits_pc = tf.ones_like(logits_c) * 1./K   #[B,N,K]
p_c = RelaxedOneHotCategorical(tau,logits=logits_pc)
# c=tf.reshape(c, [batch_size, N, K]) #[B,N,K]
p_c_eval = p_c.log_prob(c) #[B,NK]


# p(z|c)
net3 = slim.fully_connected(c,100)
logits_z3 = tf.reshape(slim.fully_connected(net3,K2*N2,activation_fn=None),[-1,N2,K2])
p_z = RelaxedOneHotCategorical(tau,logits_z3)
# z=tf.reshape(z, [batch_size, N2, K2]) #[B,N2,K2]
p_z_eval = p_z.log_prob(z) #[B,N2K2]



#for sampling prior
z2 = p_z.sample() #[B,N2,K2] 
x_mean = p_x.mean()

# net4 = slim.stack(tf.concat([c,z],axis=1),slim.fully_connected,[256,512])
# logits_x = slim.fully_connected(net4,784,activation_fn=None)
# p_x = Bernoulli(logits=logits_x)
# x_mean2 = p_x.mean()


# if kl_type=='categorical' or straight_through:
#   # Analytical KL with Categorical prior
#   p_cat_y = OneHotCategorical(logits=logits_py)
#   q_cat_y = OneHotCategorical(logits=logits_y)
#   KL_qp = tf.contrib.distributions.kl(q_cat_y, p_cat_y)
# else:
#   # Monte Carlo KL with Relaxed prior

# print(q_c_eval)
# print(q_z_eval)
# print(p_c_eval)
# print(p_z_eval)
# print(p_x_eval)

# fasd

# KL_qp = q_y.log_prob(y) - p_y.log_prob(y)



# KL = tf.reduce_sum(KL_qp,1)
# mean_recons = tf.reduce_mean(p_x_eval)
# mean_KL = tf.reduce_mean(KL)

q_c_eval = tf.reduce_sum(q_c_eval,1)
q_z_eval = tf.reduce_sum(q_z_eval,1)
p_c_eval = tf.reduce_sum(p_c_eval,1)
p_z_eval = tf.reduce_sum(p_z_eval,1)
p_x_eval = tf.reduce_sum(p_x_eval,1)


elbo = p_x_eval + p_z_eval + p_c_eval - q_z_eval - q_c_eval #[B]


loss = -tf.reduce_mean(elbo)  #over batch






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
    # res = sess.run([train_op, loss, tau, mean_recons, mean_KL], {x : batch[0]})
    res = sess.run([train_op, loss], {x : batch[0]})


    if i % 100 == 1:
      data.append([i] + res[1:])
    if i % 1000 == 1:
      print('Step %d, Loss: %0.3f' % (i,res[1]))

    if i % 5000 == 1:






      # end training - do an eval
      plt.clf()

      data1 = np.array(data).T

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

      print('1111')

      z2_ = sess.run(z2, {c : y_})
      print('2222')
      prior_x_ = sess.run(x_mean, {c:y_, z:z2_})
      print('3333')
      


      axarr[1].set_title('Prior Samples')
      # print (batch[0].shape)
      tmp = np.reshape(prior_x_,(-1,280,28)) # (10,280,28)
      # print (tmp.shape)
      img = np.hstack([tmp[i] for i in range(10)])
      # print (img.shape)
      # axarr[1].imshow(img.T)
      axarr[1].imshow(img)





      batch = mnist.test.next_batch(batch_size)
      np_x = sess.run(x_mean, {x : batch[0]})


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











