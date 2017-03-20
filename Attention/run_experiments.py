

import numpy as np
import pickle
from scipy.misc import imsave


from os.path import expanduser
home = expanduser("~")

from VAE_IWAE_attention_labels import VAE



def load_binarized_mnist(location):

    with open(location, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f)

    if 'binarized' in location:

        train_x = train_set
        valid_x = valid_set
        test_x = test_set

        return train_x, valid_x, test_x

    else:

        train_x = train_set[0]
        valid_x = valid_set[0]
        test_x = test_set[0]

        train_y = train_set[1]
        valid_y = valid_set[1]
        test_y = test_set[1]

        return train_x, valid_x, test_x, train_y, valid_y, test_y



if __name__ == "__main__":

    #Which task to run
    train_model = 0
    visualize = 1


    save_to = home + '/Documents/tmp/'

    # model_path_to_load_variables=save_to + 'attention2.ckpt'
    model_path_to_load_variables=''
    model_path_to_save_variables=save_to + 'attention3.ckpt'
    # model_path_to_save_variables=''


    if train_model ==1:

        #Load data
        # train_x, valid_x, test_x = load_binarized_mnist(location=home+'/data/binarized_mnist.pkl')
        # train_x, valid_x, test_x = load_binarized_mnist(location=home+'/Documents/MNIST_Data/binarized_mnist.pkl')
        train_x, valid_x, test_x, train_y, valid_y, test_y = load_binarized_mnist(location=home+'/Documents/MNIST_Data/mnist.pkl')

        print 'Train', train_x.shape
        print 'Valid', valid_x.shape
        print 'Test', test_x.shape
        print 'Train', train_y.shape
        print 'Valid', valid_y.shape
        print 'Test', test_y.shape

        #Init model
        vae = VAE(batch_size=25)
        #Train model
        vae.train(train_x=train_x, path_to_load_variables=model_path_to_load_variables, 
                                    path_to_save_variables=model_path_to_save_variables,
                                    epochs=10,
                                    train_y=train_y)



    if visualize ==1:

        #Load data
        # train_x, valid_x, test_x = load_binarized_mnist(location=home+'/data/binarized_mnist.pkl')
        # train_x, valid_x, test_x = load_binarized_mnist(location=home+'/Documents/MNIST_Data/binarized_mnist.pkl')
        train_x, valid_x, test_x, train_y, valid_y, test_y = load_binarized_mnist(location=home+'/Documents/MNIST_Data/mnist.pkl')

        print 'Train', train_x.shape
        print 'Valid', valid_x.shape
        print 'Test', test_x.shape
        print 'Train', train_y.shape
        print 'Valid', valid_y.shape
        print 'Test', test_y.shape


        # #Put them together
        # for i in range(5):

        #     real = np.reshape(train_x[i], [28,28])
        #     recon = np.reshape(train_x2[i], [28,28])
        #     together = np.concatenate((real, recon), axis=1)

        #     if i == 0:
        #         concat2 = together
        #     else:
        #         concat2 = np.concatenate((concat2, together), axis=0)


        # # print samps.shape

        # # for i in range(len(5)):

        # #     gen = np.reshape(samps[i], [28,28])
        # #     other = np.zeros((28,28))
        # #     together = np.concatenate((other, gen), axis=1)

        # #     concat2 = np.concatenate((concat2, together), axis=0)



        # imsave(home+'/Documents/tmp/pic.png', concat2)
        # print 'saved ' + home+'/Documents/tmp/pic.png'


        #Init model
        vae = VAE(batch_size=3)
        vae.load_parameters(path_to_load_variables=model_path_to_save_variables)

        #Get a bunch of reconstructions 
        recons, batch = vae.reconstruct(sampling='vae', data=valid_x)
        # recons = recons[0]
        print recons.shape
        print batch.shape

        #Put them together
        for i in range(len(batch)):

            real = np.reshape(batch[i], [28,28])
            recon = np.reshape(recons[i], [28,28])
            together = np.concatenate((real, recon), axis=1)

            if i == 0:
                concat2 = together
            else:
                concat2 = np.concatenate((concat2, together), axis=0)


        #Get a bunch of generations
        samps = vae.generate()
        # samps = samps[0]
        print samps.shape

        for i in range(len(samps)):

            gen = np.reshape(samps[i], [28,28])
            other = np.zeros((28,28))
            together = np.concatenate((other, gen), axis=1)

            concat2 = np.concatenate((concat2, together), axis=0)



        imsave(home+'/Documents/tmp/pic.png', concat2)
        print 'saved ' + home+'/Documents/tmp/pic.png'




    print 'Done everything'








