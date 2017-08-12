

# all models will share the same train function, which does early stopping

# import all models  L2, Dropout, BNN, BNN+
# Train each model with different hyperparameters
# For each model, take its best hyperparams, eval on test 
# Done, each model has a score. 



import numpy as np
import tensorflow as tf
from os.path import expanduser
home = expanduser("~")

# import matplotlib.pyplot as plt

import time, datetime

# from BNN import BNN

import load_data

    
import L2_NN
import dropout_NN
import B_NN







def train(model, train_x, train_y, valid_x=[], valid_y=[], 
                path_to_load_variables='', path_to_save_variables='', 
                epochs=10, batch_size=20,  display_step=5):
    '''
    Train.
    '''

    random_seed=1
    rs=np.random.RandomState(random_seed)
    n_datapoints = len(train_y)
    arr = np.arange(n_datapoints)

    if path_to_load_variables == '':
        model.sess.run(model.init_vars)

    else:
        #Load variables
        model.saver.restore(model.sess, path_to_load_variables)
        print 'loaded variables ' + path_to_load_variables

    #start = time.time()
    best_valid_se = None
    n_times_no_improvement = 0
    max_n_times_no_improvement = 5
    for epoch in range(1,epochs+1):

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
                batch_y.append(train_y[data_index])
                data_index +=1

            # Fit training using batch data
            _ = model.sess.run((model.optimizer), feed_dict={model.x: batch, model.y: batch_y})

            # Display logs per epoch step
            if epoch % display_step == 1 and step==0:

                cost, se, reg, acc = model.sess.run((model.cost, model.softmax_error, model.reg, model.acc), 
                                            feed_dict={model.x: train_x, model.y: train_y})
                vse, vacc = model.sess.run((model.softmax_error, model.acc), 
                                            feed_dict={model.x: valid_x, model.y: valid_y})


                if path_to_save_variables != '' and \
                    (vse < best_valid_se or best_valid_se==None):

                    model.saver.save(model.sess, path_to_save_variables)
                    saved='saved'
                    best_valid_se = vse
                    n_times_no_improvement =0
                else:
                    n_times_no_improvement +=1
                    saved=str(n_times_no_improvement)

                print "Epoch", str(epoch)+'/'+str(epochs), 'Step:%02d' % (step+1) +'/' \
                        + str(n_datapoints/batch_size) \
                        + " cost={:.4f}".format(float(cost)) + " se={:.3f}".format(se) \
                        + " reg={:.3f}".format(reg) + " acc={:.3f}".format(acc) \
                        + " Validation: se={:.3f}".format(vse) + " acc={:.3f}".format(vacc) + ' ' +saved



        if n_times_no_improvement > max_n_times_no_improvement:
            break

    print 'Best Validation SE: ' + str(best_valid_se) 
    return best_valid_se



def evaluation(model, data_x, data_y):

    se, acc = model.sess.run((model.softmax_error, model.acc), 
                                        feed_dict={model.x: data_x, model.y: data_y})

    return se, acc



def load_vars(model, path_to_load_variables):

    #Load variables
    model.saver.restore(model.sess, path_to_load_variables)
    print 'loaded variables ' + path_to_load_variables















if __name__ == '__main__':
 

    # save_log = 0
    L2 = 1
    drop = 1
    bnn = 1


    train_x, valid_x, test_x, train_y, valid_y, test_y = load_data.load_mnist()
    train_x, train_y = load_data.get_equal_each_class(n_classes=10, n_datapoints_each_class=100, 
                                                        data_x=train_x, data_y=train_y)
    valid_x, valid_y = load_data.get_equal_each_class(n_classes=10, n_datapoints_each_class=20, 
                                                        data_x=valid_x, data_y=valid_y)

    print train_x.shape
    print train_y.shape
    print valid_x.shape
    print valid_y.shape

    network_architecture = [784, 100, 100, 10]
    act_functions=[tf.nn.softplus,tf.nn.softplus, None]

    display_step = 300
    batch_size = 100
    epochs = 6000

    model_names = ['L2', 'Dropout', 'BNN']
    model_scores = [None, None, None]


    if L2:
        print '\n\nTrain L2 model'
        model_hypers = [.0000001, .00001, .001]

        best_hyper_index = None
        best_hyper_score = None

        for hyper_i in range(len(model_hypers)):

            print hyper_i, model_hypers[hyper_i]

            path_to_load_variables=''
            # path_to_save_variables=home+'/Documents/tmp/vars_5_classes_relu.ckpt'
            path_to_save_variables=home+'/Documents/tmp/vars_L2_'+ str(hyper_i) + '.ckpt'
            # path_to_save_variables=''


            model = L2_NN.L2_NN(network_architecture=network_architecture, 
                                act_functions=act_functions, 
                                lmba=model_hypers[hyper_i])

            best_valid_se = train(model, train_x, train_y, valid_x=valid_x, valid_y=valid_y, 
                        path_to_load_variables=path_to_load_variables, path_to_save_variables=path_to_save_variables, 
                        epochs=epochs, batch_size=batch_size, display_step=display_step)


            if best_hyper_score == None or best_valid_se < best_hyper_score:

                best_hyper_index = hyper_i
                best_hyper_score = best_valid_se
                print 'This is best hyper atm'
            else:
                print 'This is not better'


        #Get test score for best model
        print 'Testing hyepr ', hyper_i, model_hypers[hyper_i]
        model = L2_NN.L2_NN(network_architecture=network_architecture, 
                                act_functions=act_functions, 
                                lmba=model_hypers[best_hyper_index])
        load_vars(model, path_to_load_variables=home+'/Documents/tmp/vars_L2_'+ str(best_hyper_index) + '.ckpt')
        se, acc = evaluation(model, test_x, test_y)
        print 'Test SE:' + str(se) + ' ACC:' + str(acc)

        model_scores[0] = se

        print model_names
        print model_scores






    if drop:

        print '\n\nTrain Dropout model'

        model_hypers = [.1,.5,.9]

        best_hyper_index = None
        best_hyper_score = None

        for hyper_i in range(len(model_hypers)):

            print hyper_i, model_hypers[hyper_i]

            path_to_load_variables=''
            # path_to_save_variables=home+'/Documents/tmp/vars_5_classes_relu.ckpt'
            path_to_save_variables=home+'/Documents/tmp/vars_drop_'+ str(hyper_i) + '.ckpt'
            # path_to_save_variables=''


            model = dropout_NN.dropout_NN(network_architecture=network_architecture, 
                                act_functions=act_functions, 
                                lmba=.0000001,
                                keep_prob=model_hypers[hyper_i])

            best_valid_se = train(model, train_x, train_y, valid_x=valid_x, valid_y=valid_y, 
                        path_to_load_variables=path_to_load_variables, path_to_save_variables=path_to_save_variables, 
                        epochs=epochs, batch_size=batch_size, display_step=display_step)


            if best_hyper_score == None or best_valid_se < best_hyper_score:

                best_hyper_index = hyper_i
                best_hyper_score = best_valid_se
                print 'This is best hyper atm'
            else:
                print 'This is not better'


        #Get test score for best model
        print 'Testing hyper ', hyper_i, model_hypers[hyper_i]
        model = dropout_NN.dropout_NN(network_architecture=network_architecture, 
                                act_functions=act_functions, 
                                lmba=.0000001,
                                keep_prob=model_hypers[best_hyper_index])

        load_vars(model, path_to_load_variables=home+'/Documents/tmp/vars_drop_'+ str(best_hyper_index) + '.ckpt')
        se, acc = evaluation(model, test_x, test_y)
        print 'Test SE:' + str(se) + ' ACC:' + str(acc)


        model_scores[1] = se
        print model_names
        print model_scores






    if bnn:

        print '\n\nTrain BNN model'

        model_hypers = [1000.,10000.,100000.]

        best_hyper_index = None
        best_hyper_score = None

        for hyper_i in range(len(model_hypers)):

            print hyper_i, model_hypers[hyper_i]

            path_to_load_variables=''
            # path_to_save_variables=home+'/Documents/tmp/vars_5_classes_relu.ckpt'
            path_to_save_variables=home+'/Documents/tmp/vars_bnn_'+ str(hyper_i) + '.ckpt'
            # path_to_save_variables=''


            model = B_NN.B_NN(network_architecture=network_architecture, 
                                act_functions=act_functions, 
                                lmba=.0000001,
                                prior_var=model_hypers[hyper_i])

            best_valid_se = train(model, train_x, train_y, valid_x=valid_x, valid_y=valid_y, 
                        path_to_load_variables=path_to_load_variables, path_to_save_variables=path_to_save_variables, 
                        epochs=epochs, batch_size=batch_size, display_step=display_step)


            if best_hyper_score == None or best_valid_se < best_hyper_score:

                best_hyper_index = hyper_i
                best_hyper_score = best_valid_se
                print 'This is best hyper atm'
            else:
                print 'This is not better'


        #Get test score for best model
        print 'Testing hyper ', hyper_i, model_hypers[hyper_i]
        model = B_NN.B_NN(network_architecture=network_architecture, 
                                act_functions=act_functions, 
                                lmba=.0000001,
                                prior_var=model_hypers[best_hyper_index])

        load_vars(model, path_to_load_variables=home+'/Documents/tmp/vars_bnn_'+ str(best_hyper_index) + '.ckpt')
        se, acc = evaluation(model, test_x, test_y)
        print 'Test SE:' + str(se) + ' ACC:' + str(acc)


        model_scores[2] = se
        print model_names
        print model_scores







    # #Experiment log
    # if save_log:
    #     dt = datetime.datetime.now()
    #     date_ = str(dt.date())
    #     time_ = str(dt.time())
    #     time_2 = time_[0] + time_[1] + time_[3] + time_[4] + time_[6] + time_[7] 
    #     experiment_log = experiment_log_path+'experiment_' + date_ + '_' +time_2 +'.txt'
    #     print 'Saving experiment log to ' + experiment_log








    print 'Done.'







