



#need to resample every x steps. 
#also need to keep track of ancestors, or do I? I dont think so . Not for the elbo atleast


#works. 


import numpy as np
import pickle
from os.path import expanduser
home = expanduser("~")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from utils import lognormal3 as lognormal
from utils import log_bernoulli


import ball_up_down_actions as buda




def train(model, get_data, valid_x=[], 
            path_to_load_variables='', path_to_save_variables='', 
            steps=1000, batch_size=20, display_step=10, k=1):

    if path_to_load_variables != '':
        model.load_state_dict(torch.load(path_to_load_variables))
        print 'loaded variables ' + path_to_load_variables

    # train = torch.utils.data.TensorDataset(train_x, train_y)
    # train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=.001)

    for step in range(1,steps+1):

        batch = []
        batch_actions = []
        while len(batch) != batch_size:
            sequence, actions=get_data()
            batch.append(sequence)
            batch_actions.append(actions)

        # if data.is_cuda:
        #     data, target = Variable(batch), Variable(target)
        # else:
        observations, actions = Variable(torch.from_numpy(np.array(batch))), Variable(torch.from_numpy(np.array(batch_actions)))

        optimizer.zero_grad()
        elbo, logpx, logpz, logqz = model.forward(observations, actions, k=k)
        loss = -(elbo)
        loss.backward()
        optimizer.step()

        if step%display_step==0:
            print step, 'Elbo:{:.4f}'.format(elbo.data[0]), \
                'logpx:{:.4f}'.format(logpx.data[0]), \
                'logpz:{:.4f}'.format(logpz.data[0]), \
                'logqz:{:.4f}'.format(logqz.data[0]) 


    if path_to_save_variables != '':
        torch.save(model.state_dict(), path_to_save_variables)
        print 'Saved variables to ' + path_to_save_variables




# def test(model, data_x, path_to_load_variables='', batch_size=20, display_epoch=4, k=10):
    # shoudl return the iwae elbo likelihood

#     if path_to_load_variables != '':
#         model.load_state_dict(torch.load(path_to_load_variables))
#         print 'loaded variables ' + path_to_load_variables

#     elbos = []
#     data_index= 0
#     for i in range(len(data_x)/ batch_size):

#         batch = data_x[data_index:data_index+batch_size]
#         data_index += batch_size

#         elbo, logpx, logpz, logqz = model(Variable(batch), k=k)
#         elbos.append(elbo.data[0])

#         if i%display_epoch==0:
#             print i,len(data_x)/ batch_size, elbo.data[0]

#     return np.mean(elbos)



def visulize_predictions(model, get_data):

    k=3

    batch = []
    batch_actions = []
    while len(batch) != 1:
        sequence, actions=get_data()
        batch.append(sequence)
        batch_actions.append(actions)

    #chop up the sequence, only give it first 3 frames
    n_time_steps_given = 4
    given_sequence = []
    given_actions = []
    hidden_sequence = []
    hidden_actions = []
    for b in range(len(batch)):
        given_sequence.append(batch[b][:n_time_steps_given])
        given_actions.append(batch_actions[b][:n_time_steps_given])
        #and the ones that are not used
        hidden_sequence.append(batch[b][n_time_steps_given:])
        hidden_actions.append(batch_actions[b][n_time_steps_given:])

    given_sequence = Variable(torch.from_numpy(np.array(given_sequence)))
    given_actions = Variable(torch.from_numpy(np.array(given_actions)))

    #Get states [Tg,P,B,Z]
    states = model.return_current_state(given_sequence, given_actions, k=k)
    # print len(states)
    # print len(states[0])
    # print len(states[0][0])
    # print len(states[0][0][0])

    # current_state = self.sess.run(self.particles_and_logprobs, feed_dict={self.x: given_sequence, self.actions: given_actions})
    # Unpack, get states
    # current_state = current_state[n_time_steps_given-1][0][:self.z_size*self.n_particles]

    #get the last state  [P,B,Z]
    current_state = states[n_time_steps_given-1]

    # print len(current_state)
    # print len(current_state[0])
    # print len(current_state[0][0])
    # print len(current_state[0][0][0])


    #Step 2: Predict future states and decode to frames

    # print np.array(hidden_actions).shape #[B,leftovertime, A]
    # current_state = [current_state] #so it fits batch
    
    #[B,Th,A]
    hidden_actions = Variable(torch.from_numpy(np.array(hidden_actions)))
    # print hidden_actions.size()
    # fds


    # [TL, P, B, X]
    obs = model.predict_future(current_state, hidden_actions) 

    # print len(obs)
    # print len(obs[0])
    # print len(obs[0][0])
    # print len(obs[0][0][0])
    # fsadf
    given_sequence = given_sequence.data.numpy()

    # [P, T-TL, X]
    real_and_gen = []
    for p in range(k):
        real_and_gen.append(list(given_sequence[0]))

    for obs_t in range(len(obs)):
        # print 'obs_t', obs_t
        for p in range(len(obs[obs_t])):
            # print 'p', p

            obs_t_p = np.reshape(obs[obs_t][p][0], [-1])

            real_and_gen[p].append(obs_t_p)

    #[P,T,X]
    real_and_gen = np.array(real_and_gen)
    # print real_and_gen.shape #[T,X]
    
    real_sequence = np.array(batch[0])
    actions = np.array(batch_actions[0])

    # print real_and_gen[0][8]
    # # fsd[6][0]
    # fsd

    return real_sequence, actions, real_and_gen





def load_params(model, path_to_load_variables):

    if path_to_load_variables != '':
        model.load_state_dict(torch.load(path_to_load_variables))
        print 'loaded variables ' + path_to_load_variables























class FIVO(nn.Module):
    def __init__(self, specs):
        super(FIVO, self).__init__()


        self.input_size = specs['n_input']
        self.z_size = specs['n_z']
        self.action_size = specs['n_actions']

        self.encoder_net = torch.nn.Sequential(
            torch.nn.Linear(self.input_size+self.z_size+self.action_size, specs['encoder_net'][0]),
            torch.nn.ReLU(),
            torch.nn.Linear(specs['encoder_net'][0], specs['encoder_net'][1]),
            torch.nn.ReLU(),
            torch.nn.Linear(specs['encoder_net'][1], self.z_size*2),
        )

        self.decoder_net = torch.nn.Sequential(
            torch.nn.Linear(self.z_size, specs['decoder_net'][0]),
            torch.nn.ReLU(),
            torch.nn.Linear(specs['decoder_net'][0], specs['decoder_net'][1]),
            torch.nn.ReLU(),
            torch.nn.Linear(specs['decoder_net'][1], self.input_size),
        )

        self.transition_net = torch.nn.Sequential(
            torch.nn.Linear(self.z_size+self.action_size, specs['trans_net'][0]),
            torch.nn.ReLU(),
            torch.nn.Linear(specs['trans_net'][0], specs['trans_net'][1]),
            torch.nn.ReLU(),
            torch.nn.Linear(specs['trans_net'][1], self.z_size*2),
        )



    def encode(self, x, a, prev_z):
        '''
        x: [B,X]
        a: [B,A]
        prev_z: [P,B,Z]
        '''

        out = torch.cat((x,a), 1) #[B,XA]
        out = torch.unsqueeze(out, 0) #[1,B,XA]
        out = out.repeat(self.k, 1, 1)#.float() #[P,B,XA]
        out = torch.cat((out,prev_z), 2) #[P,B,XAZ]
        out = out.view(self.k*self.B, self.input_size+self.action_size+self.z_size) #[P*B,XAZ]
        out = self.encoder_net(out) #[P*B,Z*2]
        out = out.view(self.k, self.B, 2*self.z_size) #[P,B,XAZ]
        mean = out[:,:,:self.z_size]
        logvar = out[:,:,self.z_size:]
        return mean, logvar


    def sample(self, mu, logvar):
        '''
        mu, logvar: [P,B,Z]
        '''
        eps = Variable(torch.FloatTensor(self.k, self.B, self.z_size).normal_()) #[P,B,Z]
        z = eps.mul(torch.exp(.5*logvar)) + mu  #[P,B,Z]
        # logpz = lognormal(z, Variable(torch.zeros(self.k, self.B, self.z_size)), 
        #                     Variable(torch.zeros(self.k, self.B, self.z_size)))  #[P,B]
        logqz = lognormal(z, mu, logvar)
        return z, logqz


    def decode(self, z):
        '''
        z: [P,B,Z]
        '''
        out = z.view(-1, self.z_size) #[P*B,Z]
        out = self.decoder_net(out) #[P*B,X]
        out = out.view(self.k, self.B, self.input_size) #[P,B,X]
        return out


    def transition_prior(self, prev_z, a):
        '''
        prev_z: [P,B,Z]
        a: [B,A]
        '''
        a = torch.unsqueeze(a, 0) #[1,B,A]
        out = a.repeat(self.k, 1, 1) #[P,B,A]
        out = torch.cat((out,prev_z), 2) #[P,B,AZ]
        out = out.view(-1, self.action_size+self.z_size) #[P*B,AZ]
        out = self.transition_net(out) #[P*B,Z*2]
        out = out.view(self.k, self.B, 2*self.z_size) #[P,B,2Z]
        mean = out[:,:,:self.z_size] #[P,B,Z]
        logvar = out[:,:,self.z_size:]
        return mean, logvar


    def forward(self, x, a, k=1, current_state=None):
        '''
        x: [B,T,X]
        a: [B,T,A]
        output: elbo scalar
        '''
        
        self.B = x.size()[0]
        self.T = x.size()[1]
        self.k = k

        a = a.float()
        x = x.float()

        # log_probs = [[] for i in range(k)]
        # log_probs = []
        logpxs = []
        logpzs = []
        logqzs = []


        weights = Variable(torch.ones(k, self.B)/k)
        # if current_state==None:
        prev_z = Variable(torch.zeros(k, self.B, self.z_size))
        # else:
        #     prev_z = current_state
        for t in range(self.T):
            current_x = x[:,t] #[B,X]
            current_a = a[:,t] #[B,A]

            #Encode
            mu, logvar = self.encode(current_x, current_a, prev_z) #[P,B,Z]
            #Sample
            z, logqz = self.sample(mu, logvar) #[P,B,Z], [P,B]
            #Decode
            x_hat = self.decode(z)  #[P,B,X]
            logpx = log_bernoulli(x_hat, current_x)  #[P,B]
            #Transition/Prior prob
            prior_mean, prior_log_var = self.transition_prior(prev_z, current_a) #[P,B,Z]
            logpz = lognormal(z, prior_mean, prior_log_var) #[P,B]






            log_alpha_t = logpx + logpz - logqz #[P,B]
            log_weights_tmp = torch.log(weights * torch.exp(log_alpha_t))

            max_ = torch.max(log_weights_tmp, 0)[0] #[B]
            log_p_hat = torch.log(torch.sum(torch.exp(log_weights_tmp - max_), 0)) + max_ #[B]

            # p_hat = torch.sum(alpha_t,0)  #[B]
            normalized_alpha_t = log_weights_tmp - log_p_hat  #[P,B]

            weights = torch.exp(normalized_alpha_t) #[P,B]

            #if resample
            if t%2==0:
                # print weights
                #[B,P] indices of the particles for each bactch
                sampled_indices = torch.multinomial(torch.t(weights), k, replacement=True).detach()
                new_z = []
                for b in range(self.B):
                    tmp = z[:,b] #[P,Z]
                    z_b = tmp[sampled_indices[b]] #[P,Z]
                    new_z.append(z_b)
                new_z = torch.stack(new_z, 1) #[P,B,Z]
                weights = Variable(torch.ones(k, self.B)/k)
                z = new_z

            logpxs.append(logpx)
            logpzs.append(logpz)
            logqzs.append(logqz)
            # log_probs.append(logpx + logpz - logqz)
            prev_z = z



        logpxs = torch.stack(logpxs) 
        logpzs = torch.stack(logpzs)
        logqzs = torch.stack(logqzs) #[T,P,B]

        logws = logpxs + logpzs - logqzs  #[T,P,B]
        logws = torch.mean(logws, 0)  #[P,B]

        # elbo = logpx + logpz - logqz  #[P,B]

        if k>1:
            max_ = torch.max(logws, 0)[0] #[B]
            elbo = torch.log(torch.mean(torch.exp(logws - max_), 0)) + max_ #[B]
            elbo = torch.mean(elbo) #over batch
        else:
            elbo = torch.mean(logws)

        # print log_probs[0]


        # #for printing
        logpx = torch.mean(logpxs)
        logpz = torch.mean(logpzs)
        logqz = torch.mean(logqzs)
        # self.x_hat_sigmoid = F.sigmoid(x_hat)

        # elbo = torch.mean(torch.stack(log_probs)) #[1]
        # elbo = logpx + logpz - logqz

        return elbo, logpx, logpz, logqz




    def return_current_state(self, x, a, k):
        
        self.B = x.size()[0]
        self.T = x.size()[1]
        self.k = k
        a = a.float()
        x = x.float()
        states = []
        prev_z = Variable(torch.zeros(k, self.B, self.z_size))
        # prev_z = torch.zeros(k, self.B, self.z_size)
        for t in range(self.T):
            current_x = x[:,t] #[B,X]
            current_a = a[:,t] #[B,A]
            #Encode
            mu, logvar = self.encode(current_x, current_a, prev_z) #[P,B,Z]
            #Sample
            z, logqz = self.sample(mu, logvar) #[P,B,Z], [P,B]
            #Decode
            x_hat = self.decode(z)  #[P,B,X]
            logpx = log_bernoulli(x_hat, current_x)  #[P,B]
            #Transition/Prior prob
            prior_mean, prior_log_var = self.transition_prior(prev_z, current_a) #[P,B,Z]
            logpz = lognormal(z, prior_mean, prior_log_var) #[P,B]
            prev_z = z
            states.append(z)
        return states


    def predict_future(self, current_state, actions):
        '''
        current_state: [P,B,Z]
        actions: [B,T,A]
        output: [B,T,X]
        '''
        self.B = current_state.size()[1]
        self.T = actions.size()[1]
        self.k = current_state.size()[0]
        actions = actions.float()
        current_state = current_state.float()
        prev_z = current_state
        predicted_frames = []
        for t in range(self.T):
            current_a = actions[:,t] #[B,A]
            #Transition/Prior prob
            prior_mean, prior_log_var = self.transition_prior(prev_z, current_a) #[P,B,Z]
            #Sample
            z, logqz = self.sample(prior_mean, prior_log_var) #[P,B,Z], [P,B]
            #Decode
            x_hat = self.decode(z)  #[P,B,X]
            prev_z = z
            predicted_frames.append(F.sigmoid(x_hat).data.numpy())

        return predicted_frames





















if __name__ == "__main__":

    train_ = 1
    visualize = 1


    obs_height = 15
    obs_width = 2
    n_input = obs_height * obs_width
    model_specs = dict(
                encoder_net=[100,100],
                decoder_net=[100,100],
                trans_net=[100,100],
                n_input=n_input,
                n_z=20,  
                n_actions=3) 

    model = FIVO(model_specs)

    train_steps = 1000


    # path_to_load_variables=''
    path_to_load_variables=home+'/Documents/tmp/pytorch_fivo.pt'
    path_to_save_variables=home+'/Documents/tmp/pytorch_fivo.pt'
    # path_to_save_variables=''





    if train_:

        #Define the sequence
        n_timesteps = 10

        def get_data():
            sequence_obs, sequence_actions = buda.get_sequence(n_timesteps=n_timesteps, 
                                                obs_height=obs_height, obs_width=obs_width)
            return np.array(sequence_obs), np.array(sequence_actions)



        train(model=model, get_data=get_data, valid_x=[], 
                path_to_load_variables=path_to_load_variables, 
                path_to_save_variables=path_to_save_variables, 
                steps=train_steps, batch_size=4, display_step=50, k=3)

        print 'Done training'








    if visualize:

        if not train_:
            load_params(model, path_to_load_variables=path_to_save_variables)


        print 'Visualizing'
        viz_timesteps = 40
        viz_n_particles = 3
        viz_batch_size = 1

        def get_data():
            sequence_obs, sequence_actions = buda.get_sequence(n_timesteps=viz_timesteps, obs_height=obs_height, obs_width=obs_width)
            return np.array(sequence_obs), np.array(sequence_actions)
        #get actions and frames
        #give model all actions and only some frames

        # print 'Initializing model..'
        
        # dkf = DKF(network_architecture, batch_size=viz_batch_size, n_particles=viz_n_particles)


        # need to change it so i get mulitple trajectories , one for each particle

        real_sequence, actions, real_and_gen = visulize_predictions(model, get_data)

        # real_sequence, actions, real_and_gen = dkf.test(get_data=get_data, path_to_load_variables=path_to_save_variables)

        # print real_sequence.shape #[T,X]
        # print actions.shape #[T,A]
        # print real_and_gen.shape #[P, T, X]

        # fdsaf


        fig = plt.figure(figsize=(6, 8))

        per_traj = 3
        offset = per_traj+2 #2 for the actions, per_traj for real
        G = gridspec.GridSpec(per_traj*(viz_n_particles+1)+offset, 3) # +1 for avg

        axes_1 = plt.subplot(G[0:2, 1])
        plt.imshow(actions.T, vmin=0, vmax=1, cmap="gray")
        plt.ylabel('Actions', size=10)
        plt.yticks([])
        plt.xticks(size=7)

        axes_2 = plt.subplot(G[2:offset, :])
        plt.imshow(real_sequence.T, vmin=0, vmax=1, cmap="gray")
        plt.ylabel('True Trajectory', size=10)
        plt.yticks([])
        plt.xticks([])

        avg_traj = np.zeros((obs_width*obs_height, viz_timesteps))
        for p in range(len(real_and_gen)):

            plt.subplot(G[offset+(p*per_traj):offset+(p*per_traj)+per_traj, :])
            plt.imshow(real_and_gen[p].T, vmin=0, vmax=1, cmap="gray")
            plt.ylabel('Trajectory ' + str(p), size=10)
            plt.yticks([])
            plt.xticks([])

            avg_traj += real_and_gen[p].T 

            if p == len(real_and_gen)-1:
                plt.subplot(G[offset+((p+1)*per_traj):offset+((p+1)*per_traj)+per_traj, :])
                plt.imshow(avg_traj/ float(len(real_and_gen)), vmin=0, vmax=1, cmap="gray")
                plt.ylabel('Avg', size=10)
                plt.yticks([])
                plt.xticks(size=7)


        # plt.tight_layout()
        plt.show()







    print 'Done.'




# print 'Loading data'
# with open(home+'/Documents/MNIST_data/mnist.pkl','rb') as f:
#     mnist_data = pickle.load(f)

# train_x = mnist_data[0][0]
# train_y = mnist_data[0][1]
# valid_x = mnist_data[1][0]
# valid_y = mnist_data[1][1]
# test_x = mnist_data[2][0]
# test_y = mnist_data[2][1]

# train_x = torch.from_numpy(train_x)
# test_x = torch.from_numpy(test_x)
# train_y = torch.from_numpy(train_y)

# print train_x.shape
# print test_x.shape
# print train_y.shape


# model = IWAE()

# if torch.cuda.is_available():
#     print 'GPU available, loading cuda'#, torch.cuda.is_available()
#     model.cuda()
#     train_x = train_x.cuda()


# path_to_load_variables=''
# # path_to_load_variables=home+'/Documents/tmp/pytorch_first.pt'
# path_to_save_variables=home+'/Documents/tmp/pytorch_dkf_first.pt'
# # path_to_save_variables=''



# train(model=model, train_x=train_x, train_y=train_y, valid_x=[], valid_y=[], 
#             path_to_load_variables=path_to_load_variables, 
#             path_to_save_variables=path_to_save_variables, 
#             epochs=10, batch_size=100, display_epoch=2, k=1)



# # print test(model=model, data_x=test_x, path_to_load_variables='', 
# #             batch_size=20, display_epoch=100, k=1000)

# print 'Done.'


















