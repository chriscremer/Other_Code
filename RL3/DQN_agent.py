

#This one will be basic
#Just predicts expected reward given an action and a state
# THen minimizes the difference between the 1-step ahead reward

#No target net for now
#no prioritized buffer for now
#epsiolon greedy



from Replay import Replay
from DeepQNet import Qnet



class DQN_agent(object):


    def __init__(self, config):


        #make net
        self.q_net = Qnet()

        #make buffer
        self.replay_buffer = Replay(memory_size=1000, batch_size=32)


    def act(self, state):
        
        #Get q value for each action
        q_values = []
        for a in possible_actions:
            q_values.append(self.q_net.predict(state,action))

        #return action
        return np.argmax(q_values)


    def learn(self):
        #learn from buffer

        #sample buffer
        data = self.replay_buffer.sample()
        #optimize net
        self.q_net.step(data)


    def add_to_buffer(self, experience):
        #add to buffer   
        self.replay_buffer.feed(experience)

     



