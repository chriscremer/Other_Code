
import torch
import torch.nn as nn
import torch.nn.functional as F
from distributions import Categorical, DiagGaussian


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)





class CNNPolicy(nn.Module):
    def __init__(self, num_inputs, action_space):
        super(CNNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)

        self.linear1 = nn.Linear(32 * 7 * 7, 512)

        self.critic_linear = nn.Linear(512, 1)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(512, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(512, num_outputs)
        else:
            raise NotImplementedError

        self.train()
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(weights_init)

        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)


    def encode(self, inputs):

        x = self.conv1(inputs / 255.0)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = x.view(-1, 32 * 7 * 7)

        x = self.linear1(x)

        return x


    def predict_for_action(self, inputs):

        for_action = F.relu(inputs)

        return for_action

    def predict_for_value(self, inputs):

        x = F.relu(inputs)
        for_value= self.critic_linear(x)

        return for_value

    def forward(self, inputs):

        x = self.encode(inputs)
        for_action = self.predict_for_action(x)
        for_value = self.predict_for_value(x)

        return for_value, for_action


    def action_dist(self, inputs):
        x = self.encode(inputs)
        for_action = self.predict_for_action(x)

        return self.dist.action_probs(for_action)



    def act(self, inputs, deterministic=False):
        value, x_action = self(inputs)
        # action = self.dist.sample(x_action, deterministic=deterministic)
        # action_log_probs, dist_entropy = self.dist.evaluate_actions(x_action, actions)
        action, action_log_probs, dist_entropy = self.dist.sample2(x_action, deterministic=deterministic)

        return value, action, action_log_probs, dist_entropy

    # def evaluate_actions(self, inputs, actions):
    #     value, x = self(inputs)
    #     action_log_probs, dist_entropy = self.dist.evaluate_actions(x, actions)
    #     return value, action_log_probs, dist_entropy










# class CNNPolicy_dropout(CNNPolicy):

#     def forward(self, inputs):

#         x = self.encode(inputs)

#         x = self.linear1(x)
#         for_action = x
        
#         x = F.dropout(x, p=.5, training=True)  #training false has no stochasticity 
#         x = F.relu(x)
#         for_value= self.critic_linear(x)

#         return for_value, for_action










class CNNPolicy_trajectory_action_mask(CNNPolicy):

    def __init__(self, num_inputs, action_space):
        super(CNNPolicy_trajectory_action_mask, self).__init__(num_inputs, action_space)

        # []
        # self.mask = torch.bernoulli(fasdfa)

        # self.mask =None
        self.first = 1






    def predict_for_action(self, inputs):

        if self.first:
            self.mask = torch.bernoulli(inputs)
            self.first =0

        # [P,512]
        print (inputs.size())
        print (self.mask.size())

        
        

        for_action = F.relu(inputs*self.mask)

        # print (for_action.size())
        # fsafd

        return for_action


    # def reset_mask(self, done):

    #     fafdas



    def act(self, inputs, deterministic=True):
        value, x = self.forward(inputs)
        action = self.dist.sample(x, deterministic=deterministic)
        return value, action





















