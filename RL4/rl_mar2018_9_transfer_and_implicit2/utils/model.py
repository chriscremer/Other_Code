import torch
import torch.nn as nn
import torch.nn.functional as F
# from running_stat import ObsNorm
from distributions import Categorical, DiagGaussian


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)









class CNNPolicy_with_var(nn.Module):
    def __init__(self, num_inputs, action_space):
        super(CNNPolicy_with_var, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)

        self.linear1 = nn.Linear(32 * 7 * 7, 512)

        self.critic_linear1 = nn.Linear(512, 200)
        self.critic_linear_mean = nn.Linear(200, 1)
        self.critic_linear_logvar = nn.Linear(200, 1)

        self.actor_linear1 = nn.Linear(512, 200)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(200, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(200, num_outputs)
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


    def forward(self, inputs):
        x = self.conv1(inputs / 255.0)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = x.view(-1, 32 * 7 * 7)
        x = self.linear1(x) #[B,512]
        x = F.relu(x)

        x_a = self.actor_linear1(x)
        x_a = F.relu(x_a)

        x_v = self.critic_linear1(x)
        x_v = F.relu(x_v)
        value_mean = self.critic_linear_mean(x_v)
        value_logvar = self.critic_linear_logvar(x_v)

        return value_mean, value_logvar, x_a


    def action_dist(self, inputs):
        x = self.conv1(inputs / 255.0)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = x.view(-1, 32 * 7 * 7)
        x = self.linear1(x) #[B,512]
        x = F.relu(x)

        x_a = self.actor_linear1(x)
        x_a = F.relu(x_a)

        return self.dist.action_probs(x_a)



    def act(self, inputs, deterministic=False):
        value_mean, value_logvar, x_a = self.forward(inputs)
        action = self.dist.sample(x_a, deterministic=deterministic)
        return value_mean, value_logvar, action

    def evaluate_actions(self, inputs, actions):
        value_mean, value_logvar, x_a = self.forward(inputs)
        action_log_probs, dist_entropy = self.dist.evaluate_actions(x_a, actions)
        return value_mean, value_logvar, action_log_probs, dist_entropy












































class FFPolicy(nn.Module):
    def __init__(self):
        super(FFPolicy, self).__init__()

    def forward(self, x):
        raise NotImplementedError

    def act(self, inputs, deterministic=False):
        value, x = self(inputs)
        action = self.dist.sample(x, deterministic=deterministic)
        return value, action

    def evaluate_actions(self, inputs, actions):
        value, x = self(inputs)
        action_log_probs, dist_entropy = self.dist.evaluate_actions(x, actions)
        return value, action_log_probs, dist_entropy







class CNNPolicy(FFPolicy):
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

    # def forward(self, inputs):
    #     x = self.conv1(inputs / 255.0)
    #     x = F.relu(x)

    #     x = self.conv2(x)
    #     x = F.relu(x)

    #     x = self.conv3(x)
    #     x = F.relu(x)

    #     x = x.view(-1, 32 * 7 * 7)
    #     x = self.linear1(x)
    #     x = F.relu(x)

    #     return self.critic_linear(x), x



    def forward(self, inputs):
        x = self.conv1(inputs / 255.0)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = x.view(-1, 32 * 7 * 7)
        x = self.linear1(x)
        for_action = x


        x = F.relu(x)
        for_value= self.critic_linear(x)

        return for_value, for_action




class CNNPolicy_dropout(CNNPolicy):

    def forward(self, inputs):
        x = self.conv1(inputs / 255.0)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = x.view(-1, 32 * 7 * 7)
        x = self.linear1(x)
        for_action = x
        
        x = F.dropout(x, p=.5, training=True)  #training false has no stochasticity 
        x = F.relu(x)
        for_value= self.critic_linear(x)

        return for_value, for_action
























def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)







class MLPPolicy(FFPolicy):
    def __init__(self, num_inputs, action_space):
        super(MLPPolicy, self).__init__()

        self.obs_filter = ObsNorm((1, num_inputs), clip=5)
        self.action_space = action_space

        self.a_fc1 = nn.Linear(num_inputs, 64)
        self.a_fc2 = nn.Linear(64, 64)

        self.v_fc1 = nn.Linear(num_inputs, 64)
        self.v_fc2 = nn.Linear(64, 64)
        self.v_fc3 = nn.Linear(64, 1)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(64, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(64, num_outputs)
        else:
            raise NotImplementedError

        self.train()
        self.reset_parameters()

    def reset_parameters(self):
        self.apply(weights_init_mlp)

        """
        tanh_gain = nn.init.calculate_gain('tanh')
        self.a_fc1.weight.data.mul_(tanh_gain)
        self.a_fc2.weight.data.mul_(tanh_gain)
        self.v_fc1.weight.data.mul_(tanh_gain)
        self.v_fc2.weight.data.mul_(tanh_gain)
        """

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def forward(self, inputs):
        inputs.data = self.obs_filter(inputs.data)

        x = self.v_fc1(inputs)
        x = F.tanh(x)

        x = self.v_fc2(x)
        x = F.tanh(x)

        x = self.v_fc3(x)
        value = x

        x = self.a_fc1(inputs)
        x = F.tanh(x)

        x = self.a_fc2(x)
        x = F.tanh(x)

        return value, x







class ObsNorm(nn.Module):
    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        super(ObsNorm, self).__init__()
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.register_buffer('count', torch.zeros(1).double() + 1e-2)
        self.register_buffer('sum', torch.zeros(shape).double())
        self.register_buffer('sum_sqr', torch.zeros(shape).double() + 1e-2)

        self.register_buffer('mean', torch.zeros(shape),)
        self.register_buffer('std', torch.ones(shape))

    def update(self, x):
        self.count += x.size(0)
        self.sum += x.sum(0, keepdim=True).double()
        self.sum_sqr += x.pow(2).sum(0, keepdim=True).double()

        self.mean = self.sum / self.count
        self.std = (self.sum_sqr / self.count - self.mean.pow(2)).clamp(1e-2, 1e9).sqrt()

        self.mean = self.mean.float()
        self.std = self.std.float()

    def __call__(self, x):
        if self.demean:
            x = x - self.mean
        if self.destd:
            x = x / self.std
        if self.clip:
            x = x.clamp(-self.clip, self.clip)
        return x










































class CNNPolicy2(FFPolicy):
    def __init__(self, num_inputs, action_space):
        super(CNNPolicy2, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)

        self.linear1 = nn.Linear(32 * 7 * 7, 512)

        self.critic_linear1 = nn.Linear(512, 200)
        self.critic_linear2 = nn.Linear(200, 1)

        self.actor_linear1 = nn.Linear(512, 200)
        # self.actor_linear2 = nn.Linear(200, 200)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(200, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(200, num_outputs)
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


    def forward(self, inputs):
        x = self.conv1(inputs / 255.0)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = x.view(-1, 32 * 7 * 7)
        x = self.linear1(x) #[B,512]
        x = F.relu(x)

        x_a = self.actor_linear1(x)
        x_a = F.relu(x_a)

        x_v = self.critic_linear1(x)
        x_v = F.relu(x_v)
        x_v = self.critic_linear2(x_v)

        return x_v, x_a


    def action_dist(self, inputs):
        x = self.conv1(inputs / 255.0)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = x.view(-1, 32 * 7 * 7)
        x = self.linear1(x) #[B,512]
        x = F.relu(x)

        x_a = self.actor_linear1(x)
        x_a = F.relu(x_a)

        # x_v = self.critic_linear1(x)
        # x_v = F.relu(x_v)
        # x_v = self.critic_linear2(x_v)

        # print (x_a)


        return self.dist.action_probs(x_a)






class CNNPolicy_dropout2(CNNPolicy2):

    def forward(self, inputs):


        x = self.conv1(inputs / 255.0)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = x.view(-1, 32 * 7 * 7)
        x = self.linear1(x) #[B,512]
        x = F.relu(x)

        x_a = self.actor_linear1(x)
        x_a = F.relu(x_a)

        x_v = self.critic_linear1(F.dropout(x, p=.5, training=True)) 
        x_v = F.relu(x_v)
        x_v = self.critic_linear2(x_v)

        return x_v, x_a















