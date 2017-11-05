



value_loss_coef=0.5
entropy_coef=0.01
num_stack=4
cuda=True
num_processes=20
lr = 7e-4
eps=1e-5
alpha=0.99
save_interval=1e6
gamma=.99 #discount factor for rewards
tau=.95  #gae parameter

model1 = {
            'name': 'a2c',
            'algo': 'a2c',
            'num_processes': num_processes,
            'num_steps': 5,
            'cuda': cuda,
            'num_stack': num_stack,
            'log_interval':20,
            'save_interval':save_interval,
            #Optimizer
            'lr':lr,
            'eps':eps, 
            'alpha':alpha,
            #Objective
            'value_loss_coef':value_loss_coef, 
            'entropy_coef':entropy_coef,
            'gamma':gamma,
            'use_gae':False,
            'tau':tau
}

model2 = {
            'name': 'ppo',
            'algo': 'ppo',
            'num_processes': num_processes,
            'num_steps': 200,
            'cuda': cuda,
            'num_stack': num_stack,
            'log_interval':2,
            'batch_size':100,
            'save_interval':save_interval,
            'ppo_epoch': 4,
            'clip_param': .2,
            #Optimizer
            'lr':lr,
            'eps':eps, 
            #Objective
            'value_loss_coef':value_loss_coef, 
            'entropy_coef':entropy_coef,
            'gamma':gamma,
            'use_gae':False,
            'tau':tau
}









