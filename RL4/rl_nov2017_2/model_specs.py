



value_loss_coef=0.5
entropy_coef=0.01
num_stack=4
# num_processes=20
lr = 7e-4
eps=1e-5
alpha=0.99
gamma=.99 #discount factor for rewards
tau=.95  #gae parameter



a2c_rms = {
            'name': 'a2c_rms',
            'algo': 'a2c',
            # 'agent': a2c,
            # 'num_processes': num_processes,
            'num_steps': 5,
            'num_stack': num_stack,
            'log_interval':20,
            #Optimizer
            'opt': 'rms',
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


a2c_adam = {
            'name': 'a2c',
            'algo': 'a2c',
            'dropout': False,
            'num_steps': 4,
            'num_stack': num_stack,
            'log_interval':10,
            #Optimizer
            'opt': 'adam',
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


a2c_dropout = {
            'name': 'a2c_dropout',
            'algo': 'a2c',
            'dropout': True,
            'num_steps': 4,
            'num_stack': num_stack,
            'log_interval':10,
            #Optimizer
            'opt': 'adam',
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


ppo_v1 = {
            'name': 'ppo',
            'algo': 'ppo',
            'num_stack': num_stack,
            'log_interval':2,
            # 'agent': ppo,
            # 'num_processes': num_processes,
            'num_steps': 200,
            'batch_size':50,
            'ppo_epoch': 4, #number of grad steps 
            'clip_param': .2,
            #Optimizer
            'lr':1e-3,
            'eps':eps, 
            #Objective
            'value_loss_coef':value_loss_coef, 
            'entropy_coef':entropy_coef,
            'gamma':gamma,
            'use_gae':False,
            'tau':tau
}



ppo_v2 = {
            'name': 'ppo',
            'algo': 'ppo',
            # 'agent': ppo,
            'num_stack': num_stack,
            'log_interval':20,
            # 'num_processes': num_processes,
            'num_steps': 4,
            'batch_size':4,
            'ppo_epoch': 1,  #4,  #number of grad steps 
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


a2c_long = {
            'name': 'a2c',
            'algo': 'a2c_minibatch',
            # 'agent': a2c_minibatch,
            'num_steps': 200,
            'batch_size': 100,
            'a2c_epochs':20,
            'num_stack': num_stack,
            'log_interval':2,
            #Optimizer
            'opt': 'adam',
            'lr':1e-4,
            'eps':eps, 
            'alpha':alpha,
            #Objective
            'value_loss_coef':value_loss_coef, 
            'entropy_coef':entropy_coef,
            'gamma':gamma,
            'use_gae':False,
            'tau':tau
}




ppo_linear = {
            'name': 'ppo',
            'algo': 'ppo',
            'num_stack': num_stack,
            'log_interval':2,
            # 'agent': ppo,
            # 'num_processes': num_processes,
            'num_steps': 400,
            'batch_size':200,
            'ppo_epoch': 20, #number of grad steps 
            'clip_param': .2,
            #Optimizer
            'lr':1e-2,
            'eps':eps, 
            'lr_schedule':'linear',
            'final_lr':1e-5,
            #Objective
            'value_loss_coef':value_loss_coef, 
            'entropy_coef':entropy_coef,
            'gamma':gamma,
            'use_gae':False,
            'tau':tau
}





