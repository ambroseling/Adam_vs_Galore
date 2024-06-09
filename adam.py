from torch.optim import Optimizer
import math
import torch

class Adam(Optimizer):
    def  __init__(self,params,lr = 1e-6,betas = (0.99,0.9),eps=1e-6,weight_decay=0.0):
        defaults = dict(lr=lr,betas=betas,weight_decay=weight_decay,eps=eps)
        super().__init__(params,defaults=defaults)
    def step(self,closure=None):
        #Iterature through all the parameter 
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                step_size = group['lr']
                # state is a dictionary that holds all the optimizer configurations for each parameter
                if len(state) ==0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sqr'] = torch.zeros_like(p.data)
                exp_avg = state['exp_avg'] #first moment estimate
                exp_avg_sqr = state['exp_avg_sqr'] # second moment estimate
                beta1,beta2 = group['betas'] # get betas
                exp_avg.mul_(beta1).add_((1-beta1),grad) #update biased first moment estimate
                exp_avg_sqr.mul(beta2.addcmul_((1-beta2)),grad,grad) # update biased second moment estimate
                denom = exp_avg_sqr.sqrt().add_(group['eps'])

                # If there is bias correction
                if group['bias_correction'] == True:
                    bias_corrected_first_moment = 1 - beta1 ** state['step']
                    bias_corrected_second_moment = 1 - beta2 ** state['step']
                    step_size = step_size * math.sqrt(bias_corrected_second_moment) / bias_corrected_first_moment

                # Weight update
                p.data.addcdiv_(-step_size,exp_avg,denom)