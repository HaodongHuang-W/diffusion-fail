import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from .blocks import DistributionalQNet2, DACERPolicyNet #选择这两个作为critic和actor
from .diffusion import GaussianDiffusion

'''
添加了reward centering和entropy regularization
'''


class ActorCriticDACER(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation_actor='mish',
                        activation_critic='gelu',
                        output_activation='identity',
                        num_timesteps=20,
                        init_noise_std=1.0,
                        target_entropy=None,
                        **kwargs):
        super(ActorCriticDACER,self).__init__()

        activation_actor = get_activation(activation_actor)
        activation_critic = get_activation(activation_critic)
        output_activation = get_activation(output_activation)

        actor_input_dims = num_actor_obs
        critic_input_dims = num_critic_obs

        #Actor(DACERPolicyNet)
        self.actor = DACERPolicyNet(
            16 +num_actions, #这里的16是因为DACERPolicyNet中的t是一个额外的输入
            num_actions, 
            actor_hidden_dims, 
            activation_actor, 
            output_activation)
        
        #Critic(Double Q-Networks using DistributionalQNet2)
        self.q1 = DistributionalQNet2(
            critic_input_dims, 
            critic_hidden_dims, 
            activation_critic, 
            output_activation)
        
        self.q2 = DistributionalQNet2(
            critic_input_dims, 
            critic_hidden_dims, 
            activation_critic, 
            output_activation)
        
        # Target networks for Q1 and Q2
        self.q1_target = DistributionalQNet2(
            critic_input_dims, 
            critic_hidden_dims, 
            activation_critic, 
            output_activation)
        
        self.q2_target = DistributionalQNet2(
            critic_input_dims, 
            critic_hidden_dims, 
            activation_critic, 
            output_activation)
        
        #Initation target networks with the same weights as the online networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Diffusion process for Actor
        self.diffusion = GaussianDiffusion(num_timesteps)

        # Entropy Regularization
        self.log_alpha = nn.Parameter(torch.tensor(math.log(3), dtype=torch.float32))
        if target_entropy is not None:
            self.target_entropy = target_entropy   #这里我们需要给一个target_entropy，是-0.9*num_actions
        else:
            self.target_entropy = -num_actions


        #Reward Centering
        self.average_reward = 0.0
        self.step_size = 0.001 #这里应该如何取值,对应的论文中的η*α

        #Print the network
        print("Actor-Critic DACER Network:")
        print(f"Actor Network(DACERPolicyNet): {self.actor}")
        print(f"Critic Network(Double Q-Networks): Q1: {self.q1}, Q2: {self.q2}")
        print(f"Target Networks: Q1_target: {self.q1_target}, Q2_target: {self.q2_target}")
        print(f"Entropy Regularization: {self.log_alpha.item()}")


        #self.std = nn.Parameter(init_noise_std * torch.ones(num_actions)) #这一个不一定需要，因为actor中已经有了

    def forward(self):
        raise NotImplementedError
    
    def reset(self, dones=None):
        pass
    
    def update_average_reward(self, td_errors):
        '''
        更新平均奖励
        '''
        self.average_reward = self.average_reward + self.step_size * td_errors.mean().item()
    
    def act(self, obs):  
        #使用DACERPolicyNet和Diffusion生成动作 
        '''
        betas是根据num_timesteps生成的
        '''
        t = torch.arange(0, self.diffusion.num_timesteps, device=obs.device).unsqueeze(0).expand(obs.shape[0], -1)
        #t shape: torch.Size([4096, 20])
        #obs shape: torch.Size([4096, 45])
        #print((obs.shape[0],12))
        actions = self.diffusion.p_sample(self.actor, (obs.shape[0],12), obs, device=obs.device)
        #添加噪声
        noise = torch.randn_like(actions) * torch.exp(self.log_alpha) * 0.1 #0.15
        #print(f"Act: log_alpha = {self.log_alpha.item()}")
        # actions = actions + noise
        return torch.clamp(actions, -1, 1).detach()

    
    def evaluate(self, critic_obs):
        '''
        使用双Q网络评估Q值
        '''
        q_input = critic_obs
        q1_values, _ = self.q1(q_input)
        q2_values, _ = self.q2(q_input)
        return q1_values, q2_values
    
    def target_evaluate(self, critic_obs):
        '''
        使用目标网络评估Q值
        '''
        q_input = critic_obs
        target_q1_values, _ = self.q1_target(q_input)
        target_q2_values, _ = self.q2_target(q_input)
        return target_q1_values, target_q2_values
    
    def update_target_networks(self, tau :float = 0.005):
        '''
        更新目标网络
        '''
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def update_log_alpha(self, entropy):
        '''
        根据估计的熵更新log_alpha
        '''
        #定义log_alpha的损失函数
        def log_alpha_loss(log_alpha):
            alpha_loss = -torch.mean(log_alpha * (self.target_entropy - entropy))
            return alpha_loss
        self.log_alpha_optimizer.zero_grad()
        loss = log_alpha_loss(self.log_alpha)
        loss.backward()
        self.log_alpha_optimizer.step()

    def act_inference(self, observations):
        """
        用于推理模式的动作生成，不添加噪声。
        """
        # 假设推理时 t=0
        t = torch.zeros((observations.shape[0],1), device=observations.device, dtype=torch.long) #生成一个全为0的时间步长
        x = torch.randn(size=(observations.shape[0], 12), device=observations.device) #生成初始噪声
        actions = self.actor(observations, x, t)
        return torch.clamp(actions, -1, 1)


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    elif act_name == "gelu":
        return nn.GELU()
    elif act_name == "mish":
        return nn.Mish()
    elif act_name == "identity":
        return nn.Identity()
    else:
        print("invalid activation function!")
        return None
    








