import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

# 简单的去噪网络示例
class DenoiseNet(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(DenoiseNet, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ELU())
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[l], input_dim))  # 输出维度和输入维度保持一致
            else:
                layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                layers.append(nn.ELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# 简单的 MLP 网络
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ELU())
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[l], output_dim))
            else:
                layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                layers.append(nn.ELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ActorCriticdiff(nn.Module):
    is_recurrent = False
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[512, 256, 128],
                        critic_hidden_dims=[512, 256, 128],
                        activation='elu',
                        init_noise_std=1.0,
                        num_diffusion_steps=10,  # 扩散步骤数
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticdiff, self).__init__()

        self.num_diffusion_steps = num_diffusion_steps

        # 逆扩散模型的去噪网络用于 actor
        self.actor_denoise_net = DenoiseNet(num_actor_obs, actor_hidden_dims)
        # 从观测维度映射到动作维度
        self.actor_output_layer = nn.Linear(num_actor_obs, num_actions)
        # MLP 用于 critic
        self.critic_mlp = MLP(num_critic_obs, critic_hidden_dims, 1)

        print(f"Actor Denoise Net: {self.actor_denoise_net}")
        print(f"Actor Output Layer: {self.actor_output_layer}")
        print(f"Critic MLP: {self.critic_mlp}")

        # Action noise，修改为和观测维度一致
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actor_obs))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)



    def update_distribution(self, observations):
        # 简单的逆扩散过程示例
        noisy_obs = observations + torch.randn_like(observations) * self.std
        for _ in range(self.num_diffusion_steps):
            noisy_obs = self.actor_denoise_net(noisy_obs)
        # 将去噪后的观测映射到动作维度
        mean = self.actor_output_layer(noisy_obs)
        self.distribution = Normal(mean, mean*0. + self.std[:12])  # 取前 num_actions 个标准差

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        # 简单的逆扩散过程示例
        noisy_obs = observations + torch.randn_like(observations) * self.std
        for _ in range(self.num_diffusion_steps):
            noisy_obs = self.actor_denoise_net(noisy_obs)
        # 将去噪后的观测映射到动作维度
        actions_mean = self.actor_output_layer(noisy_obs)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        return self.critic_mlp(critic_observations)