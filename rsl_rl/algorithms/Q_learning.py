import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.mixture import GaussianMixture

from rsl_rl.modules import ActorCriticDACER
from rsl_rl.storage import RolloutStorageDACER

'''
这里添加了reward centering和entropy regularization
'''

class QLearningDACER:
    '''
    基于DACER的Q-learning算法实现
    '''
    actor_critic: ActorCriticDACER
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=4,
                 num_mini_batches=32,
                 gamma=0.99,
                 tau=0.005,
                 learning_rate=1e-4,
                 alpha_lr=3e-2,
                 max_grad_norm=1.0,
                 delay_alpha_update=2,   #10000，之前是10000
                 delay_policy_update=1,  #2，之前是2，因为这里有并行训练
                 num_samples=200, #估计策略网络的熵时采样的动作数量
                 device='cpu',
                 ):
        
        '''
        初始化QLearningDACER算法
        '''

        self.device = device
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)

        '''
        初始化优化器
        '''
        self.q_optimizer = optim.Adam(self.actor_critic.q1.parameters(), lr=learning_rate)
        self.q_optimizer.add_param_group({'params': self.actor_critic.q2.parameters()})
        self.policy_optimizer = optim.Adam(self.actor_critic.actor.parameters(), lr=learning_rate)
        self.alpha_optimizer = optim.Adam([self.actor_critic.log_alpha], lr=alpha_lr) #优化器用于更新alpha参数

        #Q-learning DACER 参数
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.gamma = gamma
        self.tau = tau
        self.max_grad_norm = max_grad_norm
        self.delay_alpha_update = delay_alpha_update
        self.delay_policy_update = delay_policy_update
        self.num_samples = num_samples
        self.learning_rate = learning_rate

        #初始化存储
        self.storage = None

        #初始化Transition
        self.transition = RolloutStorageDACER.Transition()

        #初始化训练状态
        self.step = 0
        self.mean_q1_std = -1.0
        self.mean_q2_std = -1.0
        self.entropy = 0.0


    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        '''
        初始化存储器
        '''
        self.storage = RolloutStorageDACER(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # 计算 actions
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.transition.actions = self.actor_critic.act(obs).detach()
        #计算并存储当前的Q值
        q1_values, q2_values = self.actor_critic.evaluate(critic_obs)
        self.transition.values = torch.min(q1_values, q2_values).detach()
        self.transition.values = self.transition.values.unsqueeze(1)  #增加维度,从[4096]变为[4096,1]
        self.transition.actions_log_prob = self.actor_critic.log_alpha.detach()
        return self.transition.actions
     

    def process_env_step(self, rewards, dones, infos):
        '''
        处理环境步骤
        '''
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        
        # 处理时间结束（time-out）
        if 'time_outs' in infos:
            #self.transition.values  [4096,1]
            #infos['time_outs'].shape [4096]
            self.transition.rewards = self.transition.rewards + self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)


        # 记录transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        '''
        计算回报
        ''' 
        #计算最后一个时刻的Q值
        last_q1, last_q2 = self.actor_critic.evaluate(last_critic_obs)
        last_values = torch.min(last_q1, last_q2).detach()
        self.storage.compute_returns(last_values, self.gamma)

    def update(self):
        '''
        更新网络参数
        '''
        mean_q_loss = 0
        mean_policy_loss = 0
        mean_alpha_loss = 0


        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for batch in generator:
                obs_batch, critic_obs_batch, actions_batch, return_batch = batch                
                # torch.autograd.set_detect_anomaly(True)
                #计算当前Q值

                current_q1, current_q2 = self.actor_critic.evaluate(critic_obs_batch)
                current_q = torch.min(current_q1, current_q2)


                #计算目标Q值
                with torch.no_grad():
                    # print("obs_batch 形状:", obs_batch.shape)
                    next_actions = self.actor_critic.act(obs_batch)
                    next_q1, next_q2 = self.actor_critic.evaluate(critic_obs_batch)
                    next_q = torch.min(next_q1, next_q2)
                    return_batch = return_batch.squeeze(1)
                    target_q = return_batch + self.gamma * next_q
                
                #计算TD误差
                td_errors = return_batch - self.actor_critic.average_reward + self.gamma * next_q - current_q
                self.actor_critic.update_average_reward(td_errors)
                
                #计算Q值损失
                # q1_loss = nn.functional.mse_loss(current_q1, target_q)
                # q2_loss = nn.functional.mse_loss(current_q2, target_q)
                # q_loss = q1_loss + q2_loss

                
                #利用奖励中心化的TD误差计算Q值损失
                q1_delta_loss = (td_errors**2).mean()
                q2_delta_loss = (td_errors**2).mean()
                q_loss = q1_delta_loss
                

                #反向传播和优化Q网络
                self.q_optimizer.zero_grad()
                q_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.q1.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.actor_critic.q2.parameters(), self.max_grad_norm)
                self.q_optimizer.step()
                

                #计算策略损失和反向传播和优化策略网络
                if self.step % self.delay_policy_update == 0:
                    policy_loss = -torch.min(self.actor_critic.evaluate(critic_obs_batch)[0]).mean()
                    # print(f"policy_loss: {policy_loss}")
                    self.policy_optimizer.zero_grad()
                    policy_loss.backward()
                    nn.utils.clip_grad_norm_(self.actor_critic.actor.parameters(), self.max_grad_norm)
                    self.policy_optimizer.step()


                #计算alpha损失
                if self.step % self.delay_alpha_update == 0:
                    entropy = self.estimate_entropy(obs_batch)
                    entropy = torch.tensor(entropy, device=self.device)
                    alpha_loss = -self.actor_critic.log_alpha * (entropy - self.actor_critic.target_entropy).detach()
                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                else:
                    alpha_loss = torch.tensor(0.0, device=self.device)


                #软更新目标网络
                if self.step % self.delay_policy_update == 0:
                    self.soft_update_target_networks()
                
                
                #更新统计量
                # mean_q_loss += q_loss.item()这是之前的
                mean_q_loss += q_loss.item()
                mean_policy_loss += policy_loss.item()
                mean_alpha_loss += alpha_loss.item() if self.step % self.delay_alpha_update == 0 else 0

                self.step += 1
        
        mean_q_loss /= (self.num_learning_epochs * self.num_mini_batches)
        mean_policy_loss /= (self.num_learning_epochs * self.num_mini_batches)
        mean_alpha_loss /= (self.num_learning_epochs * self.num_mini_batches)

        self.storage.clear()

        return mean_q_loss, mean_policy_loss, mean_alpha_loss
    
    def soft_update_target_networks(self):
        '''
        软更新目标网络
        '''
        self.actor_critic.update_target_networks(self.tau)

    def estimate_entropy(self, obs): #熵在信息论中是一个重要的概念，用来衡量随机变量的不确定性。在强化学习中，策略网络的熵可以用来评估策略的多样性
        '''
        使用高斯混合模型估计策略网络的熵
        '''
        actions = [] #初始化一个空列表 actions，用于存储生成的动作。
        for _ in range(self.num_samples): #循环采样200次
            actions.append(self.actor_critic.act(obs).detach().cpu().numpy()) #添加到actions列表中
            actions = np.stack(actions, axis=1) #将actions列表转换为numpy数组
            entropy = self._estimate_gmm_entropy(actions) #使用高斯混合模型估计动作的熵
            return entropy

    @staticmethod   
    def _estimate_gmm_entropy(actions, n_components=3):
        '''
        使用高斯混合模型估计熵
        '''
        #将动作重塑为二维数组，以适应GMM的输入要求
        actions = actions.reshape(-1, actions.shape[-1])

        #初始化高斯混合模型
        gmm = GaussianMixture(n_components=n_components, covariance_type='full')
        gmm.fit(actions)

        #获取GMM的权重和协方差矩阵
        weights = gmm.weights_
        covariances = gmm.covariances_

        #计算每个高斯分布的熵
        entropies = []
        for i in range(n_components):
            conv_matrix = covariances[i]
            d = covariances.shape[0]
            entropy = 0.5 * d * (1 + np.log(2 * np.pi)) + 0.5 * np.linalg.slogdet(conv_matrix)[1]
            entropies.append(entropy)
        
        #计算混合分布的总熵
        total_entropy = -np.sum(weights * np.array(entropies)) + np.sum(weights * np.array(entropies))
        return total_entropy
