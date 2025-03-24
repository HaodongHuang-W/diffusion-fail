import torch
import torch.nn as nn
import torch.optim as optim

# from rsl_rl.modules import ActorCritic
from rsl_rl.modules import ActorCriticdiff
from rsl_rl.storage import RolloutStorage

class PPO:
    actor_critic: ActorCriticdiff
    def __init__(self,
                 actor_critic, #策略网络和价值网络的组合实例。
                 num_learning_epochs=1, #训练时的迭代次数
                 num_mini_batches=1, #每次迭代中使用的mini-batch数量
                 clip_param=0.2,  #用于限制策略更新的范围
                 gamma=0.998, #折扣因子
                 lam=0.95, #GAE的参数
                 value_loss_coef=1.0, #价值损失的系数
                 entropy_coef=0.0, #熵损失的系数
                 learning_rate=1e-3, #学习率
                 max_grad_norm=1.0,  #梯度裁剪的最大范数，默认为1.0
                 use_clipped_value_loss=True, #是否使用截断的价值损失，默认为 True
                 schedule="fixed", #学习率调度策略，默认为固定学习率
                 desired_kl=0.01, #期望的 KL 散度，默认为 0.01
                 device='cpu',
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic #策略网络和价值网络的组合实例。
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate) #优化器
        self.transition = RolloutStorage.Transition() #初始化一个RolloutStorage.Transition实例

        # PPO parameters 从上面进来的值
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef

        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):#初始化存储器的方法。
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):#没啥用
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent: #如果是循环神经网络，才有这一步
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach() #获取动作
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach() #获取价值
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach() #获取动作的对数概率
        self.transition.action_mean = self.actor_critic.action_mean.detach() #获取动作的均值
        self.transition.action_sigma = self.actor_critic.action_std.detach() #获取动作的标准差
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs #储存观测
        self.transition.critic_observations = critic_obs #储存critic观测
        return self.transition.actions #返回动作
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone() #将环境返回的奖励复制到self.transition.rewards中
        self.transition.dones = dones #将环境返回的终止标志存储
        # Bootstrapping on time outs
        if 'time_outs' in infos: #如果有time_outs
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)# 对奖励进行引导（bootstrapping），即加上折扣后的状态价值。

        # Record the transition
        self.storage.add_transitions(self.transition) #将self.transition添加到self.storage中
        self.transition.clear() #清空self.transition
        self.actor_critic.reset(dones) #重置actor_critic
    
    def compute_returns(self, last_critic_obs): #计算最后一个状态的价值，并将其从计算图中分离。
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam) #根据最后一个状态的价值、折扣因子和GAE参数计算回报。

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0 #初始化价值函数损失和代理损失的平均值为0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:


                self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0]) #根据观测和隐藏状态，获取动作
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch) #获取动作的对数概率
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]) #获取价值
                mu_batch = self.actor_critic.action_mean #获取动作的均值
                sigma_batch = self.actor_critic.action_std #获取动作的标准差
                entropy_batch = self.actor_critic.entropy #获取熵

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive': #如果期望的KL散度不为空且学习率调度策略为自适应
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)) #ratio=exp(logπ new(a∣s)−logπ old(a∣s))
                surrogate = -torch.squeeze(advantages_batch) * ratio     #surrogate=−A(s,a)⋅ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param) #surrogate_clipped=−A(s,a)⋅clip(ratio,1−ε,1+ε)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean() #surrogate_loss=max(surrogate,surrogate_clipped)

                # Value function loss
                if self.use_clipped_value_loss: #如果使用截断的价值损失
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param) #value_clipped=target_values_batch+clip(value_batch−target_values_batch,−ε,ε)
                    value_losses = (value_batch - returns_batch).pow(2) #value_losses=(V(s)−G(s))^2
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)  #value_losses_clipped=(V_clipped(s)−G(s))^2
                    value_loss = torch.max(value_losses, value_losses_clipped).mean() #value_loss=E[max(value_losses,value_losses_clipped)]
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean() #value_loss=E[(G(s)−V(s))^2]

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() #loss=surrogate_loss+β⋅value_loss−α⋅E[entropy]

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm) #clip_grad_norm(∇θ,max_grad_norm),对梯度进行裁剪，防止梯度爆炸。
                self.optimizer.step()

                mean_value_loss += value_loss.item() #将价值损失的值加到mean_value_loss中
                mean_surrogate_loss += surrogate_loss.item() #将代理损失的值加到mean_surrogate_loss中

        num_updates = self.num_learning_epochs * self.num_mini_batches #计算更新次数
        mean_value_loss /= num_updates #计算价值损失的平均值
        mean_surrogate_loss /= num_updates #计算代理损失的平均值
        self.storage.clear() #清空self.storage

        return mean_value_loss, mean_surrogate_loss #返回价值损失和代理损失的平均值
