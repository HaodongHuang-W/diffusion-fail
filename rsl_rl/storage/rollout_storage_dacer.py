import torch
import numpy as np

from rsl_rl.utils import split_and_pad_trajectories

class RolloutStorageDACER:
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.hidden_states = None
        
        def clear(self):
            self.__init__()
    
    def __init__(self, num_envs, num_transitions_per_env, actor_obs_shape, privileged_obs_shape, action_shape, device='cpu'):
        '''
        初始化 RolloutStorageDACER
        : param num_envs: 环境数量
        : param num_transitions_per_env: 每个环境的转换数量
        : param actor_obs_shape: Actor 观测空间的形状
        : param privileged_obs_shape: Critic 观测空间的形状
        : param action_shape: 动作空间的形状
        : param device: 设备
        '''
        self.device = device
        self.actor_obs_shape = actor_obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.action_shape = action_shape

        #核心存储
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *actor_obs_shape, device=self.device)
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device)
        else:
            self.privileged_observations = None
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *action_shape, device=self.device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)  # 添加 returns

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        #RNN相关
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        self.step = 0

    def add_transitions(self, transition: Transition):
        '''
        添加转换到存储中
        : param transition: 包含观测、动作、奖励等信息的Transition对象
        '''
        if self.step >= self.num_transitions_per_env:
            raise AssertionError('Rollout buffer overflow')        
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self._save_hidden_states(transition.hidden_states)
        self.step += 1
    
    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states==(None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)

        # initialize if needed 
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))]
            self.saved_hidden_states_c = [torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        '''
        清空存储
        '''
        self.step = 0

    def compute_returns(self, last_values, gamma):
        ''''
        计算回报
        : param last_values: 最后一个状态的价值
        : param gamma: 折扣因子
        '''
        returns = torch.zeros_like(self.rewards)
        last_values = last_values.unsqueeze(1) # [num_envs, 1]
        returns[-1] = last_values

        for t in reversed(range(self.num_transitions_per_env- 1)):
            returns[t] = self.rewards[t] + gamma * returns[t + 1] * (1 - self.dones[t].float())
        self.returns = returns

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        '''
        生成小批量数据
        : param num_mini_batches: 小批量数量, 4
        : param num_epochs: 迭代次数,5
        : para  self.num_transitions_per_env: 24
        '''
        batch_size = self.num_envs * self.num_transitions_per_env # 4096*24=98304
        mini_batch_size = batch_size // num_mini_batches #98304//4=24576
        indices = torch.randperm(num_mini_batches*mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations
        
        actions = self.actions.flatten(0, 1)
        returns = self.returns.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                critic_obs_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                returns_batch = returns[batch_idx]

                yield (obs_batch, critic_obs_batch, actions_batch, returns_batch)
    
    
    
    # for RNNs only
    def reccurent_mini_batch_generator(self, num_mini_batches, num_epochs=8):

        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        if self.privileged_observations is not None: 
            padded_critic_obs_trajectories, _ = split_and_pad_trajectories(self.privileged_observations, self.dones)
        else: 
            padded_critic_obs_trajectories = padded_obs_trajectories

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i*mini_batch_size
                stop = (i+1)*mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size
                
                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                critic_obs_batch = padded_critic_obs_trajectories[:, first_traj:last_traj]

                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                # then take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [ saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous()
                                for saved_hidden_states in self.saved_hidden_states_a ] 
                hid_c_batch = [ saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj].transpose(1, 0).contiguous()
                                for saved_hidden_states in self.saved_hidden_states_c ]
                # remove the tuple for GRU
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch)==1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch)==1 else hid_a_batch

                yield obs_batch, critic_obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, \
                       old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (hid_a_batch, hid_c_batch), masks_batch
                
                first_traj = last_traj







