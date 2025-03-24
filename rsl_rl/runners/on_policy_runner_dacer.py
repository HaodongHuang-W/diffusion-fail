import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import QLearningDACER
from rsl_rl.modules import ActorCriticDACER
from rsl_rl.env import VecEnv


class OnPolicyRunnerDACER:
    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):
        
        self.cfg = train_cfg["runner"]
        self.algo_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"])
        actor_critic: ActorCriticDACER = actor_critic_class(self.env.num_obs,
                                                            num_critic_obs,
                                                            self.env.num_actions,
                                                            **self.policy_cfg
                                                            ).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"]) #QLearningDACER
        self.alg : QLearningDACER = alg_class(actor_critic, device=self.device, **self.algo_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        #初始化算法的存储结构，用于存储采样数据
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions]) #初始化存储器

        #日志
        self.log_dir = log_dir
        self.writer = None #TensorBoard 日志记录器，初始值为 None
        self.tot_timesteps = 0 #总采样步数，初始值为 0。
        self.tot_time = 0 #总时间，初始值为 0。
        self.current_learning_iteration = 0 #当前学习迭代次数，初始值为 0。

        _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length)) 
        obs = self.env.get_observations() #获取观测
        privileged_obs = self.env.get_privileged_observations()#获取特权观测量
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device) #当前 episode 长度

        tot_iter = self.current_learning_iteration + num_learning_iterations #总迭代次数
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time() #记录开始时间
            # Rollout
            with torch.inference_mode(): #进入推理模式（inference_mode），禁用梯度计算，提高采样效率。
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions) #执行动作，获取环境返回的观测、特权观测、奖励、终止标志和信息。step函数在leggged_gym/envs/legged_env.py中定义
                    critic_obs = privileged_obs if privileged_obs is not None else obs #评论者观测
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device) #将观测、评论者观测、奖励和终止标志转移到指定的设备上
                    # print(f"迭代 {it}, 步 {i}: {dones.sum().item()}/{dones.shape[0]} 环境已终止")
                    # if dones.all():
                    #     print(f"警告：迭代 {it}, 步 {i} 所有环境都已终止，可能导致数据不足")
                    self.alg.process_env_step(rewards, dones, infos) #处理环境返回的奖励、终止标志和信息

                    if self.log_dir is not None: #如果指定了日志目录
                        # Book keeping
                        if 'episode' in infos: #如果信息中包含 episode
                            ep_infos.append(infos['episode']) #将 episode 信息添加到 ep_infos 中
                        cur_reward_sum += rewards #累加奖励
                        cur_episode_length += 1 #累加 episode 长度
                        new_ids = (dones > 0).nonzero(as_tuple=False) #检测哪些环境的 episode 已经结束。
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()) #将已结束的环境的奖励添加到 rewbuffer 中
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()) #将已结束的环境的 episode 长度添加到 lenbuffer 中
                        cur_reward_sum[new_ids] = 0 #重置已结束的环境的奖励总和
                        cur_episode_length[new_ids] = 0 #重置已结束的环境的 episode 长度

                '''
                结束循环
                '''

                stop = time.time() #记录结束时间
                collection_time = stop - start #记录采样结束时间，并计算采样阶段的耗时

                # Learning step
                start = stop #记录开始时间
                self.alg.compute_returns(critic_obs) #计算回报
            
            mean_q_loss, mean_policy_loss, mean_alpha_loss = self.alg.update() #更新策略网络和价值网络，并获取价值函数损失和代理损失
            stop = time.time() #记录结束时间
            learn_time = stop - start #记录学习结束时间，并计算学习阶段的耗时

            if self.log_dir is not None:
                self.log(locals())# 如果指定了日志路径，则调用 log 方法记录训练信息。

            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it))) #如果当前迭代次数是保存间隔的整数倍，则保存模型。
            ep_infos.clear() #清空 episode 信息

        self.current_learning_iteration += num_learning_iterations #更新当前学习迭代次数
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration))) #保存模型

    def log(self, locs, width=80, pad=35): #记录训练过程中的信息，并输出到控制台和 TensorBoard，控制台输出的宽度，默认为 80，控制台输出的对齐宽度，默认为 35
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs #累加采样步数
        self.tot_time += locs['collection_time'] + locs['learn_time'] #累加时间
        iteration_time = locs['collection_time'] + locs['learn_time'] #计算迭代时间

        ep_string = f'' #初始化一个空字符串，用于存储 episode 相关信息的输出
        if locs['ep_infos']: #如果有 episode 信息
            for key in locs['ep_infos'][0]: #获取第一个 episode 信息字典的所有键（key）。
                infotensor = torch.tensor([], device=self.device) #初始化一个空张量，用于存储所有 episode 信息的值
                for ep_info in locs['ep_infos']: #遍历 locs['ep_infos'] 中的每个 episode 的统计信息字典。
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor): #检查当前 episode 的统计信息值是否为 PyTorch 张量，如果不是，则转换为张量。
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0: #检查当前 episode 的统计信息值是否为零维张量，如果是，则转换为一维张量。
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device))) #将当前 episode 的统计信息值添加到 infotensor 中
                value = torch.mean(infotensor) #计算所有 episode 的统计信息值的平均值
                self.writer.add_scalar('Episode/' + key, value, locs['it']) #将平均值记录到 TensorBoard
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n""" #将平均值格式化为字符串并添加到 ep_string 中
        # mean_std = self.alg.actor_critic.std.mean() #计算动作噪声的均值
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time'])) #计算每秒的采样步数
        '''
        将训练指标记录到 TensorBoard
        '''
        self.writer.add_scalar('Loss/q_loss', locs['mean_q_loss'], locs['it'])
        self.writer.add_scalar('Loss/policy_loss', locs['mean_policy_loss'], locs['it'])
        self.writer.add_scalar('Loss/alpha_loss', locs['mean_alpha_loss'], locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0: #如果有奖励数据，就把下面的数据记录到 TensorBoard
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "    #显示格式为：Learning iteration 当前迭代次数/总迭代次数。 就是训练中的500/1500

        if len(locs['rewbuffer']) > 0: #如果有奖励数据 ，log_string 是一个多行字符串，用于格式化输出。
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Q loss:':>{pad}} {locs['mean_q_loss']:.4f}\n"""
                          f"""{'Policy loss:':>{pad}} {locs['mean_policy_loss']:.4f}\n"""
                          f"""{'Alpha loss:':>{pad}} {locs['mean_alpha_loss']:.4f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else: #如果没有奖励数据
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Q loss:':>{pad}} {locs['mean_q_loss']:.4f}\n"""
                          f"""{'Policy loss:':>{pad}} {locs['mean_policy_loss']:.4f}\n"""
                          f"""{'Alpha loss:':>{pad}} {locs['mean_alpha_loss']:.4f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string #将 episode 信息添加到 log_string 中
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""") #将总采样步数、迭代时间、总时间和预计剩余时间添加到 log_string 中
        print(log_string) #输出 log_string 到控制台

    def save(self, path, infos=None): #保存模型
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.q_optimizer.state_dict(),
            'policy_optimizer_state_dict': self.alg.policy_optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.alg.alpha_optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True): #加载模型
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.q_optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            self.alg.policy_optimizer.load_state_dict(loaded_dict['policy_optimizer_state_dict'])
            self.alg.alpha_optimizer.load_state_dict(loaded_dict['alpha_optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None): #获取推理策略
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference









            
