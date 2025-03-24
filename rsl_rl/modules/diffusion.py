import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from typing import Protocol, Tuple 
from dataclasses import dataclass 

class DiffusionModel(Protocol):
    def __call__(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor: #该对象接受两个参数：时间步 t 和输入张量 x，并返回一个张量。这用于类型检查，确保传入的模型实现了这个调用接口
        ...

@dataclass(frozen=True) #使用 dataclass 装饰器创建一个不可变的数据类
class BetaScheduleCoefficients:
    betas: torch.Tensor #张量 betas
    alphas: torch.Tensor #张量 alphas=1-betas
    alphas_cumprod: torch.Tensor #张量 alphas 的累积乘积
    alphas_cumprod_prev: torch.Tensor #张量 alphas_cumprod 的前一个值t-1
    sqrt_alphas_cumprod: torch.Tensor #张量 alphas_cumprod 的平方根
    sqrt_one_minus_alphas_cumprod: torch.Tensor #张量 1-alphas_cumprod 的平方根
    log_one_minus_alphas_cumprod: torch.Tensor #张量 1-alphas_cumprod 的自然对数
    sqrt_recip_alphas_cumprod: torch.Tensor #张量 1/alphas_cumprod 的平方根
    sqrt_recipm1_alphas_cumprod: torch.Tensor #张量 (1/alphas_cumprod)-1 的平方根
    posterior_variance: torch.Tensor #后验方差
    posterior_log_variance_clipped: torch.Tensor #后验对数方差（截断）
    posterior_mean_coef1: torch.Tensor #后验均值系数 1
    posterior_mean_coef2: torch.Tensor #后验均值系数 2

    @staticmethod
    def from_beta(betas: torch.Tensor):        #返回的是利用 betas 计算得到的各种值
        alphas = 1. - betas #计算 alphas=1-betas
        alphas_cumprod = torch.cumprod(alphas, dim=0) #计算 alphas 的累积乘积
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=alphas_cumprod.device), alphas_cumprod[:-1]]) #计算 alphas_cumprod 的前一个值

        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod) #计算 alphas_cumprod 的平方根
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod) #计算 1-alphas_cumprod 的平方根
        log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod) #计算 1-alphas_cumprod 的自然对数
        sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod) #计算 1/alphas_cumprod 的平方根
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1) #计算 1/alphas_cumprod-1 的平方根

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) #计算后验方差
        posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20)) #计算后验对数方差（截断）
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod) #前一步的信号在后验均值中的权重
        posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)#当前步的噪声在后验均值中的权重

        return BetaScheduleCoefficients(
            betas=betas,
            alphas=alphas,
            alphas_cumprod=alphas_cumprod,
            alphas_cumprod_prev=alphas_cumprod_prev,
            sqrt_alphas_cumprod=sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
            log_one_minus_alphas_cumprod=log_one_minus_alphas_cumprod,
            sqrt_recip_alphas_cumprod=sqrt_recip_alphas_cumprod,
            sqrt_recipm1_alphas_cumprod=sqrt_recipm1_alphas_cumprod,
            posterior_variance=posterior_variance,
            posterior_log_variance_clipped=posterior_log_variance_clipped,
            posterior_mean_coef1=posterior_mean_coef1,
            posterior_mean_coef2=posterior_mean_coef2
        )

    @staticmethod
    def vp_beta_schedule(timesteps: int):#根据 VP（Variance Preserving）调度策略生成 betas 张量
        t = torch.arange(1, timesteps + 1, dtype=torch.float32) #创建一个从 1 到 timesteps 的时间步序列，表示扩散过程的每个时间点
        T = timesteps #将总时间步数 timesteps 赋值给变量 T
        b_max = 10.
        b_min = 0.1 #定义了噪声系数的最大值 b_max 和最小值 b_min
        alpha = torch.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2) #计算 alpha
        betas = 1 - alpha #计算 betas
        return betas

    @staticmethod
    def cosine_beta_schedule(timesteps: int):#根据余弦调度策略生成 betas 张量
        s = 0.008
        t = torch.arange(0, timesteps + 1, dtype=torch.float32) / timesteps #创建一个从 0 到 timesteps 的时间步序列，并将其归一化到 [0, 1] 范围内。
        alphas_cumprod = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2 #这行代码计算了每个时间步的 alphas_cumprod 值。
        alphas_cumprod /= alphas_cumprod[0] #将 alphas_cumprod 归一化，使其在时间步 0 时的值为 1。
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1] #根据 alphas_cumprod 计算 betas
        betas = torch.clamp(betas, 0, 0.999) #torch.clamp 将 betas 的值限制在 [0, 0.999] 范围内，确保噪声系数的合理性
        return betas

@dataclass(frozen=True)
class GaussianDiffusion:
    num_timesteps: int

    def beta_schedule(self):
        betas = BetaScheduleCoefficients.vp_beta_schedule(self.num_timesteps) #调用 BetaScheduleCoefficients.vp_beta_schedule 方法，根据时间步数 num_timesteps 生成噪声系数 betas 
        return BetaScheduleCoefficients.from_beta(betas) #返回计算的各种值

    def p_mean_variance(self, t: int, x: torch.Tensor, noise_pred: torch.Tensor): #根据当前时间步 t、输入 x 和预测的噪声 noise_pred，计算去噪过程中的均值和方差。 该方法计算了 p(x_{t-1} | x_t) 的均值和方差，用于生成去噪后的数据，计算去噪过程（p过程）中的均值和方差。
        B = self.beta_schedule() #获得了各个参数
        x_recon = x * B.sqrt_recip_alphas_cumprod[t] - noise_pred * B.sqrt_recipm1_alphas_cumprod[t]#这个和实际的计算公式应该是对不上的，但是可能存在着某一种简化
        x_recon = torch.clamp(x_recon, -1, 1) #将 x_recon 的值限制在 [-1, 1] 范围内
        model_mean = x_recon * B.posterior_mean_coef1[t] + x * B.posterior_mean_coef2[t]    #计算去噪后的数据的均值
        model_log_variance = B.posterior_log_variance_clipped[t] #计算去噪后的数据的对数方差
        return model_mean, model_log_variance #返回去噪后的数据的均值和对数方差
    
    '''
    去噪之后的数据 x0
    '''

    def p_sample(self, model: DiffusionModel, shape: Tuple[int, ...], obs:torch.Tensor, device: torch.device) -> torch.Tensor:#从噪声中逐步去噪，生成最终的样本    

        # 确保 shape 是元组类型

                   
        x = torch.randn(size=shape, device=device)#生成初始样本
        noise = torch.randn(size=(self.num_timesteps, *shape), device=device) #生成噪声[20, 4096, 12]

        for t in reversed(range(self.num_timesteps)):
            noise_t = noise[t] #获取当前时间步的噪声
            t_batch = torch.full((shape[0],1), t, device=device) #创建一个形状为 shape[0] 的张量，每个元素的值都是当前时间步 t [4096, 1]
            noise_pred = model(obs, x, t_batch) #根据当前时间步 t 和输入 x，预测噪声
            model_mean, model_log_variance = self.p_mean_variance(t, x, noise_pred) #计算去噪后的数据的均值和对数方差
            x = model_mean + (t > 0) * torch.exp(0.5 * model_log_variance) * noise_t #根据均值和方差生成去噪后的数据

        return x

    '''
    加噪之后的数据 xt
    '''

    def q_sample(self, t: int, x_start: torch.Tensor, noise: torch.Tensor): #根据时间步 t 和初始数据 x_start，生成带噪声的数据 x_t                              
        B = self.beta_schedule() #获得了各个参数
        return B.sqrt_alphas_cumprod[t] * x_start + B.sqrt_one_minus_alphas_cumprod[t] * noise #根据公式生成带噪声的数据 x_t
    
    
    '''
    预测的噪声和和实际的噪声之间的MSE
    '''

    def p_loss(self, model: DiffusionModel, t: torch.Tensor, x_start: torch.Tensor): #计算模型预测噪声与真实噪声之间的均方误差损失。
        assert t.ndim == 1 and t.shape[0] == x_start.shape[0]
        noise = torch.randn_like(x_start) #生成噪声
        x_noisy = torch.stack([self.q_sample(t[i].item(), x_start[i], noise[i]) for i in range(len(t))]) #生成带噪声的数据
        noise_pred = model(t, x_noisy) #根据时间步 t 和带噪声的数据 x_noisy，预测噪声
        loss = F.mse_loss(noise_pred, noise) #计算预测噪声与真实噪声之间的均方误差损失
        return loss

    '''
    计算加权的均方误差损失，允许对不同样本或时间步赋予不同的权重
    '''
    def weighted_p_loss(self, weights: torch.Tensor, model: DiffusionModel, t: torch.Tensor, x_start: torch.Tensor): 
        if len(weights.shape) == 1:
            weights = weights.view(-1, 1)
        assert t.ndim == 1 and t.shape[0] == x_start.shape[0]
        noise = torch.randn_like(x_start)
        x_noisy = torch.stack([self.q_sample(t[i].item(), x_start[i], noise[i]) for i in range(len(t))])
        noise_pred = model(t, x_noisy) #根据时间步 t 和带噪声的数据 x_noisy，预测噪声
        loss = weights * F.mse_loss(noise_pred, noise, reduction='none') #计算预测噪声与真实噪声之间的均方误差损失，并乘以权重
        return loss.mean()
    
    #tensor.ndim 是一个属性，用于获取张量（Tensor）的维度数量
    #x = torch.tensor([1, 2, 3])
    #print(x.ndim)  # 输出：1
    #x = torch.tensor([[1, 2], [3, 4]])
    #print(x.ndim)  # 输出：2