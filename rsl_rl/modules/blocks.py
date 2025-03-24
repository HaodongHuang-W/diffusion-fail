import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from typing import Callable, Optional, Sequence, Tuple, Union #用于类型提示



'''
mlp函数调用的结构[input, output, hidden, activation, output_activation, squeeze_output]
'''

class SqueezeLayer(nn.Module): #去掉指定维度
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(self.dim) 
    
    
def mlp(input_dims:int, output_dims: int, hidden_dims: Sequence[int], activation, output_activation, squeeze_output: bool = False) -> nn.Sequential:
    layers = []
    layers.append(nn.Linear(input_dims, hidden_dims[0])) #输入
    layers.append(activation)
    for l in range(len(hidden_dims)-1):
        layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
        layers.append(activation)
    if output_dims:
        layers.append(nn.Linear(hidden_dims[-1], output_dims))#输出
        layers.append(output_activation)
    if squeeze_output:
        layers.append(SqueezeLayer(-1))#去掉最后一个维度
    return nn.Sequential(*layers)
    

def scaled_sinusoidal_encoding(t: torch.Tensor, *, dim: int, theta: int = 10000, batch_shape=None) -> torch.Tensor: #用于生成缩放的正弦位置编码,dim是16
    assert dim % 2 == 0 #确保 dim 是偶数，因为后续的编码过程会将编码向量分成两部分（正弦和余弦），所以 dim 必须是偶数
    if batch_shape is not None:
        assert t.shape[:-1] == batch_shape #如果 batch_shape 不为 None，则检查输入张量 t 的除最后一维外的形状是否与 batch_shape 一致。这里都是4096
    
    scale = 1 / dim ** 0.5 #计算缩放因子
    half_dim = dim // 2 #计算 dim 的一半，因为后续会分别计算正弦和余弦编码
    freq_seq = torch.arange(half_dim, device=t.device) / half_dim #生成一个从 0 到 half_dim - 1 的序列，并除以 half_dim，该序列用于计算不同频率的位置编码。
    inv_freq = theta ** -freq_seq #计算频率的倒数

    emb = torch.einsum('..., j -> ... j', t, inv_freq) #计算中间编码张量 
    emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1) #将正弦和余弦编码拼接在一起
    emb *= scale #乘以缩放因子，到此处维度为[4096, 20, 16]
    
    
    '''
    这里是不可运行的,因为3维不能扩展成2维
    '''
    # if batch_shape is not None:
    #     emb = emb.expand(*batch_shape, dim) 
    '''
    此处是我自行修改的，为了保持两个维度，将[4096,20,16]转为[4096,20*16]
    '''
    if batch_shape is not None:
        emb = emb.view(*batch_shape, -1)

    return emb #为输入的位置索引张量t生成了维度为dim的缩放正弦位置编码张量 。

class ValueNet(nn.Module): #定义 ValueNet 类
    def __init__(self, input_dims: int, hidden_dims: Sequence[int], activation, output_activation):
        super().__init__() 
        self.mlp = mlp(input_dims, 1, hidden_dims, activation, output_activation, squeeze_output=True) #输出维度为 1，squeeze_output=True：在输出时去掉最后一个维度（从 [batch_size, 1] 转换为 [batch_size]）

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs) 


class QNet(nn.Module): #定义 QNet 类，用于实现一个 Q 网络。
    def __init__(self, input_dims:int, hidden_dims: Sequence[int], activation, output_activation):
        super().__init__()
        self.mlp = mlp(input_dims, 1, hidden_dims, activation, output_activation, squeeze_output=True) #同上

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs) 



class DistributionalQNet(nn.Module): #用于实现一个分布式 Q 网络，两个MLP，输出是均值和标准差
    def __init__(self, input_dims: int, hidden_dims: Sequence[int], activation, output_activation,
                 min_log_std: float = -0.1, max_log_std: float = 4.0): #初始化方法，新增了 min_log_std 和 max_log_std 参数，用于限制对数标准差的范围
        super().__init__()
        self.mlp_mean = mlp(input_dims, 1, hidden_dims, activation, output_activation, squeeze_output=True) #两个MLP，这个是用于输出均值
        self.mlp_log_std = mlp(input_dims, 1, hidden_dims, activation, output_activation, squeeze_output=True) #这个是用于输出对数标准差
        self.min_log_std = min_log_std 
        self.max_log_std = max_log_std 

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        value_mean = self.mlp_mean(obs) #计算均值
        value_log_std = self.mlp_log_std(obs) #计算对数标准差
        denominator = max(abs(self.min_log_std), abs(self.max_log_std)) #绝对值的最大值。
        value_log_std = (
            torch.maximum(self.max_log_std * torch.tanh(value_log_std / denominator), torch.tensor(0.0)) +
            torch.minimum(-self.min_log_std * torch.tanh(value_log_std / denominator), torch.tensor(0.0)) #上一行最大就是 max_log_std，这一行最小就是 min_log_std，就是限制在这个范围之内
        )
        return value_mean, value_log_std
    

class DistributionalQNet2(nn.Module): #就是一个MLP，输出是均值和标准差
    def __init__(self, input_dims: int, hidden_dims: Sequence[int], activation, output_activation):
        super().__init__()
        self.mlp = mlp(input_dims, 2, hidden_dims, activation, output_activation) #输出维度为 2，输出的两个值分别用于表示 Q 值的均值和标准差的对数

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.mlp(obs) 
        value_mean = output[..., 0] #第一个值表示均值，[..., 0] 表示取最后一个维度的第一个值
        value_std = F.softplus(output[..., 1]) #第二个值表示标准差的对数，使用 softplus 函数将其转换为标准差
        return value_mean, value_std


class PolicyNet(nn.Module): #定义 PolicyNet 类，用于实现一个策略网络。用于输出动作的均值和标准差。
    def __init__(self, input_dims: int, act_dims: int, hidden_dims: Sequence[int], activation, output_activation,
                 min_log_std: float = -20.0, max_log_std: float = 0.5, log_std_mode: Union[str, float] = 'shared'): #log_std_mode：定义标准差的计算模式，可以是 'shared'（均值和标准差共享网络）、'separate'（均值和标准差分别由独立网络计算）或固定值。
        super().__init__()
        self.act_dims = act_dims #动作空间的维度
        self.min_log_std = min_log_std 
        self.max_log_std = max_log_std 
        self.log_std_mode = log_std_mode 

        if log_std_mode == 'shared': #如果标准差的计算模式是 'shared'
            self.mlp = mlp(input_dims, act_dims * 2, hidden_dims, activation, output_activation) #输出维度为 2*act_dims，前 act_dims 个值表示均值，后 act_dims 个值表示对数标准差
        elif log_std_mode == 'separate': #如果标准差的计算模式是 'separate'
            self.mlp_mean = mlp(input_dims, act_dims, hidden_dims, activation, output_activation) #输出维度为 act_dims，表示均值
            self.mlp_log_std = mlp(input_dims, act_dims, hidden_dims, activation, output_activation) #输出维度为 act_dims，表示对数标准差
        else:
            self.mlp_mean = mlp(input_dims, act_dims, hidden_dims, activation, output_activation) #输出维度为 act_dims，表示均值
            self.log_std = nn.Parameter(torch.full((act_dims,), float(log_std_mode))) #是一个表示对数标准差初始值的变量，将其转换为浮点数后填充到张量中。

    def forward(self, obs: torch.Tensor, *, return_log_std: bool = False) -> torch.Tensor:
        if self.log_std_mode == 'shared':
            output = self.mlp(obs) #调用 mlp 函数
            mean, log_std = torch.split(output, self.act_dims, dim=-1) #将输出拆分为均值和对数标准差
        elif self.log_std_mode == 'separate': #如果标准差的计算模式是 'separate'
            mean = self.mlp_mean(obs) #计算均值
            log_std = self.mlp_log_std(obs) #计算对数标准差
        else:
            mean = self.mlp_mean(obs) #计算均值
            log_std = self.log_std.expand_as(mean) #对数标准差的维度与均值相同

        if self.min_log_std is not None or self.max_log_std is not None: #如果对数标准差的范围有限制
            log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std) #将对数标准差限制在指定范围内

        if return_log_std: #如果需要返回对数标准差
            return mean, log_std #返回均值和对数标准差
        else: #否则
            return mean, torch.exp(log_std) #返回均值和标准差


class DeterministicPolicyNet(nn.Module): #定义 DeterministicPolicyNet 类，用于实现一个确定性策略网络。
    def __init__(self, input_dims: int, act_dims: int, hidden_dims: Sequence[int], activation, output_activation):
        super().__init__()
        self.mlp = mlp(input_dims, act_dims, hidden_dims, activation, output_activation) #输出维度为 act_dims，表示动作

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs) 


class ModelNet(nn.Module): #定义 ModelNet 类，用于实现一个动力学模型网络。mlp输出是obs
    def __init__(self, input_dims: int, obs_dims: int, hidden_dims: Sequence[int], activation, output_activation):
        super().__init__()
        self.mlp = mlp(input_dims, obs_dims, hidden_dims, activation, output_activation) 

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs) 


class QScoreNet(nn.Module): #定义 QScoreNet 类，用于实现一个 Q 分数网络。
    def __init__(self,input_dims: int, act_dims:int, hidden_dims: Sequence[int], activation, output_activation):
        super().__init__()
        self.mlp = mlp(input_dims, act_dims, hidden_dims, activation, output_activation) 

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs) 


class DiffusionPolicyNet(nn.Module):
    def __init__(self, input_dims:int, act_dims:int, time_dim: int, hidden_dims: Sequence[int], activation, output_activation):
        super().__init__()
        self.time_dim = time_dim #时间维度
        self.mlp = mlp(input_dims, act_dims, hidden_dims, activation, output_activation) 

    def forward(self, obs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = scaled_sinusoidal_encoding(t, dim=self.time_dim, batch_shape=obs.shape[:-1]) #对时间进行编码
        input = torch.cat((obs, te), dim=-1)
        return self.mlp(input) 


class DACERPolicyNet(nn.Module):
    def __init__(self, input_dims:int, act_dims:int, hidden_dims: Sequence[int], activation, output_activation, time_dim: int = 16):
        super().__init__()
        self.time_dim = time_dim #时间维度
        self.mlp = mlp(input_dims, act_dims, hidden_dims, activation, output_activation) 
        self.time_encoder = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            activation,
            nn.Linear(time_dim * 2, time_dim)
        ) 
    def forward(self, obs: torch.Tensor,act:torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = scaled_sinusoidal_encoding(t, dim=self.time_dim, batch_shape=obs.shape[:-1]) #对时间进行编码 [4096,16]
        te = self.time_encoder(te) #再次编码
        input = torch.cat((act, te), dim=-1) #[4096,12+16]，这里我先删掉obs
        return self.mlp(input) 