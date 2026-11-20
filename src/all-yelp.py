import json
from datetime import datetime
import time
import random
import numpy as np
import torch,dgl
import os
from tqdm import tqdm
import logging
import sys
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader
import  pickle
import scipy.sparse as sp
import torch.utils.data as data
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
import math
import torch.nn.init as init
from torch.optim.lr_scheduler import CosineAnnealingLR
import itertools
flag= 1

def fix_random_seed_as(seed):
    # 分别对应python内置、PyTorch的CPU和GPU、numpy的随机种子
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_model(model, difussion_model, save_path, optimizer1=None, optimizer2=None, optimizer3=None):
    data2save = {
        'state_dict1': model.state_dict(),
        'optimizer1': optimizer1.state_dict(),
        'state_dict2': difussion_model.state_dict(),
        'optimizer2': optimizer2.state_dict()
    }
    torch.save(data2save, save_path)

def compute_bpr_loss(user_emb, pos_emb, neg_emb):
    """
    计算 BPR 损失
    返回:
        bpr_loss (torch.Tensor): 标量损失值
    """
    # 计算正负样本得分差
    pos_scores = torch.sum(user_emb * pos_emb, dim=-1)
    neg_scores = torch.sum(user_emb * neg_emb, dim=-1)
    bpr_loss = torch.nn.functional.softplus(neg_scores - pos_scores).mean() # 这直接计算了bprloss，把激活函数展开即可
    return bpr_loss

def mean_flat(tensor):
    # 对输入张量在 **所有非 batch 维度** 上求平均值。
    # 获取除 batch 维度外的所有维度编号，在这些维度上求平均值,[B,]
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    生成正弦/余弦形式的时间步嵌入（sinusoidal positional embeddings），
    用于将时间步 t 编码为一个固定维度的向量表示。
    :param timesteps: 一个一维 Tensor，shape = [N]，表示每个 batch 元素对应的时间步
    :param dim: 输出嵌入的维度
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    # 我们将分别用 cos 和 sin 各表示一半维度
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)  #一个线性序列，=[1.0000, 0.3162, 0.1000, 0.0316, 0.0100, 0.0032, 0.0010, 0.0003]
    temp = timesteps[:, None].float() * freqs[None] # 将时间步扩展为二维张量，并与频率相乘，wt：
    # 使用 cos 和 sin 构建位置编码：
    # embedding.shape = [N, half * 2] = [N, dim]
    embedding = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
    if dim % 2: # 如果 dim 是奇数，则在最后补一个 0，使最终维度等于 dim
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def parse_args():
    parser = argparse.ArgumentParser(description='SDR Arguments')

    parser.add_argument('--cuda', type=str, default='0', help='Cuda device')
    parser.add_argument('--seed', type=int, default=2023)

    # gcn Arguments
    parser.add_argument('--n_hid', type=int, default=64, help='Number of hidden units in the model')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers in the ui layer')
    parser.add_argument('--s_layers', type=int, default=2, help='Number of layers in uu layer')

    # Optimization Arguments
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate for general optimizer')
    parser.add_argument('--difflr', type=float, default=1e-4, help='Learning rate specifically for SDNet')
    parser.add_argument('--reg', type=float, default=1e-2, help='Regularization coefficient')
    parser.add_argument('--epoch', type=int, default=200, help='Total number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size used during training')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')

    # Valid/Test Arguments
    parser.add_argument('--topk', type=int, default=20, help='Top-k value used for evaluation metric calculation')
    parser.add_argument('--test_batch_size', type=int, default=2048, help='Batch size used during testing')

    # Data Arguments
    parser.add_argument('--dataset', type=str, default="yelp", help='Name of the dataset to be used')
    parser.add_argument('--save_name', type=str, default='temp', help='Filename used to save the trained model')
    parser.add_argument('--model_dir', type=str, default="./ckpt/yelp/", help='Directory to store model files')

    # params for the denoiser
    parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
    parser.add_argument('--dims', type=int, default=128, help='the dims for the DNN')
    parser.add_argument('--norm', type=bool, default=True, help='Normalize the input or not')
    parser.add_argument('--emb_size', type=int, default=16, help='timestep embedding size')

    # params for diffusions
    parser.add_argument('--steps', type=int, default=100, help='diffusion steps for SDnet')
    parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
    parser.add_argument('--noise_scale', type=float, default=10, help='noise scale for noise generating SDnet')
    parser.add_argument('--noise_min', type=float, default=0.001, help='noise lower bound for noise generating')
    parser.add_argument('--noise_max', type=float, default=0.1, help='noise upper bound for noise generating')
    parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
    parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')
    parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')

    # other params
    #parser.add_argument('--temperature', type=float, default=0.2, help='temperature coefficient')
    parser.add_argument('--warmup_epoch', type=int, default=30, help='only train GCN')
    parser.add_argument('--diff_epoch', type=int, default=60, help='only train diffusion')

    return parser.parse_args()
args = parse_args()

class DataHandler:
    def __init__(self):
        # 根据命令行参数指定的数据集名称（args.dataset）设置对应的数据路径
        predir = ''
        if args.dataset == 'yelp':
            predir = './data/yelp/'
            self.datapath = predir + 'dataset.pkl'
        elif args.dataset == 'ciao':
            predir = './data/ciao/'
            self.datapath = predir + 'dataset.pkl'
        elif args.dataset == 'epinions':
            predir = './data/epinions/'
            self.datapath = predir + 'dataset.pkl'
        self.predir = predir

    def loadOneFile(self,data_path):
        # 加载数据集
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        return data

    def LoadData(self):
        # 加载整个数据集，返回的是一个包含训练、测试等信息的字典
        self.dataset = self.loadOneFile(self.datapath)
        trnMat = self.dataset['train']
        tstMat = self.dataset['test']
        valMat = self.dataset['val']
        # 封装dataset对象
        trainset = TrainData(trnMat)
        testset = TestData(tstMat, trnMat)
        valset = TestData(valMat, trnMat)
        # 获取用户和物品数量并赋值
        self.n_user, self.n_item = self.dataset['userCount'], self.dataset['itemCount']
        args.user, args.item = self.n_user, self.n_item
        # 构建训练、测试、验证数据集的数据加载器对象
        self.trainloader = DataLoader(
            dataset=trainset,
            batch_size=args.batch_size,
            shuffle=True,
        )
        self.valloader = DataLoader(
            dataset=valset,
            batch_size=args.test_batch_size,
            shuffle=False,
        )
        self.testloader = DataLoader(
            dataset=testset,
            batch_size=args.test_batch_size,
            shuffle=False,
        )
        # 使用dgl从稀疏矩阵构建  社交图
        self.uu_graph = dgl.from_scipy(self.dataset['trust'])
        # 获取用户-物品交互矩阵（CSR格式），用于构建  二分图
        self.uimat = self.dataset['train'].tocsr()
        # 调用 makeBiAdj 方法构建用户-物品二分图
        self.ui_graph = self.makeBiAdj(self.uimat, self.n_user, self.n_item)
        self.ui_graph = dgl.add_self_loop(self.ui_graph)
        self.uu_graph = dgl.add_self_loop(self.uu_graph)


    def makeBiAdj(self, mat, n_user, n_item):
        # 创建两个 零 矩阵：a 为用户-用户矩阵，b 为物品-物品矩阵
        a = sp.csr_matrix((n_user, n_user))
        b = sp.csr_matrix((n_item, n_item))
        # sp.hstack，sp.vstack分别表示水平堆叠和垂直堆叠
        # mat上左角是用户-用户形状的全零矩阵，上右角是用户-物品矩阵，下左角是物品-用户矩阵（即用户-物品矩阵的转置），下右角是物品-物品形状的全零矩阵。
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])]) # [m+n,m+n]
        mat = (mat != 0) * 1.0  #原始矩阵中的非零元素表示用户与物品之间存在交互，
        mat = mat.tocoo()   #转换为 COO 格式以便后续构建图结构
        edge_src,edge_dst = mat.nonzero()   # 获取非零元素的行和列索引
        # 创建一个 DGL 图对象，idtype 为 torch.int32，num_nodes 为 m+n
        ui_graph = dgl.graph(data=(edge_src, edge_dst),
                            idtype=torch.int32,
                             num_nodes=mat.shape[0]
                             )
        return ui_graph  #确实只是用户物品交互图。

class TrainData(data.Dataset):
    def __init__(self, coomat):
        # 从稀疏矩阵中提取所有非零元素的行索引（用户ID）和列索引（物品ID）
        self.rows = coomat.row
        self.cols = coomat.col
        # 将稀疏矩阵转换为字典形式 {key: (user, item),  value: 1}，便于快速查找是否存在某个交互
        self.dokmat = coomat.todok()
        # 初始化一个与正样本数量相同的数组，用来保存为每个正样本采样的负样本物品
        self.negs = np.zeros( len(self.rows) ).astype(np.int32)

    def negSampling(self):
        # 负采样函数：为每一个正样本生成一个未交互过的负样本物品
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                # 随机生成一个物品 ID 作为候选负样本
                iNeg = np.random.randint(args.item) # args.item来自DataHandler里
                if (u, iNeg) not in self.dokmat:    # 如果该用户 u 没有和这个物品 iNeg 发生过交互，则选中它作为负样本
                    break
            self.negs[i] = iNeg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        # 获取单个样本：返回用户ID、正样本物品ID、负样本物品ID
        return self.rows[idx], self.cols[idx], self.negs[idx]

class TestData(data.Dataset):
    def __init__(self, coomat, trnMat):
        # 将 训练集 用户-物品交互矩阵转换为 CSR 格式，并二值化为 0/1 矩阵（表示是否交互过）,不等于0的位置为1，即有交互的位置为1
        self.csrmat = (trnMat.tocsr() != 0) * 1.0

        # 初始化测试集中每个用户的正样本位置列表，也是测试集用户ID列表
        tstLocs = [None] * coomat.shape[0]
        # 用于记录所有有测试样本的用户ID
        tstUsrs = set()
        # 遍历测试集中的每一条交互记录（即稀疏矩阵中的非零元素）
        for i in range(len(coomat.data)):
            row = coomat.row[i] # 用户id
            col = coomat.col[i] # 物品id，也是位置
            # 如果该用户还没有初始化正样本列表，则创建一个空列表
            if tstLocs[row] is None:
                tstLocs[row] = list()
            # 将当前物品加入该用户的正样本列表中，将该用户加入测试用户集合中
            tstLocs[row].append(col)
            tstUsrs.add(row)
        # 将测试用户集合转换为 NumPy 数组，便于后续索引访问
        tstUsrs = np.array(list(tstUsrs))
        self.tstUsrs = tstUsrs   # 所有测试用户ID列表
        self.tstLocs = tstLocs   # 每个用户对应的测试正样本物品位置列表，两重列表

    def __len__(self):
        return len(self.tstUsrs)

    def __getitem__(self, idx):
        # 获取指定索引处的用户ID
        usr = self.tstUsrs[idx]

        # 获取该用户在 训练集 中交互过的所有物品的历史（转换为一维数组）
        mask_vec = np.reshape(self.csrmat[usr].toarray(), [-1])

        # 返回用户ID和其历史交互物品的掩码向量（用于后续评估时排除已交互物品）
        return usr, mask_vec

class DiffusionProcess(nn.Module):
    def __init__(self,noise_schedule, noise_scale, noise_min, noise_max, steps, device, keep_num=10):
        # 噪声调度方式linear-var,噪声强度1,噪声下界0.0001,噪声上界0.01,步数20,设备gpu,保存的采样点数10
        super(DiffusionProcess, self).__init__()
        self.noise_schedule = noise_schedule
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps
        self.device = device

        self.keep_num = keep_num    #每个 diffusion step 中记录损失的样本数量。
        self.Lt_record = torch.zeros(steps, keep_num, dtype=torch.float64).to(device) # 记录每个时间步 t 的损失值（最多保留 keep_num 个）。全0张量
        self.Lt_count = torch.zeros(steps, dtype=torch.int64).to(device)    # 每个时间步记录了多少个损失值，用于计算平均损失。

        #The important parameters for gaussian diffusion
        self.beta_nums = torch.tensor(self.betas_num(), dtype=torch.float64).to(self.device)    # 调用 self.betas_num() 方法生成每一步的噪声系数 β_t。
        # 进行一系列断言检查，确保 β 参数是合法的一维张量，并且数值在合理范围 [0, 1] 内。
        assert len(self.beta_nums.shape) == 1, "betas must be 1-D"
        assert len(self.beta_nums) == self.steps, "num of betas must equal to diffusion steps"
        assert (self.beta_nums > 0).all() and (self.beta_nums <= 1).all(), "betas out of range"

        self.diffusion_setting()
        self.sigma = 2.5

    def betas_num(self):
        """
        给定调度名称，为扩散过程创建beta。
        - 'linear': 直接线性插值得到 beta_t
        - 其他（默认）: 基于线性变化的方差计算 beta_t（保证 alpha_bar 递减）
        返回:  np.ndarray: shape=(steps, ), 每个时间步对应的 beta 值，范围在 (0, max_beta] 之间
        """
        # 计算起始和结束边界
        st_bound = self.noise_scale * self.noise_min
        e_bound = self.noise_scale * self.noise_max

        if self.noise_schedule == "linear":
            # 线性调度：beta_t 在 [st_bound, e_bound] 区间内均匀分布
            return np.linspace(st_bound, e_bound, self.steps, dtype=np.float64) # 在[st_bound, e_bound]内均匀采样step个点
        else:
            # 非线性调度：基于线性方差的方法计算 beta_t。这里和DDRM中是同一种方法
            variance = np.linspace(st_bound, e_bound, self.steps, dtype=np.float64) # 在[st_bound, e_bound]内均匀采样step个点，逐渐增加
            alpha_bar = 1.0 - variance  # 这就是 alpha_bar，alpha的连乘，这里alpha_bar设置成这样就是递减的。然后利用alpha_bar反推beta_t
            betas = []
            betas.append(1.0 - alpha_bar[0])  # 第一个 beta

            # 通过 alpha_bar[t] / alpha_bar[t-1] 推导后续 beta
            for i in range(1, self.steps):
                beta = 1.0 - alpha_bar[i] / alpha_bar[i - 1]
                max_beta = 0.999  # 限制最大 beta 值
                betas.append(min(beta, max_beta))

            return np.array(betas, dtype=np.float64)

    def diffusion_setting(self):
        # 各种系数的预先计算
        alphas = 1.0 - self.beta_nums
        self.alphas_cumprod = torch.cumprod(alphas, dim=0).to(self.device)  #连乘，alpha_bar{t}
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(self.device), self.alphas_cumprod[:-1]]).to(self.device)  # alpha_bar{t-1}
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).to(self.device)]).to(self.device)  # alpha_bar{t+1}
        assert self.alphas_cumprod_prev.shape == (self.steps,)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # 后验分布的方差
        self.posterior_variance = (
            self.beta_nums * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # 对后验方差取对数,第一个时间步的方差设为与第二个相同（因为 t=0 无后验意义）。
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
                self.beta_nums * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * torch.sqrt(alphas)
                / (1.0 - self.alphas_cumprod)
        )


    def caculate_losses(self, model, emb_s, reweight=False):
        # model是 SDNet或CDNet，社交图编码的用户embedding，协作信息embedding，加权True
        # 返回：dict: 包含 'loss' 和 'pred_xstart' 的字典
        batch_size, device = emb_s.size(0), emb_s.device
        ts, _ = self.sample_timesteps(batch_size, device, 'uniform')   # 从采样策略（uniform）中随机选取一批时间步 ts和采样概率
        noise = torch.randn_like(emb_s)
        emb_t = self.forward_process(emb_s, ts, noise)  # 为嵌入添加标准正态分布噪声（模拟第 t 步的噪声）
        model_output = model(emb_t, ts) # 将当前带噪声的嵌入 emb_t 和时间步 ts 输入模型，得到embed预测

        assert model_output.shape == emb_s.shape, "模型输出 shape 不匹配"
        mse = mean_flat((model_output - emb_s) ** 2)    # 均方误差（每个样本一个标量）

        if reweight == True:
            weight = self.SNR(ts - 1) - self.SNR(ts)
            weight = torch.where(ts == 0, torch.tensor(1.0).to(device), weight)
        else:
            weight = torch.tensor([1.0] * len(model_output)).to(device)

        return {
            "loss": weight * mse,
            "pred_xstart": model_output  # 返回不带梯度的预测值用于记录
        }

    @torch.no_grad()  # 禁用梯度计算，节省内存
    def p_sample(self, model, emb_s, steps, sampling_noise=False):
        """
        执行扩散模型的反向过程（inference阶段），从带噪嵌入中恢复"干净"的用户嵌入。
        Args:
            model (nn.Module): 去噪神经网络（SDNet）
            emb_s (Tensor): 初始带噪嵌入（来自社交图编码）
            con_emb:协作信息
            steps (int): 扩散模型中前向过程所使用的步数，若为 0 表示直接返回输入
            sampling_noise (bool): 是否在每一步采样中加入噪声，默认 False
        Returns:
            Tensor: 去噪后的用户嵌入
        """
        # 检查输入的去噪步数不能超过训练时的最大步数
        assert steps <= self.steps, "Too much steps in inference."
        if steps == 0:
            emb_t = emb_s
        else: # 否则从初始时间步开始加噪（此处未用到）
            t = torch.tensor([steps - 1] * emb_s.shape[0]).to(emb_s.device)
            emb_t = self.forward_process(emb_s, t)  # 前向过程
        # 获取所有时间步索引并倒序排列（从高噪声到低噪声）[19,18,17,,,0],从 T->1
        indices = list(range(self.steps))[::-1]
        # 如果noise_scale == 0，则跳过扩散过程，直接使用模型输出作为结果
        if self.noise_scale == 0.:
            for i in indices:
                t = torch.tensor([i] * emb_t.shape[0]).to(emb_s.device)
                emb_t = model(emb_t, t) # 直接通过 SDNet 得到输出
            return emb_t

        # 正常的多步去噪过程,一共T步
        for i in indices:
            t = torch.tensor([i] * emb_t.shape[0]).to(emb_s.device)
            # 使用模型预测当前时间步的均值和方差
            # 利用模型预测当前时间步的去噪结果（x_0）并计算下一步采样的均值和方差
            # out是字典: 包含以下键值：均值，方差，对数方差，预测的x0
            out = self.p_mean_variance(model, emb_t, t)
            # 如果启用采样噪声，在非 t=0 时间步添加随机噪声
            if sampling_noise:
                noise = torch.randn_like(emb_t) #随机产生正态分布的噪声
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(emb_t.shape) - 1)))
                )  # 在 t == 0 时不加噪声。t=0的位置为0，其他位置为1.*是解包，调整张量形状
                emb_t = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
            else:   #不启用时，直接赋值
                emb_t = out["mean"] #这是 x_{t-1}
        return emb_t    #经过多步去噪，最终得到x^_0

    def sample_timesteps(self, batch_size, device, method='uniform', uniform_prob=0.001):
        # 时间步中采样一批 t，用于训练时选择扩散步骤。
        if method == 'importance':  # importance sampling，目的：让模型更关注那些 loss 较大的时间步（即“困难”的去噪阶段）
            if not (self.Lt_count == self.keep_num).all():
                return self.sample_timesteps(batch_size, device, method='uniform')
            Lt_sqrt = torch.sqrt(torch.mean(self.Lt_record ** 2, axis=-1))
            pt_all = Lt_sqrt / torch.sum(Lt_sqrt)
            pt_all *= 1 - uniform_prob
            pt_all += uniform_prob / len(pt_all)
            assert pt_all.sum(-1) - 1. < 1e-5
            t = torch.multinomial(pt_all, num_samples=batch_size, replacement=True)
            pt = pt_all.gather(dim=0, index=t) * len(pt_all)
            return t, pt
        elif method == 'uniform':  # uniform sampling，目的：简单粗暴地对所有时间步等概率采样
            t = torch.randint(0, self.steps, (batch_size,), device=device).long()   # 随机生成 [0, steps) 范围内的整数，表示每个样本的时间步
            pt = torch.ones_like(t).float() # 所有时间步的采样概率相同，均为 1.0
            return t, pt
        else:
            raise ValueError(f"未知采样方法: {method}")

    def forward_process(self, emb_s, t, noise=None):
        # 执行前向扩散过程：根据时间步 t 对原始数据 x_0 添加高斯噪声得到 x_t。这个函数原名q_sample
        if noise is None:
            noise = torch.randn_like(emb_s) # 生成标准正态分布噪声
        assert noise.shape == emb_s.shape   # 确保噪声和embed同样形状
        # 前向扩散过程，由x0求xt的公式
        return (
                self._extract_into_tensor(self.sqrt_alphas_cumprod, t, emb_s.shape) * emb_s
                + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, emb_s.shape) * noise
        )

    def q_posterior_mean_variance(self, emb_s, emb_t, t):
        """
        计算扩散模型中后验分布 q(x_{t-1} | x_t, x_0) 的均值和方差。
        给定当前带噪数据 x_t 和原始数据 x_0（或其预测值），计算前一个时间步 x_{t-1} 的分布参数，用于指导下一步采样（去噪）。
        Args:
            emb_s (Tensor): 对原始数据 x_0 的估计（通常是模型输出 pred_xstart）
            emb_t (Tensor): 当前时间步的带噪数据 x_t
            t (Tensor): 当前时间步索引，shape = [B]
        Returns:
                posterior_mean,               # 后验均值 μ(x_t, t)
                posterior_variance,           # 后验方差 σ²(x_t, t)
                posterior_log_variance_clipped # 后验方差的对数（数值稳定用）
        """
        # 从预定义的系数中提取对应时间步的均值系数
        coef1 = self._extract_into_tensor(self.posterior_mean_coef1, t, emb_t.shape)
        coef2 = self._extract_into_tensor(self.posterior_mean_coef2, t, emb_t.shape)
        # 根据公式计算后验均值：
        posterior_mean = coef1 * emb_s + coef2 * emb_t

        posterior_variance = self._extract_into_tensor(self.posterior_variance, t, emb_t.shape)
        posterior_log_variance_clipped = self._extract_into_tensor(self.posterior_log_variance_clipped, t, emb_t.shape)
        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == emb_s.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t):
        """
        该函数用于扩散模型的推理阶段（去噪过程），其核心作用是：
            利用模型预测当前时间步的去噪结果（x_0）并计算下一步采样的均值和方差
        Args:
            model (nn.Module): 扩散过程中的去噪网络（如 SDNet）
            x (Tensor): 当前时间步的带噪嵌入张量，shape = [B, 64]（B: batch size）
            t (Tensor): 当前时间步索引，shape = [B]
        Returns:
            dict: 包含以下键值：
                - "mean": 下一时间步的均值 μ(x_t, t)
                - "variance": 对应的方差 σ²(x_t, t)
                - "log_variance": 方差的对数形式（用于数值稳定性）
                - "pred_xstart": 模型对初始输入 x_0 的预测结果
        """
        B, C = x.shape[:2]
        assert t.shape == (B,)
        # 使用模型预测当前时间步下的输出x0
        model_output = model(x,t)
        # 从扩散过程中提取对应的 后验方差和对数方差（训练时已定义）
        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped
        # 根据当前时间步 t 提取出对应的方差和对数方差值（广播至 x 相同 shape）
        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)
        # 输出 model_output 即为模型对 x_0（原始用户嵌入）的预测 pred_xstart
        pred_xstart = model_output
        # 计算下一时间步的条件分布均值μ(x_t, t)
        # 调用 q_posterior_mean_variance 函数，传入了预测的 x_0（pred_xstart）、当前 x_t 和 t。得到 μ(x_t, t)
        model_mean, _, _ = self.q_posterior_mean_variance(emb_s=pred_xstart, emb_t=x, t=t)
        # 断言确保所有输出张量的维度一致
        assert (
                model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        # 均值，方差，对数方差，预测的x0
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, emb_t, t, eps):
        # 是扩散模型中用于从带噪样本 x_t 和噪声预测 eps 推导出原始样本 x_0 的估计值的重要函数
        assert emb_t.shape == eps.shape
        # 提取两个预定义的系数
        coef1 = self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, emb_t.shape)
        coef2 = self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, emb_t.shape)

        # 公式：x0 = coef1 * x_t - coef2 * eps
        x0_pred = coef1 * emb_t - coef2 * eps

        return x0_pred

    def SNR(self, t):
        """
        计算指定时间步 t 的 信噪比SNR：SNR 表示当前状态中原始信号成分与噪声成分的比例，
        返回：SNR 值，shape 与 t 相同，表示每个时间步对应的信噪比
        """
        self.alphas_cumprod = self.alphas_cumprod.to(t.device)  # 连乘，alpha_bar{t}
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])

    def _extract_into_tensor(self, tensor, timesteps, broadcast_shape):
        """
        从一个一维张量中根据时间步 `timesteps` 提取对应的值，并将其扩展为指定的广播形状 `broadcast_shape`，以便进行后续的张量运算。
        tensor (Tensor): 一维张量，shape = [T]，表示 T 个时间步的系数
        timesteps (Tensor): 时间步索引，shape = [B]
        broadcast_shape (tuple): 目标广播形状，通常与 emb_s 形状一致
        返回：扩展后的张量，shape = broadcast_shape
        """
        tensor = tensor.to(timesteps.device)
        res = tensor[timesteps].float() #索引，[B]
        # 如果 res 的维度小于目标广播形状，则不断在末尾添加新维度（None 表示新增一个轴）。
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

class UUGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, bias=False, activation=None):
        # 用户-用户图卷积层
        super(UUGCNLayer, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats

        # 使用 DGL 提供的 GraphConv 实现图卷积操作
        self.gc = GraphConv(in_feats, out_feats, norm='both', weight=True, bias=bias)
        self._activation = activation

    def forward(self, graph, u_f):
        with graph.local_scope():
            h = self.gc(graph, u_f)  # GraphConv 自动进行线性变换 + 图聚合
            if self._activation is not None:
                h = self._activation(h)
            return h

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, bias=False, activation=None):
        """
        用户-物品图卷积层，使用 GraphConv 内置权重，不再手动定义 u_w / v_w
        """
        super(GCNLayer, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats

        # 使用 DGL 的 GraphConv 进行图卷积操作，并启用内置权重
        self.gc = GraphConv(in_feats, out_feats, norm='both', weight=True, bias=bias)
        self._activation = activation

    def forward(self, graph, u_f, v_f):
        with graph.local_scope():
            # 将用户和物品特征拼接为完整节点特征
            node_f = torch.cat([u_f, v_f], dim=0)
            h = self.gc(graph, node_f)  # 自动进行线性变换 + 图聚合
            if self._activation is not None:
                h = self._activation(h)
            return h

class GCNModel(nn.Module):
    def __init__(self, args, n_user, n_item):
        super(GCNModel, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.n_hid = args.n_hid # 隐藏层维度
        self.n_layers = args.n_layers   # 交互图的图卷积层数
        self.s_layers = args.s_layers   # 社交图的图卷积层数
        self.embedding_dict = self.init_weight(n_user, n_item, self.n_hid)  # 初始化用户和物品的嵌入向量
        self.act = nn.LeakyReLU(0.5, inplace=True)  # LeakyReLU 激活函数，斜率 0.5
        self.layers = nn.ModuleList()   # 存放用户-物品图的多层 GCN 层
        self.uu_Layers = nn.ModuleList()    # 存放用户-用户图的多层 UU-GCN 层

        # 构建用户-物品图的多层 GCN 网络
        for i in range(self.n_layers):
            self.layers.append(GCNLayer(self.n_hid, self.n_hid, bias=False, activation=self.act))
        # 构建用户-用户图的多层 UU-GCN 网络
        for i in range(self.s_layers):
            self.uu_Layers.append(UUGCNLayer(self.n_hid, self.n_hid, bias=False, activation=self.act))

    def init_weight(self, userNum, itemNum, hid_dim):
        # 初始化用户和物品的嵌入向量，维度是hid_dim，使用 Xavier 均匀分布初始化方法。
        initializer = nn.init.xavier_uniform_   # 初始化器

        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(userNum, hid_dim))),
            'item_emb': nn.Parameter(initializer(torch.empty(itemNum, hid_dim))),
        })
        return embedding_dict

    def forward(self, uigraph, uugraph, isTrain=True):
        """
        前向传播函数，用于图神经网络的消息传递与节点表示学习。
        参数:
            uigraph: 用户-物品交互图（DGLGraph 对象）
            uugraph: 用户-用户社交图（DGLGraph 对象）
        返回:
            ui_embeddings: 经过用户-物品图多层传播后的最终用户+物品嵌入
            uu_embeddings: 经过用户-用户图多层传播后的最终用户嵌入
        """
        # uigraph = dgl.add_self_loop(uigraph)
        # uugraph = dgl.add_self_loop(uugraph)
        # 初始化完整的节点嵌入：将用户嵌入和物品嵌入拼接在一起
        init_embedding = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], dim=0)
        init_user_embedding = self.embedding_dict['user_emb']
        all_embeddings = [init_embedding]           # 存储每一层用户-物品图传播后的完整嵌入
        all_uu_embeddings = [init_user_embedding]   # 存储每一层用户-用户图传播后的用户嵌入

        # 用户-物品
        embeddings = init_embedding
        for layer in self.layers:
            # 将当前嵌入拆分为用户部分和物品部分，传入 GCNLayer 进行传播
            embeddings = layer(uigraph, embeddings[:self.n_user], embeddings[self.n_user:])
            norm_embeddings = F.normalize(embeddings, p=2, dim=1)   # L2 归一化嵌入向量，防止梯度爆炸并提升泛化能力
            all_embeddings.append(norm_embeddings)
        ui_embeddings = sum(all_embeddings) # 所有层的输出进行累加（类似残差连接或集成），得到最终的用户+物品嵌入

        # 用户-用户
        embeddings = init_user_embedding
        for layer in self.uu_Layers:
            embeddings = layer(uugraph, embeddings)
            norm_embeddings = F.normalize(embeddings, p=2, dim=1)
            all_uu_embeddings.append(norm_embeddings)
        uu_embeddings = sum(all_uu_embeddings)
        # 返回最终的用户+物品嵌入 和 用户嵌入，供后续推荐任务使用
        return ui_embeddings, uu_embeddings

class DDNet(nn.Module):
    """
    一个用于 **扩散模型逆向过程** （去噪）的深度神经网络。
    输入：带噪声的社会关系嵌入 + 时间步嵌入
    输出：去噪后的社会关系嵌入
    """
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5):
        super(DDNet, self).__init__()
        # 编码器、解码器部分各层输入维度列表
        #  input_dims  [64，128]，output_dims [128，64]，16，cat，True，默认0.5
        self.in_dims = in_dims
        self.out_dims = out_dims

        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm
        # 定义时间步嵌入的线性层
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)    # 16，16
        # 如果时间嵌入方式是拼接（cat），则增加编码器输入维度
        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]     #[80,128]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims   #[128，64]

        # 构建编码器、解码器部分的全连接层（MLP）。都是单层的，80->128, 128->64
        self.in_layers = nn.ModuleList( [ nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:]) ] )
        self.out_layers = nn.ModuleList( [ nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:]) ] )

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        def _init_layer(layer):
            # 使用 Xavier 初始化权重
            init.xavier_normal_(layer.weight.data)
            # 使用正态分布初始化偏置
            init.normal_(layer.bias.data, mean=0.0, std=0.001)

        # 对编码器中每一层进行初始化
        for layer in self.in_layers:
            _init_layer(layer)
        # 对解码器中每一层进行初始化
        for layer in self.out_layers:
            _init_layer(layer)
        # 对时间嵌入层进行初始化
        _init_layer(self.emb_layer)

    def forward(self, x, timesteps):
        # 带噪声的嵌入 emb_t 和时间步 ts
        # 输出是预测的embedding
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.norm:   # True
            x = F.normalize(x)
        x = self.drop(x)

        h = torch.cat([x, emb], dim=-1)
        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)

        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:   #如果不是最后一层（即 i 不是最后的索引），则对输出 h 应用 tanh 激活函数。
                h = torch.tanh(h)
        return h

class Coach:
    def __init__(self, handler, args):

        self.args = args  # 保存全局参数配置
        self.device = torch.device('cuda' if self.args.cuda and torch.cuda.is_available() else 'cpu')
        self.handler = handler  # 保存数据处理器对象 handler
        # 从handler获取各项数据
        self.train_loader = self.handler.trainloader
        self.valloader = self.handler.valloader
        self.testloader = self.handler.testloader
        self.n_user, self.n_item = self.handler.n_user, self.handler.n_item

        self.uigraph = self.handler.ui_graph.to(self.device)
        self.uugraph = self.handler.uu_graph.to(self.device)
        # 初始化 GCNModel（图卷积网络模型）并移动到设备
        self.GCNModel = GCNModel(self.args, self.n_user, self.n_item).to(self.device)

        ### 构建扩散模型部分 ###
        input_dims = [self.args.n_hid] + [self.args.dims]  # [64，128]
        output_dims = [self.args.dims] + [self.args.n_hid]  # [128，64]
        # 初始化 SDNet（社交图扩散模型的去噪网络）并移动到设备
        self.SDNet = DDNet(input_dims, output_dims, self.args.emb_size, time_type="cat", norm=self.args.norm).to(self.device)
        # 初始化 CDNet（交互图扩散模型的去噪网络）并移动到设备
        self.CDNet = DDNet(input_dims, output_dims, self.args.emb_size, time_type="cat", norm=self.args.norm).to(self.device)
        # 构建社交图的 DiffusionProcess 对象，定义扩散过程中的噪声调度等参数，并移动到设备
        self.DiffProcess = DiffusionProcess(self.args.noise_schedule, self.args.noise_scale, self.args.noise_min, self.args.noise_max, self.args.steps, self.device).to(self.device)
        # 构建交互图的 DiffusionProcess 对象，使用相同参数
        self.IDiffProcess = DiffusionProcess(self.args.noise_schedule, self.args.noise_scale, self.args.noise_min, self.args.noise_max, self.args.steps, self.device).to(self.device)

        # 设置两个优化器，学习率不同
        self.optimizer1 = torch.optim.Adam([
            {'params': self.GCNModel.parameters(), 'weight_decay': 0},
        ], lr=self.args.lr)
        self.optimizer2 = torch.optim.Adam([
            {'params': self.SDNet.parameters(), 'weight_decay': 0},
            {'params': self.CDNet.parameters(), 'weight_decay': 0},  # 添加CDNet参数
        ], lr=self.args.difflr)

        # 替换现有的调度器
        self.scheduler1 = CosineAnnealingLR(
            self.optimizer1,
            T_max=self.args.epoch,  # 周期长度等于总训练轮数
            eta_min=1e-5       # 最小学习率
        )
        self.scheduler2 = CosineAnnealingLR(
            self.optimizer2,
            T_max=self.args.epoch,
            eta_min=1e-5
        )

        # 记录训练损失、验证集 recall 和 ndcg 指标
        self.train_loss = []
        self.his_recall = []
        self.his_ndcg = []
        # 记录每个 epoch 的平均损失，便于绘画分析
        self.sum_loss = []
        self.uue_losses = []
        self.iue_losses = []  # 交互图扩散损失
        self.reg_losses = []
        self.bpr_losses = []

    def train(self):
        # 配置日志记录（控制台 + 文件）
        self.save_history = True
        log_format = '%(asctime)s %(message)s'
        # 配置日志记录器，输出到控制台
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S ')
        log_save = './logs/' + args.dataset + '/'  # 日志文件保存路径
        os.makedirs(log_save, exist_ok=True)
        log_file = args.save_name
        # 记录器实例
        logger = logging.getLogger()
        global flag
        if flag==1:
            fname = f'{log_file}.txt'
            # 添加 FileHandler，将日志同时写入本地文件
            fh = logging.FileHandler(os.path.join(log_save, fname))
            fh.setFormatter(logging.Formatter(log_format))
            logger.addHandler(fh)  # 将文件 handler 加入 logger
            flag = 0
        logger.info(self.args)
        logger.info('=======================')

        best_recall, best_ndcg, best_epoch, wait = 0, 0, 0, 0  # wait用于早停
        start_time = time.time()

        for self.epoch in range(1, self.args.epoch + 1):
            epoch_losses = self.train_one_epoch()  # 调用训练函数，返回损失值列表
            self.train_loss.append(epoch_losses)
            print('epoch {} done! elapsed {:.2f}.s, epoch_losses {}'.format(
                self.epoch, time.time() - start_time, epoch_losses), flush=True)

            # 每隔 5 个 epoch 测试一次模型在验证集/测试集上的表现
            if self.epoch % 5 == 0:
                recall, ndcg = self.test(self.testloader)
                # 保存本次测试的平均 Recall 和 NDCG 历史记录
                self.his_recall.append(recall)
                self.his_ndcg.append(ndcg)
                if self.epoch > self.args.diff_epoch:
                    cur_best = recall + ndcg > best_recall + best_ndcg
                    if cur_best:
                        best_recall, best_ndcg, best_epoch = recall, ndcg, self.epoch
                        wait = 0
                    else:
                        wait += 1
                    # logger记录本次
                    logger.info('+ epoch {} tested, elapsed {:.2f}s, Recall@{}: {:.4f}, NDCG@{}: {:.4f}, lr:{:.5f}, diff_lr:{:.5f}'.format(
                        self.epoch, time.time() - start_time, self.args.topk, recall, self.args.topk, ndcg,
                        self.optimizer1.param_groups[0]["lr"], self.optimizer2.param_groups[0]["lr"]))
                    if wait >= self.args.patience:
                        print(f'Early stop at epoch {self.epoch}, best epoch {best_epoch}')
                        break
        # 如果启用了历史记录保存功能，将当前状态保存下来
        # if self.save_history:
        #     self.saveHistory()
        logger.info(f'Best  Recall@{self.args.topk} {best_recall:.5f}, NDCG@{self.args.topk} {best_ndcg:.5f} \n')
        print(f'Best  Recall@{self.args.topk} {best_recall:.5f}, NDCG@{self.args.topk} {best_ndcg:.5f}', flush=True)
        return best_recall, best_ndcg

    def train_one_epoch(self):
        epoch = self.epoch
        # ===== Stage control (parameter freezing) =====
        if epoch <= self.args.warmup_epoch:
            for p in self.GCNModel.parameters():
                p.requires_grad = True
            for p in self.SDNet.parameters():
                p.requires_grad = False
            for p in self.CDNet.parameters():
                p.requires_grad = False

        elif epoch <= self.args.diff_epoch:
            for p in self.GCNModel.parameters():
                p.requires_grad = False
            for p in self.SDNet.parameters():
                p.requires_grad = True
            for p in self.CDNet.parameters():
                p.requires_grad = True
        else:
            for p in self.GCNModel.parameters():
                p.requires_grad = True
            for p in self.SDNet.parameters():
                p.requires_grad = True
            for p in self.CDNet.parameters():
                p.requires_grad = True

        self.GCNModel.train()
        self.SDNet.train()
        self.CDNet.train()

        dataloader = self.train_loader
        dataloader.dataset.negSampling()
        tqdm_dataloader = tqdm(dataloader)

        epoch_losses = [0] * 5  # bpr, reg, social_diff, interact_diff, contrast
        since = time.time()

        for batch in tqdm_dataloader:
            user_idx, pos_idx, neg_idx = batch
            user_idx = user_idx.long().cuda()
            pos_idx = pos_idx.long().cuda()
            neg_idx = neg_idx.long().cuda()

            # ===== GCN forward (always needed) =====
            uiEmbeds, uuEmbeds = self.GCNModel(self.uigraph, self.uugraph, True)
            uEmbeds = uiEmbeds[:self.n_user]
            iEmbeds = uiEmbeds[self.n_user:]

            user = uEmbeds[user_idx]
            pos = iEmbeds[pos_idx]
            neg = iEmbeds[neg_idx]

            # ===== Init losses =====
            bprLoss = torch.zeros(1, device=user.device)
            regLoss = torch.zeros(1, device=user.device)
            diffloss_s = torch.zeros(1, device=user.device)
            diffloss_i = torch.zeros(1, device=user.device)

            # =========================================================
            # Stage 1: pure GCN (recommendation warmup)
            # =========================================================
            if epoch <= self.args.warmup_epoch:
                final_user = user + uuEmbeds[user_idx]
                bprLoss = compute_bpr_loss(final_user, pos, neg)
                regLoss = ((torch.norm(final_user) ** 2 +
                            torch.norm(pos) ** 2 +
                            torch.norm(neg) ** 2) * self.args.reg) / 2 / self.args.batch_size
                loss = bprLoss + regLoss
            # =========================================================
            # Stage 2: diffusion only (static target)
            # =========================================================
            elif epoch <= self.args.diff_epoch:
                user_static = user.detach()
                uu_static = uuEmbeds[user_idx].detach()

                uu_terms = self.DiffProcess.caculate_losses(self.SDNet, uu_static, self.args.reweight)
                ui_terms = self.IDiffProcess.caculate_losses(self.CDNet, user_static, self.args.reweight)

                diffloss_s = uu_terms["loss"].mean()
                diffloss_i = ui_terms["loss"].mean()
                loss = diffloss_s + diffloss_i
            # =========================================================
            # Stage 3: joint training
            # =========================================================
            else:
                uu_terms = self.DiffProcess.caculate_losses(self.SDNet, uuEmbeds[user_idx], self.args.reweight)
                ui_terms = self.IDiffProcess.caculate_losses(self.CDNet, user, self.args.reweight)

                diffloss_s = uu_terms["loss"].mean()
                diffloss_i = ui_terms["loss"].mean()

                final_user = user + uu_terms["pred_xstart"] + ui_terms["pred_xstart"]

                bprLoss = compute_bpr_loss(final_user, pos, neg)
                regLoss = ((torch.norm(final_user) ** 2 +torch.norm(pos) ** 2 +
                            torch.norm(neg) ** 2) * self.args.reg) / 2 / self.args.batch_size
                loss = bprLoss + regLoss + diffloss_s + diffloss_i

            # ===== Backward & step =====
            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            loss.backward()

            if epoch <= self.args.warmup_epoch:
                self.optimizer1.step()
            elif epoch <= self.args.diff_epoch:
                self.optimizer2.step()
            else:
                self.optimizer1.step()
                self.optimizer2.step()

            # ===== Record =====
            epoch_losses[0] += bprLoss.item()
            epoch_losses[1] += regLoss.item()
            epoch_losses[2] += diffloss_s.item()
            epoch_losses[3] += diffloss_i.item()

        # ===== Scheduler =====
        if epoch <= self.args.warmup_epoch:
            self.scheduler1.step()
        elif epoch <= self.args.diff_epoch:
            self.scheduler2.step()
        else:
            self.scheduler1.step()
            self.scheduler2.step()

        epoch_losses = [sum(epoch_losses)] + epoch_losses
        self.sum_loss.append(epoch_losses[0])
        self.bpr_losses.append(epoch_losses[1])
        self.reg_losses.append(epoch_losses[2])
        self.uue_losses.append(epoch_losses[3])
        self.iue_losses.append(epoch_losses[4])

        print(f"Training complete in {time.time() - since:.4f}s")
        return epoch_losses


    def test(self, dataloader):
        self.SDNet.eval()
        self.CDNet.eval()
        self.GCNModel.eval()
        Recall, NDCG = [0] * 2
        num = dataloader.dataset.__len__()  # 总共需要处理的样本数（用户数）

        since = time.time()
        with torch.no_grad():
            # 前向传播获取所有用户和物品的嵌入表示
            uiEmbeds, uuEmbeds = self.GCNModel(self.uigraph, self.uugraph, True)
            uEmbeds = uiEmbeds[:self.n_user]  # 交互图的用户嵌入
            iEmbeds = uiEmbeds[self.n_user:]  # 交互图的物品嵌入
            tqdm_dataloader = tqdm(dataloader)

            for iteration, batch in enumerate(tqdm_dataloader, start=1):
                user_idx, trainMask = batch  # 返回用户ID和其历史交互物品的掩码向量
                user_idx = user_idx.long().cuda()
                trainMask = trainMask.cuda()

                uiemb = uEmbeds[user_idx]  # 获取当前 batch 的交互图用户嵌入
                uuemb = uuEmbeds[user_idx]  # 获取当前 batch的社交图用户嵌入用于扩散模型输入

                # =========================================================
                # Stage-aware inference
                # =========================================================
                if self.epoch <= self.args.diff_epoch:
                    # Stage 1 & 2: NO diffusion in test
                    final_user = uiemb + uuemb
                else:
                    # Stage 3: use diffusion for denoising
                    user_social_predict = self.DiffProcess.p_sample(
                        self.SDNet, uuemb,
                        self.args.sampling_steps,
                        self.args.sampling_noise)
                    user_interact_predict = self.IDiffProcess.p_sample(
                        self.CDNet, uiemb,
                        self.args.sampling_steps,
                        self.args.sampling_noise)
                    final_user = uiemb + user_social_predict + user_interact_predict
                user = final_user

                # 预测评分矩阵：用户与所有物品的内积得分
                allPreds = torch.mm(user, torch.transpose(iEmbeds, 1, 0))
                # 屏蔽掉训练集中已交互过的物品（防止推荐重复）.trainMask有交互的位置为1
                allPreds = allPreds * (1 - trainMask) - trainMask * 1e8  # 将已交互的位置设为极小值
                #  从allPreds 中找出每个用户得分最高的前 args.topk 个物品（item），并返回它们的索引。[B,topk]
                _, topLocs = torch.topk(allPreds, self.args.topk)

                recall, ndcg = self.calcRes(topLocs.cpu().numpy(), dataloader.dataset.tstLocs, user_idx)
                Recall += recall
                NDCG += ndcg
            time_elapsed = time.time() - since
            print('Testing complete in {:.4f}s'.format(time_elapsed))
            # 计算测试集平均 Recall 和 NDCG
            Recall = Recall / num
            NDCG = NDCG / num
        return Recall, NDCG

    def calcRes(self, predicted, test_items, user_ids):
        # 推荐的物品的位置,每个用户对应的测试正样本物品位置列表（两重列表），用户id
        assert predicted.shape[0] == len(user_ids)  # B
        total_recall = total_ndcg = 0
        for i in range(len(user_ids)):  #对每个用户
            recommended = list(predicted[i])
            actual = test_items[user_ids[i]]
            num_interact = len(actual)  # 测试集中该用户实际交互的物品数量
            # 计算IDCG
            num_positions = min(num_interact, self.args.topk)  # 真实交互物品数量和 topk 中较小的那个，决定了最多能有多少个命中位置
            maxdcg = np.sum([1.0 / np.log2(loc + 2) for loc in range(num_positions)])
            recall = dcg = 0  # 初始化当前用户的 recall 和 dcg
            # 遍历该用户在测试集中的每一个真实交互物品
            for val in actual:
                if val in recommended:
                    recall += 1
                    dcg += 1.0 / (np.log2(recommended.index(val) + 2))  # dcg公式,真实交互的物品在推荐列表里的位置，按公式计算
            recall = recall / num_interact  # 除以用户真实交互物品总数，查全率
            ndcg = dcg / maxdcg  # 计算当前用户的 ndcg
            total_recall += recall
            total_ndcg += ndcg
        return total_recall, total_ndcg  # 返回当前 batch 的累计 Recall 和 NDCG

    def saveHistory(self):
        # 保存损失记录
        loss_history = {
            'total_loss': self.train_loss,
            'sum_loss':self.sum_loss,
            'bpr_loss': self.bpr_losses,
            'reg_loss': self.reg_losses,
            'uue_loss': self.uue_losses,
            'iue_loss': self.iue_losses,  # 交互图扩散损失
            'cl_loss': self.cl_losses,  # 对比学习损失
            'Recall': self.his_recall,
            'NDCG': self.his_ndcg
        }

        # 获取当前时间戳，用于生成唯一的文件名
        timestamp = datetime.now().strftime("%m%d_%H%M")
        desc = self.args.save_name
        fname = f'{timestamp}_{desc}.json'
        save_path = os.path.join('./logs', self.args.dataset, fname)
        with open(save_path, 'w') as fs:
            json.dump(loss_history, fs, indent=4)   # indent=4是为了美观易读

def grid_search():
    # 读取JSON文件中的结果
    def get_runned_combinations():
        with open("results/yelp_grid.json", "r") as f:
            data = json.load(f)

        runned = set()
        for result in data["results"]:
            # 按照param_space中参数的顺序创建元组
            combination = (
                result["steps"],
                result["noise_min"],
                result["noise_max"],
                result["noise_scale"],
                result["lr"],
                result["difflr"]
            )
            runned.add(combination)

        return runned
    # 获取已运行的参数组合
    runned = get_runned_combinations()
    # 定义参数空间
    param_space = {
        'steps': [50],             # 扩散步数
        'noise_min': [1e-3, 1e-2],        # 噪声下界
        'noise_max': [1e-3, 1e-2, 1e-1],        # 噪声上界
        'noise_scale':[10],          # 噪声规模
        'lr': [1e-4, 5e-4, 1e-3, 5e-3, 1e-2,5e-2],         # 学习率
        'difflr': [1e-5, 1e-4, 1e-3, 5e-3, 1e-2]      # 扩散模型学习率
    }
    # 生成所有参数组合
    param_combinations = list(itertools.product(
        param_space['steps'],
        param_space['noise_min'],
        param_space['noise_max'],
        param_space['noise_scale'],
        param_space['lr'],
        param_space['difflr']
    ))
    # 过滤非法组合：noise_min < noise_max
    param_combinations = [
        (steps, nmin, nmax, nscale, lr, difflr)
        for steps, nmin, nmax, nscale, lr, difflr in param_combinations
        if nmin < nmax
    ]
    # 过滤掉已经运行过的参数组合
    param_combinations = [
        combo for combo in param_combinations
        if combo not in runned
    ]

    results = []
    # 初始化数据（只做一次，避免重复 IO）
    handler = DataHandler()
    handler.LoadData()

    # 遍历参数组合
    for i, (steps, noise_min, noise_max, noise_scale, lr, difflr) in enumerate(param_combinations):
        print(
            f"\n正在试测组合             {i + 1}/{len(param_combinations)} | "
            f"steps={steps}, noise_min={noise_min}, noise_max={noise_max}, "
            f"noise_scale={noise_scale}, lr={lr}, difflr={difflr}"
        )
        # 固定随机种子steps=25, noise_min=0.0001, noise_max=0.001, noise_scale=0.1, lr=0.0001, difflr=1e-05
        fix_random_seed_as(args.seed)
        # 构造新的 args
        new_args = parse_args()
        new_args.steps = steps
        new_args.noise_min = noise_min
        new_args.noise_max = noise_max
        new_args.noise_scale = noise_scale
        new_args.lr = lr
        new_args.difflr = difflr

        # 创建Coach实例并训练
        app = Coach(handler, new_args)
        best_recall, best_ndcg = app.train()

        # 记录结果
        result= {
            'steps': steps,
            'noise_min': noise_min,
            'noise_max': noise_max,
            'noise_scale': noise_scale,
            'lr': lr,
            'difflr': difflr,
            'best_recall': best_recall,
            'best_ndcg': best_ndcg
        }
        results.append(result)
        # 定期保存中间结果
        if (i + 1) % 20 == 0:
            save_results(results)
        # 清理显存，否则可能会导致内存泄漏
        del app
        torch.cuda.empty_cache()

    # 保存最终结果
    save_results(results)
    # 找最优组合（Recall + NDCG）
    best_result = max(results, key=lambda x: x['best_recall'] + x['best_ndcg'])
    print("\n最佳参数组合:")
    print(json.dumps(best_result, indent=2))
    return results

def save_results(results):
    """保存搜索结果到JSON文件"""
    timestamp = datetime.now().strftime("%m%d_%H%M")
    save_dir = './results'
    os.makedirs(save_dir, exist_ok=True)

    filename = f'{args.dataset}_grid_{timestamp}.json'
    filepath = os.path.join(save_dir, filename)

    with open(filepath, 'w') as f:
        json.dump({
            'results': results,
            'best_result': max(results, key=lambda x: x['best_recall'])
        }, f, indent=2)

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda  # 设置可见的CUDA设备
    fix_random_seed_as(args.seed)  # 固定所有与深度学习实验相关的随机数种子

    handler = DataHandler()  # 创建数据处理器实例`handler`，用于加载和处理实验所需的数据集。
    handler.LoadData()  # 调用`handler`对象的`LoadData`方法，加载数据集，设置dataloader对象，创建社交图和用户物品交互图
    app = Coach(handler,args)  # Coach 类负责模型的训练逻辑以及与数据的交互。
    app.train()  # 调用训练函数

    # 网格搜索
    # grid_search()
