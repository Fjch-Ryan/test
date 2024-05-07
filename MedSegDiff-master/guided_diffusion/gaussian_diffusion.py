"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py
Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""
from torch.autograd import Variable
import enum
import torch.nn.functional as F
from torchvision.utils import save_image
import torch
import math
import os
# from visdom import Visdom
# viz = Visdom(port=8850)
import numpy as np
import torch as th
import torch.nn as nn
from .train_util import visualize
from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood
from scipy import ndimage
from torchvision import transforms
from .utils import staple, dice_score, norm
import torchvision.utils as vutils
from .dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver
import string
import random

def standardize(img):
    mean = th.mean(img)
    std = th.std(img)
    img = (img - mean) / std
    return img


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.
    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()  #enum.auto() 枚举函数可自动为枚举成员分配值：从1开始
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB
    BCE_DICE = enum.auto()

    def is_vb(self):  #用于检查当前损失类型是否为变分下界相关的损失类型，即 KL 或 RESCALED_KL。
        return self == LossType.KL or self == LossType.RESCALED_KL   #self 是指 LossType 枚举类的实例；self == LossType.KL 表示当前实例是否等于 LossType.KL 枚举成员，即检查当前损失类型是否为 KL 散度相关的损失类型。


class GaussianDiffusion:  #ddim相关的函数是斯坦福提出的相关概念，对于扩散分割应用用不上。可以删除。
    """
    Utilities for training and sampling diffusion models.     用于训练和采样扩散模型的实用程序。
    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42
    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the    #如果为True，则将浮点时间步长传入模型，以便它们总是按比例缩放原始论文(0 ~ 1000)。
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        dpm_solver,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type  #model_mean_type，知道这个模型要预测什么，预测的是方差还是噪声还是x0，
        self.model_var_type = model_var_type  #model_var_type，方差是固定还是可学习的，还是预测学习线性加权的权重
        self.loss_type = loss_type  #loss_type，损失函数是预测mse还是加kl
        self.rescale_timesteps = rescale_timesteps  #rescale-timesteps，对时间进行scale，使得timestep永远缩放到在0到1000之间
        self.dpm_solver = dpm_solver

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)  #将betas数据类型转化为np的array类型 ，这样就能使用numpy里面的shape函数得到其形状。默认是一个线性列表。
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"  #只有betas是一维形状，程序才能运行下去。
        #numpy.all(a, axis=None, out=None, keepdims=np._NoValue)；测试沿给定轴的所有数组元素是否为True
        assert (betas > 0).all() and (betas <= 1).all()  #numpy.ndarray的list和int可以直接进行比较，在比较之前先将0复制成betas等长的n份，再进行元素比较，返回的这些布尔值是numpy.ndarray类型。

        self.num_timesteps = int(betas.shape[0])#self.num_timesteps为T，是一个标量。

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0) #返回一个累积结果的alphas列表
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])  #去掉了T时刻的alpha_ba值；并在列表最前端插入值：1. [1,self.alphas_cumprod[0],...,self.alphas_cumprod[-2]]
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)   #去掉了0时刻的alpha_ba值；并在列表末尾插入值：0。[self.alphas_cumprod[1],...,self.alphas_cumprod[-1],0.0]
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others  因为有现成的推导公式，可以从从x0直接推导出xt的概率分布。所以这部分可以不需要。
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)   #这里的log与误差函数有关。
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)  #知道 方差 和 均值 就能使用参数重整化技巧计算出x(t-1)的概率分布。
        self.posterior_variance = (         #后验概率分布的方差值，它是一个数组，[v0,v1,v2,...,v999],v表示方差，后面的数字表示扩散时刻(步)。
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(  #将self.posterior_variance第1个位置的元素代替第0个位置上的元素，防止第0个位置上的元素为0，所以截断一下。
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (  #在后验概率分布中，均值表达式中x0的系数；它也是一个数组[conef1_0,conef1_2,...,conef1_999]  ,conef1下划线(_)后的数字表示时间步t，因此可以通过索引取到当前时间步t对应的系数。
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = ( #在后验概率分布中，均值表达式中xt的系数；它也是一个数组[conef2_0,conef2_2,...,conef2_999]conef2下划线(_)后的数字表示时间步t，因此可以通过索引取到当前时间步t对应的系数。
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start   #_extract_into_tensor函数其实就是通过时间步标量t(也是索引)，从第一个参数数组里面提取该索引对应的元素。
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape) #_extract_into_tensor函数其实就是通过时间步标量t(也是索引)，从第一个参数数组里面提取该索引对应的元素。
        log_variance = _extract_into_tensor(   #_extract_into_tensor函数其实就是通过时间步标量t(也是索引)，从第一个参数数组里面提取该索引对应的元素。
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start    #_extract_into_tensor函数其实就是通过时间步标量t(也是索引)，从第一个参数数组里面提取该索引对应的元素。
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):  #扩散过程的x(t-1)在后验条件概率下的均值和方差
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start   #_extract_into_tensor函数其实就是通过时间步标量t(也是索引)，从第一个参数数组里面提取该索引对应的元素。
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)   #_extract_into_tensor函数其实就是通过时间步标量t(也是索引)，从第一个参数数组里面提取该索引对应的元素。
        posterior_log_variance_clipped = _extract_into_tensor(     #_extract_into_tensor函数其实就是通过时间步标量t(也是索引)，从第一个参数数组里面提取该索引对应的元素。
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    def p_mean_variance(  #p_mean_variance，p分布是神经网络的分布，去建模拟合的分布，得到前一时刻（逆扩散过程）的均值和方差，也包括x0的预测.
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}
        B, C = x.shape[:2]
        C=1
        cal = 0
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)
        if isinstance(model_output, tuple):
            model_output, cal = model_output
        x=x[:,-1:,...]  #loss is only calculated on the last channel, not on the input brain MR image
        #得到方差和对数方差
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            # 可学习的方差  ：有两种预测方法，分别是直接预测方差或者预测方差的范围两种预测方法。
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                #直接预测方差，这里取对数是因为方差非负，而模型的输出可正可负，所以只能预测方差的对数。
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                # 预测方差插值的系数
                #预测的范围是[-1,1]之间。
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            #不可学习的方差
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            #对x[0]进行一定的处理
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            # case 1:预测x[t-1]的期望值
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)  #在训练中用不上，但是后面的验证评估中可以用上
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                #case 2：预测x[0]的期望值
                pred_xstart = process_xstart(model_output)
            else:
                #case 3: 预测eps的期望值
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            'cal': cal,
        }



    def _predict_xstart_from_eps(self, x_t, t, eps):  #辅助函数，从预测处的噪声预测x0，其返回的是x0(x0是通过xt求出的。其实就是在x0直接预测xt的表达式写成：x0=式子的形式。  )
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev): #从xt-1中预测出x0  ？？？？？不确定
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):  #从x0和xt，推导eps(根据x0直接预测xt的公式求出eps。)
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:

            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, org, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.
        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        a, gradient = cond_fn(x, self._scale_timesteps(t),org,  **model_kwargs)


        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return a, new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t,  model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.
        See condition_mean() for details on cond_fn.
        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])

        eps = eps.detach() - (1 - alpha_bar).sqrt() *p_mean_var["update"]*0

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x.detach(), t.detach(), eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out, eps


    def sample_known(self, img, batch_size = 1):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop_known(model,(batch_size, channels, image_size, image_size), img)

    def p_sample(  #从xt采样出xt-1，所有的p分布都是模型预测的，其实就是推理的函数
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
    ):
        """
        基于x[t]采样出x[t-1]

        Sample x_{t-1} from the model at the given timestep.
        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        # 得到x[t-1]的均值、方差、对数方差、x[0]的预测值。
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x[:, -1:,...])

        #非零时刻的掩码矩阵，非零时刻值为1，o时刻值为0.
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        #0.5是为了得到标准差
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise

        return {"sample": sample, "pred_xstart": out["pred_xstart"], "cal": out["cal"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,

    ):
        """
        Generate samples from the model.
        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]


    def p_sample_loop_known(
        self,
        model,
        shape,
        img,
        step = 1000,
        org=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        conditioner = None,
        classifier=None
    ):
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        img = img.to(device)
        noise = th.randn_like(img[:, :1, ...]).to(device)
        x_noisy = torch.cat((img[:, :-1,  ...], noise), dim=1)  #add noise as the last channel
        img=img.to(device)

        if self.dpm_solver:
            final = {}
            noise_schedule = NoiseScheduleVP(schedule='discrete', betas= th.from_numpy(self.betas))

            model_fn = model_wrapper(
                model,
                noise_schedule,
                model_type="noise",  # or "x_start" or "v" or "score"
                model_kwargs=model_kwargs,
            )

            dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++",
                            correcting_x0_fn="dynamic_thresholding", img = img[:, :-1,  ...])

            ## Steps in [20, 30] can generate quite good samples.
            sample, cal = dpm_solver.sample(
                noise.to(dtype=th.float),
                steps= step,
                order=2,
                skip_type="time_uniform",
                method="multistep",
            )
            sample = sample.detach()    ### MODIFIED: for DPM-Solver OOM issue
            sample[:,-1,:,:] = norm(sample[:,-1,:,:])
            final["sample"] = sample
            final["cal"] = cal

            cal_out = torch.clamp(final["cal"] + 0.25 * final["sample"][:,-1,:,:].unsqueeze(1), 0, 1)
        else:
            print('no dpm-solver')
            i = 0
            letters = string.ascii_lowercase
            name = ''.join(random.choice(letters) for i in range(10)) 
            for sample in self.p_sample_loop_progressive(
                model,
                shape,
                time = step,
                noise=x_noisy,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                org=org,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
            ):
                final = sample
                # i += 1
                # '''vis each step sample'''
                # if i % 5 == 0:

                #     o1 = th.tensor(img)[:,0,:,:].unsqueeze(1)
                #     o2 = th.tensor(img)[:,1,:,:].unsqueeze(1)
                #     o3 = th.tensor(img)[:,2,:,:].unsqueeze(1)
                #     o4 = th.tensor(img)[:,3,:,:].unsqueeze(1)
                #     s = th.tensor(final["sample"])[:,-1,:,:].unsqueeze(1)
                #     tup = (o1/o1.max(),o2/o2.max(),o3/o3.max(),o4/o4.max(),s)
                #     compose = th.cat(tup,0)
                #     vutils.save_image(s, fp = os.path.join('../res_temp_norm_6000_100', name+str(i)+".jpg"), nrow = 1, padding = 10)

            if dice_score(final["sample"][:,-1,:,:].unsqueeze(1), final["cal"]) < 0.65:
                cal_out = torch.clamp(final["cal"] + 0.25 * final["sample"][:,-1,:,:].unsqueeze(1), 0, 1)
            else:
                cal_out = torch.clamp(final["cal"] * 0.5 + 0.5 * final["sample"][:,-1,:,:].unsqueeze(1), 0, 1)
            

        return final["sample"], x_noisy, img, final["cal"], cal_out

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        time=1000,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        org=None,
        model_kwargs=None,
        device=None,
        progress=False,
        ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """

        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        # 对时间进行倒序索引，从T到0
        indices = list(range(time))[::-1]
        org_c = img.size(1)
        org_MRI = img[:, :-1, ...]      #original brain MR image
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        else:
           for i in indices:
                t = th.tensor([i] * shape[0], device=device)
                # if i%100==0:
                    # print('sampling step', i)
                    # viz.image(visualize(img.cpu()[0, -1,...]), opts=dict(caption="sample"+ str(i) ))

                with th.no_grad(): #推理过程是不需要算梯度的。
                    # print('img bef size',img.size())
                    if img.size(1) != org_c:
                        img = torch.cat((org_MRI,img), dim=1)       #in every step, make sure to concatenate the original image to the sampled segmentation mask

                    out = self.p_sample(
                        model,
                        img.float(),
                        t,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        model_kwargs=model_kwargs,
                    )
                    yield out
                    img = out["sample"]

    def ddim_sample(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.
        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )


        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
                eta
                * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x[:, -1:, ...])

        mean_pred = (
                out["pred_xstart"] * th.sqrt(alpha_bar_prev)
                + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}


    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}



    def ddim_sample_loop_interpolation(
        self,
        model,
        shape,
        img1,
        img2,
        lambdaint,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        b = shape[0]
        t = th.randint(499,500, (b,), device=device).long().to(device)

        img1=torch.tensor(img1).to(device)
        img2 = torch.tensor(img2).to(device)

        noise = th.randn_like(img1).to(device)
        x_noisy1 = self.q_sample(x_start=img1, t=t, noise=noise).to(device)
        x_noisy2 = self.q_sample(x_start=img2, t=t, noise=noise).to(device)
        interpol=lambdaint*x_noisy1+(1-lambdaint)*x_noisy2

        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            time=t,
            noise=interpol,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"], interpol, img1, img2

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.
        Same usage as p_sample_loop().
        """
        final = None
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        b = shape[0]
        t = th.randint(99, 100, (b,), device=device).long().to(device)

        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            time=t,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):

            final = sample
       # viz.image(visualize(final["sample"].cpu()[0, ...]), opts=dict(caption="sample"+ str(10) ))
        return final["sample"]



    def ddim_sample_loop_known(
            self,
            model,
            shape,
            img,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            eta = 0.0
    ):
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        b = shape[0]

        img = img.to(device)

        t = th.randint(499,500, (b,), device=device).long().to(device)
        noise = th.randn_like(img[:, :1, ...]).to(device)

        x_noisy = torch.cat((img[:, :-1, ...], noise), dim=1).float()
        img = img.to(device)

        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            time=t,
            noise=x_noisy,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample

        return final["sample"], x_noisy, img


    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        time=1000,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.
        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(time-1))[::-1]
        orghigh = img[:, :-1, ...]


        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
                t = th.tensor([i] * shape[0], device=device)
                with th.no_grad():
                 if img.shape != (1, 5, 224, 224):
                     img = torch.cat((orghigh,img), dim=1).float()

                 out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                 )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(      #计算最终的kl散度。bpd：bit per domain  ，求平均再除以对数log.
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        #需要优化的KL散度
        """
        Get a term for the variational lower-bound.
        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.
        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        #真实的x[0]，x[t]和t去计算出x[t-1]的均值和方差
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        # x[t]、t和预测的x[0]去计算出x[t-1]的均值和方差
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        # kl散度包括两项，当t在0到t之间，用模型预测分布计算高斯分布算一个kl散度，另一项是最后一个时刻，L0 loss，使用的是似然函数，
        # 负对数似然函数，使用的是累积分布函数的差分拟合离散的高斯分布

        #kl是p_theta与q分布之间的KL散度。对应着L[t-1]损失函数
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)  #对kl取平均再除以log2.
        # 对应着L[0]损失函数。
        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        #t=0时刻，用离散的高斯分布去计算似然
        #t>0时刻，直接用KL散度。
        output = th.where((t == 0), decoder_nll, kl)   #decoder_nll表示LT，kl表示L(t-1),L0
        return {"output": output, "pred_xstart": out["pred_xstart"]}



    def training_losses_segmentation(self, model, classifier, x_start, t, model_kwargs=None, noise=None): #计算一个使用的loss
        """
        根据超参数可以选择多种损失函数。MSEloss、和kl loss进行组合。
        Compute training losses for a single timestep.
        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.      这里被传入的是micro，是一个小批次的融合张量（images+masks）
        :param t: a batch of timestep indices.       #t为一个含有micro_batchsize个随机时间的步索引序列
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start[:, -1:, ...]) #在通道维度中取最后一个通道。array[...] 就是array[:,:,:]

        #x_start==micro，是一个小批次的融合张量（images+masks）
        mask = x_start[:, -1:, ...]#在通道维度中取最后一个通道。array[...] 就是array[:,:,:]
        res = torch.where(mask > 0, 1, 0)   #merge all tumor classes into one to get a binary segmentation mask，res是一个分割图，里面只有两种值0和1.分别代表背景和前景。

        res_t = self.q_sample(res, t, noise=noise)     #add noise to the segmentation channel  ;返回值为经过t次扩散后的mask。
        x_t=x_start.float()
        x_t[:, -1:, ...]=res_t.float()  #x_t是原图和分割图以通道维度cat的张量。
        terms = {}


        if self.loss_type == LossType.MSE or self.loss_type == LossType.BCE_DICE or self.loss_type == LossType.RESCALED_MSE:

            model_output, cal = model(x_t, self._scale_timesteps(t), **model_kwargs)  ##t为一个含有micro_batchsize个随机时间的步索引序列
            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                C=1
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.使用变分界学习方差，但不要让它影响均值预测。
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=res,
                    x_t=res_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=res, x_t=res_t, t=t
                )[0],
                ModelMeanType.START_X: res,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]

            # model_output = (cal > 0.5) * (model_output >0.5) * model_output if 2. * (cal*model_output).sum() / (cal+model_output).sum() < 0.75 else model_output
            # terms["loss_diff"] = nn.BCELoss(model_output, target)
            terms["loss_diff"] = mean_flat((target - model_output) ** 2 )
            terms["loss_cal"] = mean_flat((res - cal) ** 2)
            # terms["loss_cal"] = nn.BCELoss()(cal.type(th.float), res.type(th.float)) 
            # terms["mse"] = (terms["mse_diff"] + terms["mse_cal"]) / 2.
            if "vb" in terms:
                terms["loss"] = terms["loss_diff"] + terms["vb"]
            else:
                terms["loss"] = terms["loss_diff"] 

        else:
            raise NotImplementedError(self.loss_type)

        return (terms, model_output)


    def _prior_bpd(self, x_start):#先验的kl散度（即L[T]），不影响模型的训练
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):#计算从T时刻到0时刻，所有的loss都计算出来，训练中不需要。可以用来做评估。
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.
        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)

            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bptimestepsd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):  # #_extract_into_tensor函数其实就是通过时间步标量t(也是索引)，从第一个参数数组arr里面提取该索引对应的元素。
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)  # torch.expand()函数是PyTorch中的一个张量变形函数，用于将一个张量沿着指定的维度进行扩展。在扩展过程中，指定的维度会被复制多次，从而增加了该维度上的大小。expand() 函数只能将size=1的维度扩展到更大的尺寸，如果扩展其他维度会报错。
