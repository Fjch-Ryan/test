import numpy as np
import torch as th

from .gaussian_diffusion import GaussianDiffusion


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper（DDIM：Denoising diffusion implicit models论文） is used, and only one section is allowed.
    备注：the DDIM paper（DDIM：Denoising diffusion implicit models论文）
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing    它有三种情况；数字列表或者一个逗号分隔符的字符串；之时每个部分的步数；作为一种特殊情况，使用“ddimN”，其中N是用若干个台阶跨步而来的。是用若干个台阶跨步而来的。
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):   #startsWith() 方法用于检测字符串是否以指定的子字符串开始。
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))    #set()集合函数将里面的数据类型转化为集合类型(不含相同元素)。
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]   #当section_counts是以逗号为分隔符的字符串时。
    size_per = num_timesteps // len(section_counts)   #每部分（per section）的时间步数 ,这里size_per默认是1000
    extra = num_timesteps % len(section_counts)   #余数 .extra默认为0
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):  #enumerate里面的section_counts为[1000];for里面的section_count为1000.
        size = size_per + (1 if i < extra else 0) #size默认为1000. 小括号里面的值表示将剩余时间步数单个的形式添加在前面的section，即前面的section额外加1个时间步数，直到增加的数量等与余数为止。
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)    #间隔(frac_stride):默认值为1；后面的时间步以间隔值大小进行采样到列表taken_steps(每个section的时间步)。
        cur_idx = 0.0   #当前索引
        taken_steps = []
        for _ in range(section_count):#循环1000次
            taken_steps.append(start_idx + round(cur_idx))    #round() 函数是 Python 中的一个内置函数，用于将浮点数近似为指定精度的小数或将浮点数近似为整数。
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):  #SpacedDiffusion类的出现是因为在IDPM中提出了一个space timestep，即对timespace进行一定的优化。否则可以直接使用GaussianDiffusion类
    """
    A diffusion process which can skip steps in a base diffusion process.
    一种可以跳过基本扩散过程的步骤（skip steps）的扩散过程。
    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):#init函数定义了加噪方案的β，timestep哪些时刻要保留，numstep加噪次数
        self.use_timesteps = set(use_timesteps)   #use_timesteps集合为[0,1,2,...,999]。
        self.timestep_map = []  #在经过下面的for循环后去，其默认值跟use_timesteps一样为[0,1,2,3,4,......,999]。
        self.original_num_steps = len(kwargs["betas"])  #betas为总扩散步数个β值的betas列表。这里返回值是总扩散步数。
        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []  #在经过下面这个for循环后得到的是跟use_timestep扩散步骤同步的β值列表。
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):  # base_diffusion.alphas_cumprod返回一个累积结果的alphas列表 其实就是alpha_ba的集合
            if i in self.use_timesteps:   #如果使用新的扩散步骤列表，则修改对应的betas列表的值。
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)  #将更新后的beata列表传入字典里面
        super().__init__(**kwargs)#将更新后的beatas传入到父类里面，这样以便在添加子类特有的功能之前，初始化父类中定义的属性。
        #super()用来调用父类(基类)的方法，__init__()是类的构造方法，super().__init__() 就是调用父类的__init__()方法， 同样可以使用super()去调用父类的其他方法。
    def p_mean_variance(  #p_mean_variance函数，p就是神经网络所预测的分布，故p_mean_variance就是神经网络预测的均值和方差，这里调用的是父类的方法super().
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses( #training_loss函数，根据传入的超参数不同得到不同目标函数的公式，最简单的就是MSE loss，我们也可以加上kl loss联合起来作为目标函数
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model2(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model): #wrap_model函数。对timestep进行后处理，比如对timestep进行scale，对timestep进行一定的优化。返回的是model.

        # isinstance()与type()区别：
        # type()  不会认为子类是一种父类类型，不考虑继承关系。
        # isinstance()  会认为子类是一种父类类型，考虑继承关系。

        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )
    def _wrap_model2(self, model):
        if isinstance(model, _WrappedModel2):
            return model
        return _WrappedModel2(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:  #对模型进行包裹，用于在传入model之前对timestep进行预处理。
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps


    def __call__(self, x, ts, **kwargs):
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]  #new_ts是一个标量，表示第一次扩散。

        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)  #对扩散步骤标量进行放缩
        return self.model(x, new_ts, **kwargs)#将放缩后的扩散步骤标量放入model参数里面



class _WrappedModel2:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, org, **kwargs):
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts,org, **kwargs)
