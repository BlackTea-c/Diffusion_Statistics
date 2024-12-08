


from typing import List

import torch
import torch.utils.data
import torchvision
from PIL import Image

from labml import lab, tracker, experiment, monit
from labml.configs import BaseConfigs, option
from labml_helpers.device import DeviceConfigs
from labml_nn.diffusion.ddpm import DenoiseDiffusion
from labml_nn.diffusion.ddpm.unet import UNet



def train(self):
   """
   单epoch训练DDPM
   """

   # 遍历每一个batch（monit是自定义类）
   for data in monit.iterate('Train', self.data_loader):
       # step数+1（tracker是自定义类）
       tracker.add_global_step()
       # 将这个batch的数据移动到GPU上
       data = data.to(self.device)

       # 每个batch开始时，梯度清0
       self.optimizer.zero_grad()
       # self.diffusion即为DenoiseModel实例，执行forward，计算loss
       loss = self.diffusion.loss(data)
       # 计算梯度
       loss.backward()
       # 更新
       self.optimizer.step()
       # 保存loss，用于后续可视化之类的操作
       tracker.save('loss', loss)

def sample(self):
    """
    利用当前模型，将一张随机高斯噪声(xt)逐步还原回x0,
    x0将用于评估模型效果（例如FID分数）
    """
    with torch.no_grad():
        # 随机抽取n_samples张纯高斯噪声
        x = torch.randn([self.n_samples, self.image_channels, self.image_size, self.image_size],
                            device=self.device)

        # 对每一张噪声，按照sample公式，还原回x0
        for t_ in monit.iterate('Sample', self.n_steps):
            t = self.n_steps - t_ - 1
            x = self.diffusion.p_sample(x, x.new_full((self.n_samples,), t, dtype=torch.long))

        # 保存x0
        tracker.save('sample', x)

def run(self):
    """
    train主函数
    """
    # 遍历每一个epoch
    for _ in monit.loop(self.epochs):
        # 训练模型
        self.train()
        # 利用当前训好的模型做sample，从xt还原x0，保存x0用于后续效果评估
        self.sample()
        # 再console上新起一行
        tracker.new_line()
        # 保存模型（experiment是自定义类）
        experiment.save_checkpoint()


#通过Config.run()执行Train与Sampling