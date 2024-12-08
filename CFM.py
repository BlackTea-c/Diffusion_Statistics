import os
import matplotlib.pyplot as plt
import torch
import torchdiffeq
import torchsde
from torchdyn.core import NeuralODE
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from tqdm import tqdm

#主要模块
from torchcfm.conditional_flow_matching import *
from torchcfm.models.unet import UNetModel

#循环主逻辑：
sigma = 0.0
model = UNetModel(
    dim=(1, 28, 28), num_channels=32, num_res_blocks=1, num_classes=10, class_cond=True
).to(device)
optimizer = torch.optim.Adam(model.parameters())
FM = ConditionalFlowMatcher(sigma=sigma)
# Users can try target FM by changing the above line by
# FM = TargetConditionalFlowMatcher(sigma=sigma)
node = NeuralODE(model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)

for epoch in range(n_epochs):
    for i, data in enumerate(train_loader):

        optimizer.zero_grad()

        x1 = data[0].to(device) #真实图像
        y = data[1].to(device)  #标签
        x0 = torch.randn_like(x1) #噪声

        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1) #sample t时刻  拿到向量场


        vt = model(t, xt, y) #预测向量场

        loss = torch.mean((vt - ut) ** 2)
        loss.backward()
        optimizer.step()
        print(f"epoch: {epoch}, steps: {i}, loss: {loss.item():.4}", end="\r")



