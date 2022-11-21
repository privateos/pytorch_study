
#1)CNN图片分类
#2) Precision/Recall xxxx
#3)可视化图片


import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict
from sklearn.datasets import load_digits
from torch.utils.data import Dataset,DataLoader



class CNN_classfication(nn.Module):
    def __init__(self,out_channels,kernel_size,pool_size) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.cnn = nn.Sequential(
            #(batch_size, in_channels, H, W)
            nn.Conv2d(1,out_channels,kernel_size,padding='same', dilation=1),
            #(batch_size, out_channels, H_out, W_out)
            #x.shape=(batch_size,out_channels,8,8)
            nn.ReLU(),

            #point-wise 
            # nn.Conv2d(out_channels, out_channels//3, kernel_size=1),
            # nn.ReLU(),
            nn.Conv2d(out_channels, out_channels*4, 
                    kernel_size=kernel_size, padding='same',
                    groups=3),
            nn.ReLU(),

            #depth-wise
            nn.MaxPool2d((pool_size,pool_size),stride=(pool_size,pool_size))
            #x.shape=(batch_size,out_channels,8//pool_size,8//pool_size)
        )
        self.fc = nn.Linear(64//pool_size//pool_size*out_channels//3,10)

    def forward(self,x):
        # print(x.shape);exit()
        batch_size = x.shape[0]
        x = self.cnn(x)
        x = x.reshape(batch_size,-1)
      #  print(x.shape);exit()
        return self.fc(x)

class SELayer(nn.Module):
    def __init__(self, channels: int) -> None:
        super(SELayer, self).__init__()
        self.seq = nn.Sequential(
            #(b, channels, 1, 1)
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
            #(b, channel, 1, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, channel, H, W = x.size()
        x1 = torch.max_pool2d(x, kernel_size=(H, W))
        x2 = self.seq(x1)
        return x*x2

digits = load_digits()
x,y = digits.data,digits.target
print(x.shape)#;exit()
x = x/16.0


class MyDataset(Dataset):
    def __init__(self,x,y) -> None:
        super(MyDataset,self).__init__()
        self.x = torch.tensor(x).reshape(-1,1,8,8).float()
        self.y = torch.tensor(y).long()#.reshape(-1,1)

    def __getitem__(self, index):
        # return self.x[index],torch.tensor(y[index])
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]

out_channels,kernel_size,pool_size = 6,3,4
model = CNN_classfication(out_channels,kernel_size,pool_size)

dataset = MyDataset(x,y)
dataloader = DataLoader(dataset=dataset,batch_size=32,shuffle=True)
loss_fn = nn.CrossEntropyLoss()
lr = 0.0008
optim = torch.optim.Adam(model.parameters(),lr=lr)
epochs = 100
for epoch in range(epochs):
    print(f'epoch={epoch}')
    for batch_x,batch_y in dataloader:
        output = model(batch_x)
        loss = loss_fn(output,batch_y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        pre=torch.argmax(output,1)
        acc = torch.sum(pre==batch_y).item()
        print(f'loss={loss.item()}')
        print(f'acc={float(acc)/batch_x.size(0)}')






#(b, m*n, H, W)#groups=m
#(b, m, n, H, W)
#(b, n, m, H, W)
#(b, n*m, H, W)#groups =n

x#(b, m, H, W)
conv1 = nn.Conv2d(m*n, m*n, kernel_size, padding='same', groups=m)
x1 = relu(conv1(x))#(b, m*n, H, W)
conv2 = nn.Conv2d(m*n, m*n, kernel_size, xx, groups=n)
x2 = torch.transpose(x1, 0, 1)

x3 = conv2(x2)




x#(b, C1, H, W)
x1 = torch.max_pool2d(x, kernel_size=(H, W))#(b, C1, 1, 1)

conv1 = nn.Conv2d(C1, C1, kernel_size=1)
# torch.conv2d(x1, conv1.weight, bias=conv1.bias)
x2 = torch.sigmoid(conv1(x1))#(b, C1, 1, 1)

y = x*x2