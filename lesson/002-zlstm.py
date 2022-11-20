import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict
from sklearn.datasets import load_digits
from torch.utils.data import Dataset,DataLoader

class Sigmoid(nn.Module):
    def __init__(self) -> None:
        super(Sigmoid,self).__init__()

    def forward(self,x):
        return 1.0/(1.0+torch.exp(-x))

class Gate(nn.Module):
    def __init__(self,input_size,output_size) -> None:
        super(Gate,self).__init__()
        self.fc = nn.Linear(input_size+output_size,output_size)

    def forward(self,x,h):
        #x.shape = (batch_size,input_size )
        #h.shape = (batch_size,output_size )
        x_h = torch.concat([x,h],dim=1)
        #x_h.shape = (batch_size,input_size+output_size )
        return self.fc(x_h)

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from base.zlstm import ZLSTM

class Model(nn.Module):
    def __init__(self,input_size,hid_size,n_classs) -> None:
        super(Model,self).__init__()
        # self.lstm = nn.LSTM(input_size,hid_size,batch_first=True)
        self.lstm = ZLSTM(input_size,hid_size)

        self.fc = nn.Linear(hid_size,n_classs)

    def forward(self,x):
        output, (h_n, c_n) = self.lstm(x)
        # return self.fc(torch.squeeze(h_n,0))
        return self.fc(output[:,-1,:])
# batch_size,seq_len,input_size,output_size = 5,6,7,8
# input = torch.randn(batch_size,seq_len,input_size)
# mylstm = Lstm(input_size,output_size)
# # lstm = nn.LSTM(input_size,output_size)
# # online_output, (online_h_t_1, online_c_t_1) = lstm(input)
# output, (h_t_1, c_t_1) = mylstm(input)
# print(output.shape)
# print(output[:,-1,:],h_t_1)
# print("--------------")
# # print(online_output.shape)
# # print(online_output[:,-1,:],online_h_t_1)


# xxx = torch.randn(2,3)
# mysigmoid = Sigmoid()
# y1 = mysigmoid(xxx)
# y2 = torch.sigmoid(xxx)
# print(torch.abs(y1-y2))

digits = load_digits()
x,y = digits.data,digits.target
x = x/16.0
# print(np.max(x),np.min(x))
# exit()
# print(type(x))

class MyDataset(Dataset):
    def __init__(self,x,y) -> None:
        super(MyDataset,self).__init__()
        self.x = torch.tensor(x).reshape(-1,8,8).float()
        self.y = torch.tensor(y).long()#.reshape(-1,1)
        print('dataset', self.x.shape, self.y.shape)

    def __getitem__(self, index):
        # return self.x[index],torch.tensor(y[index])
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]

seq_len , hid_size, n_class = 8,16,10
#model = Lstm(seq_len , hid_size)
model = Model(seq_len , hid_size, n_class)

# model = torch.nn.LSTM(8, hid_size, batch_first=True)
dataset = MyDataset(x,y)
dataloader = DataLoader(dataset=dataset,batch_size=32,shuffle=True)
loss_fn = nn.CrossEntropyLoss()
lr = 0.0008
optim = torch.optim.Adam(model.parameters(),lr=lr)
epochs = 100
for epoch in range(epochs):
    print(f'epoch={epoch}')
    for batch_x,batch_y in dataloader:
        #print(x.shape,y.shape, x.dtype, y.dtype)

        #output, (h_t_1, c_t_1)  = model(batch_x)
        output = model(batch_x)
        loss = loss_fn(output,batch_y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        pre=torch.argmax(output,1)
      #  print(pre,batch_y)
        acc = torch.sum(pre==batch_y).item()
        print(f'loss={loss.item()}')
        print(f'acc={float(acc)/batch_x.size(0)}')



