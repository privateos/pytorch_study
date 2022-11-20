import torch
import torch.nn as nn
import numpy as np
  

class X_H_trans(nn.Module):
    def __init__(self,input_size:int,output_size:int) -> None:
        super(X_H_trans,self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc = nn.Linear(input_size+output_size,output_size*3)
        self.list = [output_size] * 3
    def forward(self,x,h):
        x_h = torch.concat([x,h],dim=1)
        #x_h.shape = (batch_size,input_size+output_size)
        return torch.split(self.fc(x_h),self.list,dim=1)

class Zigmoid(nn.Module):
    def __init__(self,b:float) -> None:
        super(Zigmoid,self).__init__()
        self.b = b

    def forward(self,x:torch.Tensor):
        # return torch.sigmoid(torch.abs(torch.exp(self.b*x)-1))
        cond = x >= 0.0
        ebx = torch.exp(self.b*x)

        cond_1 = ebx - 1.0
        cond_2 = 1.0 - 1.0/ebx
        result = torch.where(cond, cond_1, cond_2)
        return torch.sigmoid(result)


class ZLSTMCell(nn.Module):
    def __init__(self,input_size:int,output_size:int) -> None:
        super(ZLSTMCell,self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.x_h_trans = X_H_trans(input_size,output_size)
        self.zigmoid = Zigmoid(1.0)

    def forward(self,x,hc_t_1) -> tuple:
        h_t_1,c_t_1 = hc_t_1
        #x.shape = (batch_size,input_size)
        #h_1.shape = (batch_size,output_size)
        # output = torch.zeros(batch_size,seq_len,self.output_size, dtype=dtype, device=device)
        
        o_t,c_t_hat,f_t = self.x_h_trans(x,h_t_1)
        o_t = torch.sigmoid(o_t)
        c_t_hat = torch.tanh(c_t_hat)
        f_t = self.zigmoid(f_t)

        c_t = f_t * c_t_1 +(1.0 - f_t)*c_t_hat

        h_t_1 = o_t*torch.tanh(c_t)
        c_t_1 = c_t
        return h_t_1, c_t_1


class ZLSTM(nn.Module):
    def __init__(self,input_size:int,output_size:int) -> None:
        super(ZLSTM,self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.x_h_trans = X_H_trans(input_size,output_size)
        self.zigmoid = Zigmoid(1.0)
        self.zlstmcell = ZLSTMCell(input_size,output_size)

    def forward(self,input:torch.Tensor):
        batch_size,seq_len,d_model = input.size()

        dtype = input.dtype
        device = input.device
        # h_0,c_0 = h_0_c_0
        h_0 = torch.zeros(batch_size,self.output_size, dtype=dtype, device=device)
        c_0 = torch.zeros(batch_size,self.output_size, dtype=dtype, device=device)
        h_t_1 = h_0
        c_t_1 = c_0
        #x.shape = (batch_size,input_size)
        #h_1.shape = (batch_size,output_size)
        # output = torch.zeros(batch_size,seq_len,self.output_size, dtype=dtype, device=device)
        h_list = []
        for t in range(seq_len):
            x = input[:,t,:]
            h_t_1, c_t_1 = self.zlstmcell(x,(h_t_1,c_t_1))
            h_list.append(h_t_1)
        
        #h_list = [ .shape = (b, o),...]
        #H.shape = (b, s, 0)
        H = torch.stack(h_list, dim=1)
        return H, (h_t_1, c_t_1) 
