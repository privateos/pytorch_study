import torch
import torch.nn as nn
from typing import Tuple
class Liner_concat(nn.Module):
    def __init__(self,input_size: int,output_size: int) -> None:
        super(Liner_concat,self).__init__()
        self.output_size = output_size
        self.fc = nn.Linear(input_size+output_size,output_size*4)
        self.split_list = [output_size]*4

    def forward(self,x: torch.Tensor,h: torch.Tensor) -> Tuple:
        #x.shape = (batch_size,input_size )
        #h.shape = (batch_size,output_size )
        x_h = torch.concat([x,h],dim=1)
        #x_h.shape = (batch_size,input_size+output_size )
        x_h_concat = self.fc(x_h)
        #x_h_concat.shape = (batch_size,output_size*4)
        return torch.split(x_h_concat,self.split_list,dim=1)

class LSTMCell(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.liner_concat = Liner_concat(input_size,output_size)

    def forward(self, x_t: torch.Tensor, hc_t_1: Tuple[torch.Tensor, torch.Tensor])\
        -> Tuple[torch.Tensor, torch.Tensor]:
        h_t_1, c_t_1 = hc_t_1
        i_t, f_t, o_t, c_hat_t = self.liner_concat(x_t, h_t_1)
        i_t = torch.sigmoid(i_t)
        f_t = torch.sigmoid(f_t)
        o_t = torch.sigmoid(o_t)
        c_hat_t = torch.tanh(c_hat_t)

        c_t = f_t * c_t_1 + i_t * c_hat_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t

class LSTM(nn.Module):
    def __init__(self,input_size: int, output_size: int) -> None:
        super(LSTM,self).__init__()
        self.cell = LSTMCell(input_size, output_size)

    def forward(self, input: torch.Tensor) \
        -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size,seq_len, d_model = input.size()
        device = input.device
        dtype = input.dtype
        cell = self.cell
        h_t_1 = torch.zeros(batch_size,cell.output_size,device=device,dtype=dtype)
        c_t_1 = torch.zeros(batch_size,cell.output_size,device=device,dtype=dtype)
        

        output = []

        for t in range(seq_len):
            x_t = input[:,t,:]
            hc_t_1: Tuple[torch.Tensor, torch.Tensor] = cell(x_t, (h_t_1, c_t_1))#__call__
            h_t_1, c_t_1 = hc_t_1
            


            output.append(h_t_1)

        output = torch.stack(output,dim=1)
        return output, (h_t_1, c_t_1)
