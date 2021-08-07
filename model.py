import torch.nn as nn


class simple_energy(nn.Module):
    "simpliest energy form"
    def __init__(self,in_dim,kernel_size,image_size):
        super(simple_energy,self).__init__()
        self.net = nn.Conv2d(in_channels = in_dim, out_channels = 1, kernel_size = kernel_size)
        output_size = image_size - kernel_size + 1
        self.fc = nn.Linear(output_size * output_size,1)



    def forward(self,x):
        m_batchsize ,C ,width ,height = x.size()
        x = self.fc(self.net(x).view(m_batchsize,-1))
        return x