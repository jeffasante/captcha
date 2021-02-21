import torch
from torch import nn, optim
import torch.nn.functional as F


class CRNN(nn.Module):
    
    def __init__(self, h, w, outputs, hidden_size=0, num_channels=3,
                num_layers=2):
        super(CRNN, self).__init__()
        
        self.conv1 = nn.Conv2d(num_channels, 512, kernel_size=3, padding=(1,1))
        self.maxpool1 = nn.MaxPool2d(2)
        self.batchnorm1 = nn.BatchNorm2d(512)
        
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=(1,1))
        self.maxpool2 = nn.MaxPool2d(2)
        self.batchnorm2 = nn.BatchNorm2d(256)
        
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=(1,1))
        self.maxpool3 = nn.MaxPool2d(2)
        self.batchnorm3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=(1,1))
        self.maxpool4 = nn.MaxPool2d(2)
        self.batchnorm4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=(1,1))
        self.maxpool5 = nn.MaxPool2d(2)
        self.batchnorm5 = nn.BatchNorm2d(64)
        
        self.dropout = nn.Dropout(p=0.2)
        
        self.head = nn.Linear(1152, 64)

        self.gru = nn.GRU(64, 32, num_layers,
                                batch_first=True,
                                bidirectional=True, dropout=0.25)

        self.output = nn.Linear(64, outputs+1)
        
    def forward(self, input):
        bs, c, h, w = input.size()
                
        x  = F.leaky_relu(self.conv1(input))
        x  = self.maxpool1(x)
        x  = F.leaky_relu(self.conv2(x))
        conv  = self.maxpool2(x) 
        conv =  conv.permute(0, 3, 1, 2) 
        conv = conv.view(bs, conv.size(1), -1)
        
        conv = conv.permute(1, 0, 2)
        
        x = self.head(conv)
        x = self.dropout(x)
        
        # gru features
        outs, _ = self.gru(x)
        
        return outs 
        
        
        
        
    
    