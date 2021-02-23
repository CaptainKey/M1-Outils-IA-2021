import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__() 
        self.conv1 = nn.Conv2d(1,6,5) # [6, 3, 5, 5] [0,0,0,0]
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(6,16,5) # [6, 3, 5, 5] [0,0,0,0]
        self.pool = nn.MaxPool2d(2,2)
        self.dense1 = nn.Linear(256,120)
        self.dense2  = nn.Linear(120,84)
        self.dense3 =  nn.Linear(84,10)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)

        x = self.relu(x)
        x = self.pool(x)

        x = x.view(-1,256)

        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)

        return x 

net = LeNet()
print('Avant chargement',net.conv1.bias)
# torch.save(net.state_dict(),'lenet_params.pt')
params = torch.load('lenet_params.pt')
net.load_state_dict(params)
print('Apres chargement',net.conv1.bias)
