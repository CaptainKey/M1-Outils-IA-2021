import torch 
import torch.nn as nn 

class conv_relu_pooling(nn.Module):
    def __init__(self,in_channel, nb_filtre, kernel,pooling_window=2,stride=2):
        super(conv_relu_pooling,self).__init__() 
        self.conv = nn.Conv2d(in_channel,nb_filtre,kernel) # [6, 3, 5, 5] [0,0,0,0]
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(pooling_window,stride)
    def forward(self,x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class dense_relu(nn.Module):
    def __init__(self,in_features, out_features, activation = False):
        super(dense_relu,self).__init__()
        self.dense = nn.Linear(in_features,out_features)        
        self.relu = nn.ReLU()
        self.activation = activation
    def forward(self,x):
        x = self.dense(x)
        if(self.activation):
            x = self.relu(x)
        return x

class LeNetBlock(nn.Module):
    def __init__(self):
        super(LeNetBlock,self).__init__() 
        self.features_extractor_1 = conv_relu_pooling(3,6,5)
        self.features_extractor_2 = conv_relu_pooling(6,16,5)
        
        self.prediction1 = dense_relu(400,120,activation=True)
        self.prediction2 = dense_relu(120,84,activation=True)
        self.prediction3 = dense_relu(84,10)
    def forward(self,x):
        x = self.features_extractor_1(x)
        x = self.features_extractor_2(x)

        x = x.view(-1,256)
        x = self.prediction1(x)
        x = self.prediction2(x)
        x = self.prediction3(x)
        return x

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__() 
        self.conv1 = nn.Conv2d(3,6,5) # [6, 3, 5, 5] [0,0,0,0]
        self.conv2 = nn.Conv2d(6,16,5) # [6, 3, 5, 5] [0,0,0,0]

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)
        self.dense1 = nn.Linear(400,120)
        self.dense2  = nn.Linear(120,84)
        self.dense3 =  nn.Linear(84,10)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)

        x = self.relu(x)
        x = self.pool(x)

        x = x.view(-1,400)

        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        x = self.dense3(x)

        return x 

if __name__ == '__main__':
    net = LeNet()
    print('net',net)
    netblock = LeNetBlock()
    print('netblock',netblock)
