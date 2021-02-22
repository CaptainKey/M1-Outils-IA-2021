import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__() 
        self.conv1 = nn.Conv2d(3,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
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
    model = LeNet()

    pre_processing = transforms.Compose([
        transforms.ToTensor()
    ])

    # Chargement des donnees d entrainement
    set_entrainement = torchvision.datasets.CIFAR10(root='./cifar10',train=True,download=True,transform=pre_processing)
    # Chargement des donnees de tests
    set_test = torchvision.datasets.CIFAR10(root='./cifar10',train=False,download=True,transform=pre_processing)


    load_entrainement = torch.utils.data.DataLoader(set_entrainement,batch_size=32,num_workers=2)
    load_test = torch.utils.data.DataLoader(set_test,batch_size=32,num_workers=2)

    fonction_erreur = nn.CrossEntropyLoss()

    opti = torch.optim.Adam(model.parameters(),lr=0.01)

    # classes = ['airplaine','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    epochs = 1
    for epoch in range(epochs):
        avg_loss = 0
        for i, data in enumerate(load_entrainement):
            imgs, labels = data 

            opti.zero_grad()

            out = model(imgs)
            
            loss = fonction_erreur(out,labels)

            loss.backward()

            avg_loss += loss.item()
            opti.step()

            if i % 500 == 499:
                print('avg loss {}'.format(avg_loss/500))

        ok = 0
        total = 0
        print('TEST EPOCH 1')
        with torch.no_grad():
            for j,data in enumerate(load_test):
                imgs, labels = data 
                outputs = model(imgs)
                _, prediction = torch.max(outputs,1)
                total += imgs.size(0)
                ok += (prediction == labels).sum().item()
            print('Performance : {} %'.format((ok/total)*100))
"""

    Chat ou Chien 

    out = [Chien, Chat]
    out = [0.7, 0.2]

    img 1 => label = Chat => [0,1]

    out = [0.7,0.2]
    label =  [0,1 ]

"""