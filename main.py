import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

# Intialisation du modèle 
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__() 
        self.conv1 = nn.Conv2d(1,6,5) # [6, 3, 5, 5] [0,0,0,0]
        self.conv2 = nn.Conv2d(6,16,5) # [6, 3, 5, 5] [0,0,0,0]

        self.relu = nn.ReLU()
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

if __name__ == '__main__':
    # Definition des pretraitements 
    pre_processing = transforms.Compose([
        transforms.ToTensor()
    ])

    # Chargement des donnees d entrainement
    set_entrainement = torchvision.datasets.MNIST(root='./cifar10',train=True,download=True,transform=pre_processing)
    # Chargement des donnees de tests
    set_test = torchvision.datasets.MNIST(root='./cifar10',train=False,download=True,transform=pre_processing)

    # DataLoader 
    loader_entrainement = torch.utils.data.DataLoader(set_entrainement,batch_size=4,num_workers=2)
    loader_test = torch.utils.data.DataLoader(set_test,batch_size=4,num_workers=2)


    # Sélection de l'unité de calcul
    if torch.cuda.is_available():
        device = torch.device("cuda:0") 
    else:
        device = torch.device("cpu")

    # Création de l'instance de classe du modèle
    model = LeNet()
    # Mise en place du model sur GPU
    model.to(device)
        

    # Definition du calcul de l'erreur/perte
    error = nn.CrossEntropyLoss()

    # Définition de la technique d'optimisation pour la mise à jour de poids
    optimisation_gd= torch.optim.Adam(model.parameters(),lr=0.001)

    """
        W 

        W_(t+1)

        error = nn.CrossEntropyLoss()
        loss = error(out,labels)

        loss.backward()

        d(loss) / dW = gradient 

        W_(t+1) = W_t - lr * d(loss) / dW 

    """

    epochs = 1
    for epoch in range(epochs):
        avg_loss = 0
        for i, data in enumerate(loader_entrainement):
            # Récupération des images et labels
            imgs, labels = data 
            # [BATCH,CHANNEL,HEIGHT,WIDTH]
            # Mise en place des données sur le GPU
            imgs = imgs.to(device)
            labels = labels.to(device)

            # Mise à Zéro des gradients
            optimisation_gd.zero_grad()

            # Propagation des images dans le réseau
            out = model(imgs)
            
            # Calcul de la perte
            loss = error(out,labels)

            # Ajout de la perte courante à l'accumulateur pour visuel
            avg_loss += loss.item()

            # Rétropropagation
            loss.backward()

            # Mise à jour des poids
            optimisation_gd.step()

            if i%500== 499:
                # Affichage de la perte
                print('avg loss {}'.format(avg_loss/500))
                avg_loss = 0
        ok = 0
        total = 0
        # Désactivation du calcul des gradients
        with torch.no_grad():
            for j,data in enumerate(loader_test):
                # Récupération des images et labels
                imgs, labels = data 

                # Mise en place des données sur le GPU
                imgs = imgs.to(device)
                labels = labels.to(device)

                # Propagation des images dans le réseau
                outputs = model(imgs)
                # Récupération de la prédiction
                _, prediction = torch.max(outputs,1)

                # Ajout du nombre d'images calculé
                total += imgs.size(0)
                # [BATCH,CHANNEL,HEIGHT,WIDTH]

                # Compte de bonnes prédictions
                ok += (prediction == labels).sum().item()
            # Affichage résultat
            print('Performance  Epoch {} : {} %'.format(epoch,(ok/total)*100))
