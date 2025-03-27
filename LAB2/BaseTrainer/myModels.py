import torch.nn as nn
import torch.nn.functional as F
import torch

# Se vuoi togliere i gradienti, un esempio
# for param in self.encoder.parameters():
#     param.requires_grad = False
# self.encoder.head = nn.Identity()
# self.head = nn.Linear(192,10)

#Esempio di sequential con vari moduli che potrebbero fare comodo
# self.classic = nn.Sequential(
#     nn.Conv2d(3, 64, 3, 1, padding='same'),
#     nn.BatchNorm2d(64),
#     nn.MaxPool2d(2, 2),
#     nn.Flatten(start_dim=1),
#     nn.Linear(2048, 1024),
#     nn.LeakyReLU(),
#     nn.Dropout(p=0.3),                #Ricordati che i dropout si cerca di utilizzarli solo all'ultimo!! quando ormai hai raggiunto già buoni risultati
#     nn.Linear(1024, 10),
#     nn.LeakyReLU()
# )

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        

    def forward(self, x):
        # RICORDA: Se togli i grandienti devi metterlo altrimenti te li calcola per poi buttarli via....
        # with torch.no_grad():
        #     x = self.encoder(x)
        return x