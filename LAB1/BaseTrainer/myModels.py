import torch.nn as nn
import torch.nn.functional as F
import torch

# if you want to remove the gradients, an example
# for param in self.encoder.parameters():
#     param.requires_grad = False
# self.encoder.head = nn.Identity()
# self.head = nn.Linear(192,10)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers=[2048, 1024, 512, 1024, 512], output_dim=10):
        super().__init__()

        layers = [nn.Flatten(start_dim=1)]  # Flatten per trasformare l'input in un vettore

        prev_dim = input_dim  # Dimensione iniziale (numero di pixel)
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LeakyReLU())
            prev_dim = hidden_dim  # Aggiorna la dimensione per il prossimo strato

        layers.append(nn.Linear(prev_dim, output_dim))  # Ultimo layer senza attivazione
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MLPBaseBlock(nn.Module):
    def __init__(self, in_features, out_feature=None):
        """
        Parametri:
        - in_features: dimensione attesa (e prodotta) dall'output del blocco.
        - out_feature: dimensione dell'input ricevuto dal blocco.
                        Se None, si assume uguale a in_features.
        Il blocco esegue:
            1) Linear(in_features, in_features//2) -> ReLU
            2) Linear(in_features//2, in_features//2) -> ReLU
            3) Linear(in_features//2, in_features)
        Dopo, viene sommata la skip connection (con proiezione se necessario) e applicata una ReLU.
        """
        super().__init__()
        if out_feature is None:
            out_feature = in_features

        mid_features = in_features // 2
        
        self.block = nn.Sequential(
            nn.Linear(in_features, mid_features),
            nn.ReLU(),
            nn.Linear(mid_features, mid_features),
            nn.ReLU(),
            nn.Linear(mid_features, out_feature)
        )

        if out_feature != in_features:
            self.skip = nn.Linear(in_features, out_feature)
        else:
            self.skip = nn.Identity()
            
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.skip(x)
        out = self.block(x)
        out = out + residual
        return self.relu(out)   #Da controllare cosa fa la relu

class ResidualMLP(nn.Module):
    def __init__(self, inFeatures, block_dims = [2048, 1024, 1024, 512, 1024], dim_output = 10):
        """
        Parametri:
        - block_dims: lista di dimensioni; per esempio, [1024, 2048, 1024]
                        crea tanti blocchi quanti gli elementi nella lista.
                        Il blocco i-esimo attende in input:
                           * per il primo blocco: out_feature (se fornito) o block_dims[0]
                           * per gli altri: block_dims[i-1] (output del blocco precedente)
                        e produce in output dimensione block_dims[i].
        - dim_output: dimensione dell'output finale (head della rete).
        - out_feature: dimensione dell'input iniziale; se None si assume uguale a block_dims[0].
        """
        super().__init__()
        layers = []
        layers.append(nn.Flatten(start_dim=1))
        layers.append(MLPBaseBlock(in_features=inFeatures, out_feature=block_dims[0]))
        
        for i in range(len(block_dims) - 1):
            layers.append(MLPBaseBlock(in_features=block_dims[i], out_feature=block_dims[i+1]))

        self.blocks = nn.Sequential(*layers)
        self.head = nn.Linear(block_dims[-1], dim_output)

    def forward(self, x):
        x = self.blocks(x)
        x = self.head(x)
        return x
    
class Mornet_light(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, padding='same'),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, 1, padding='same'),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.3),

            nn.Conv2d(64, 128, 3, 1, padding='same'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 3, 1, padding='same'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.3),

            nn.Conv2d(128, 256, 3, 1, padding='same'),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 3, 1, padding='same'),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.3),

            nn.Conv2d(256, 512, 3, 1, padding='same'),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, 3, 1, padding='same'),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.3)
        )
        self.mlp = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(2 * 2 * 512, 1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 10),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.mlp(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_features ,out_features):
        super().__init__()

        #First convolutional block
        self.conv1 = nn.Conv2d(in_channels= in_features , out_channels=out_features, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(inplace=True)

        #Second convolutional block
        self.conv2 = nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_features)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        return x

class CNNresidual(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()
        #Lets use the old ConvBlock and add the residual connection with a shortcut layer

        self.layer = nn.Conv2d(3,64, kernel_size=1, padding=1)
        self.cb1 = ConvBlock(64, 64)
        self.cb2 = ConvBlock(64, 64)
        self.cb3 = ConvBlock(64, 64)
        self.cb4 = ConvBlock(64, 64)

        self.fc1 = nn.Linear(64 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.sc1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=1, stride = 1), nn.BatchNorm2d(64))
        self.sc2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, stride = 1), nn.BatchNorm2d(128))
        self.sc3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, stride = 1), nn.BatchNorm2d(256))
        self.sc4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, stride = 1), nn.BatchNorm2d(512))

        self.maxpool = nn.AvgPool2d(kernel_size=2, stride=2)
        #TODO: provare ad aggiungere un drop out tra il layer FC magari del 0.3

    def forward(self, x):
        #identity = self.sc1(x)
        x = self.layer(x)
        identity = x
        x = self.cb1(x)
        x = F.relu(x + identity)
        x = self.maxpool(x)

        #identity = self.sc2(x)
        identity = x
        x = self.cb2(x)
        x = F.relu(x + identity)
        x = self.maxpool(x)

        #identity = self.sc3(x)
        identity = x
        x = self.cb3(x)
        x = F.relu(x + identity)
        x = self.maxpool(x)

        #identity = self.sc4(x)
        identity = x
        x = self.cb4(x)
        x = F.relu(x + identity)
        x = self.maxpool(x)

        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x
    
class CNNresidualFull(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()

        #Lets use the old ConvBlock and add the residual connection with a shortcut layer

        self.cb1 = ConvBlock(3, 64)
        self.cb2 = ConvBlock(64, 128)
        self.cb3 = ConvBlock(128, 256)
        self.cb4 = ConvBlock(256, 512)

        self.fc1 = nn.Linear(512 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.sc1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=1, stride = 1), nn.BatchNorm2d(64))
        self.sc2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, stride = 1), nn.BatchNorm2d(128))
        self.sc3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, stride = 1), nn.BatchNorm2d(256))
        self.sc4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, stride = 1), nn.BatchNorm2d(512))

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        #TODO: provare ad aggiungere un drop out tra il layer FC magari del 0.3

    def forward(self, x):
        identity = self.sc1(x)
        x = self.cb1(x)
        x = F.relu(x + identity)
        x = self.maxpool(x)

        identity = self.sc2(x)
        x = self.cb2(x)
        x = F.relu(x + identity)
        x = self.maxpool(x)

        identity = self.sc3(x)
        x = self.cb3(x)
        x = F.relu(x + identity)
        x = self.maxpool(x)

        identity = self.sc4(x)
        x = self.cb4(x)
        x = F.relu(x + identity)
        x = self.maxpool(x)

        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x