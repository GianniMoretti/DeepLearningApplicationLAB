import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F

from colorama import Fore, Style, init
from datetime import datetime
import wandb
import uuid

from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, StepLR, ReduceLROnPlateau, OneCycleLR, CyclicLR

import baseModelTrainer as bMT
import myModels
import os

init(autoreset=True)

########################################## PHASE 0: CONFIGURATION FILE #####################################################
config_file_name = "config_EX_1_1.yaml"

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the absolute path of the file, e.g., if the file is named "data.txt" 
# and is located in the same folder as the script:
config_file_path = os.path.join(script_dir, "config_files/" + config_file_name)

# Get parameters from options
configr = bMT.load_yaml_config(config_file_path)

########################################## PHASE 0.1: RUN NAME and LOGGER #########################################################
# Create an ID for each run
runID = str(uuid.uuid4())[:8]
configr["runID"] = runID

# Create the path for saving checkpoints 
configr["checkpoint_path"] = os.path.join(script_dir, "checkpoint/" + runID)

current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_name = str(runID)+ "_" + str(configr['model_name']) + "_" + \
    str(configr['optim_name']) + "_lr" + str(configr['learning_rate']) + "_bs" + \
    str(configr['batch_size']) + "_" + current_time

#logger = TrainLogger(run_name, run_name)

# Configuration for wandb
print(Fore.CYAN + Style.BRIGHT + "\nWANDB:\n-------------------------------------")
if configr["use_wandb"]:
    wandb.login()
    if configr["wand_project_name"]:
        if run_name:
            wandb.init(project=configr["wand_project_name"],
                        name=run_name,
                        config=configr)
        else:
            wandb.init(project=configr["wand_project_name"],
                        name=configr["model_name"],
                        config=configr)
    else:
        print(Fore.RED + Style.BRIGHT + "Wand Error: Insert the wand_project_name!!")

print(Fore.CYAN + Style.BRIGHT + "\nSTARTING RUN: " + Fore.GREEN + runID)

########################################## PHASE 0.2: SEED ####################################################
if configr['seed'] > 0:
    bMT.set_seed(configr['seed'])
    print(Fore.CYAN + Style.BRIGHT + "\nSEED: " + Fore.GREEN + configr['seed'])

print(Fore.CYAN + Style.BRIGHT + "\nCHECKPOINT PATH: " + Fore.GREEN + configr['checkpoint_path'])

########################################## PHASE 1: MODEL SELECTION #########################################################
# Create model
model = myModels.MLP_mnist()

print(Fore.CYAN + Style.BRIGHT + "\nNETWORK: " + Fore.GREEN + configr['model_name'])
print(Fore.CYAN + Style.BRIGHT + "----------------------------------------")

if configr['print_model']: print(model.modules)
nParam = sum(p.numel() for p in model.parameters())
print("\nNumber of parameters: " + Fore.GREEN + f"{nParam} parameters")
configr["num_param"] = nParam

########################################## PHASE 2: DEVICE SELECTION #########################################################
# Setting up CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Model running on: " + Fore.GREEN + f"{device}\n")

########################################## PHASE 3: OPTIMIZER SELECTION #########################################################
# Set the optimizer
if configr['optim_name'] == "SGD":
    optimizer = optim.SGD(params=model.parameters(), lr=configr['learning_rate'], momentum=configr['momentum'], weight_decay=configr['weight_decay'])
elif configr['optim_name'] == "ADAM":
    optimizer = optim.Adam(params=model.parameters(), lr=configr['learning_rate'])
elif configr['optim_name'] == "ADAMW":
    optimizer = optim.AdamW(params=model.parameters(), lr=configr['learning_rate'], weight_decay=configr['weight_decay'])
else:
    print(Fore.RED + Style.BRIGHT + "\nATTENTION: No optimizer selected! Setting default option SGD.")
    optimizer = optim.SGD(model.parameters(), lr=configr['learning_rate'], momentum=configr['momentum'], weight_decay=configr['weight_decay'])

########################################## PHASE 4: SCHEDULER SELECTION #########################################################
# Set the learning rate scheduler
scheduler = None

#scheduler = StepLR(optimizer, step_size = 10, gamma=0.5)
#scheduler = StepLR(optimizer, step_size= int(configr['epochs'] / 3), gamma=0.5)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, threshold=0.0001)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

configr["scheduler"] = "None"    #change the scheduler name

########################################## PHASE 5: DATASET TRANSFORMATION #########################################################
# Train transform
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

# Test transform
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

########################################## PHASE 7: DATASET and DATALOADER ###########################################
# Train loader settings
print(Fore.CYAN + Style.BRIGHT + "LOADING DATASET:\n----------------------------------------")
print(Fore.CYAN + "Trainset loading:")
trainset = torchvision.datasets.MNIST(root=configr['database_path'], train=True, download=True, transform=transform_train)
print("Trainset shape: "+ Fore.GREEN + f"{trainset.data.shape}")

# Test loader settings
print(Fore.CYAN + "Testset loading:")
testset = torchvision.datasets.MNIST(root=configr['database_path'], train=False, download=True, transform=transform_test)
print("Testset shape: "+ Fore.GREEN + f"{testset.data.shape}")
test_loader = torch.utils.data.DataLoader(testset, batch_size=configr['batch_size'], shuffle=False, num_workers=configr["num_loader_workers"])

# Validation loader settings
# Setting the percentage split
if configr["validation_size"] > 1:
    print(f"Validation set size: " + Fore.GREEN + f"{configr["validation_size"]}%")
    val_size = int(configr["validation_size"] / 100 * len(trainset))
    train_size = len(trainset) - val_size
    # Splitting the dataset
    train_dataset, val_dataset = torch.utils.data.random_split(trainset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=configr['batch_size'], shuffle=True, num_workers=configr["num_loader_workers"])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=configr['batch_size'], shuffle=False, num_workers=configr["num_loader_workers"])
else:
    val_loader = test_loader
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=configr['batch_size'], shuffle=True, num_workers=configr["num_loader_workers"])


########################################## PHASE 8: TRAINER #########################################################
# Create the trainer with the path to the YAML configuration file
trainer = bMT.BaseTrainer(model=model,
                        optimizer=optimizer,
                        loss_fn=torch.nn.CrossEntropyLoss(),
                        train_loader=train_loader,
                        device = device,
                        scheduler=scheduler,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        wand_config = configr,
                        logger = None)

# Start training
trainer.train()

########################################## PHASE 9: TESTING #########################################################
print(Fore.CYAN + Style.BRIGHT +"\nEND TRAINING:\n----------------------------------------")
trainer.test_eval()
print(Fore.CYAN + Style.BRIGHT +"\n------------------END-------------------")
