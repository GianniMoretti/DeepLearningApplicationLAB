import yaml
import torch
import wandb
from tqdm import tqdm
from colorama import Fore, Style, init
import os

# Import the metrics dictionary from the external file
from myMetrics import metrics

init(autoreset=True)

def load_yaml_config(config_path):
    """
    Loads the configuration from the YAML file.
    :param config_path: Path to the YAML file.
    :return: Configuration as a dictionary.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def set_seed(seed):
    # Only include these if you use them
    # random.seed(seed)  # Python
    # np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # For multiple GPUs
    torch.backends.cudnn.deterministic = True  # Ensures cuDNN determinism
    # torch.backends.cudnn.benchmark = False  # Disables dynamic cuDNN optimization


class BaseTrainer:
    def __init__(self, model, optimizer, loss_fn, train_loader, device='CPU', scheduler=None,
                val_loader=None, test_loader=None, wand_config=None, logger = None):
        """
        Initializes the trainer.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = wand_config["epochs"]
        self.start_epoch = 0
        self.device = device
        self.num_parameters = wand_config["num_param"]
        self.scheduler = scheduler
        self.logger = logger
        self.runID = wand_config["runID"]
        self.checkpoint_path = wand_config["checkpoint_path"]
        self.save_best_checkpoint = wand_config["save_best_checkpoint"]
        self.delta_save_checkpoint = wand_config["delta_save_checkpoint"]
        if self.delta_save_checkpoint > 0 or self.save_best_checkpoint == True:
            os.makedirs(self.checkpoint_path, exist_ok=True)

        self.model.to(self.device)

        if wand_config["resume_checkpoint_path"] != "None":
            self.resume_from_checkpoint(wand_config["resume_checkpoint_path"])

        # Select the metric (default "accuracy")
        metric_name = wand_config.get("metric", "accuracy")
        if metric_name in metrics:
            self.metric_fn = metrics[metric_name]
            self.metric_name = metric_name
        else:
            self.log(Fore.RED + Style.BRIGHT + f"Metric '{metric_name}' not recognized. Choose from: {list(metrics.keys())}")

        # Parameters for Early Stopping:
        # If early_stopping_mode is set to None or "None", the early stopping logic is disabled.
        self.early_stopping_mode = wand_config.get("early_stopping_mode", "None")
        self.use_early_stopping = self.early_stopping_mode not in [None, "None"]
        if self.use_early_stopping:
            self.early_stopping_patience = wand_config.get("early_stopping_patience", 5)
            self.early_stopping_delta = wand_config.get("early_stopping_delta", 0.0)

    def resume_from_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        # Add other information if saved
        print(Fore.RED + Style.BRIGHT + f"\nResuming training from epoch {self.start_epoch}")

    def save_checkpoint(self, epoch, checkpoint_path="checkpoint.pth"):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            # potentially, other information like loss state if any
        }
        torch.save(checkpoint, checkpoint_path)

    def train(self):
        """
        Training loop with early stopping support.
        If early_stopping_mode is None, the check is skipped.
        """
        self.log(Fore.CYAN + Style.BRIGHT + "\nTRAINING:\n----------------------------------------")
        epoch_bar = tqdm(range(self.start_epoch, self.epochs), desc=Fore.CYAN + "Epoch Progress: ", ncols=150)
        
        # Variables for early stopping (only if enabled)
        if self.use_early_stopping and self.val_loader:
            best_metric = None
            early_stop_counter = 0
            earlystopperc = 0

        for epoch in epoch_bar:
            self.model.train()
            batch_bar = tqdm(self.train_loader, desc=Fore.GREEN + f"  ┗━━ Training: ", unit="batch", leave=False, ncols=150, position=1)
            for batch in batch_bar:
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)

                loss.backward()
                self.optimizer.step()

                batch_bar.set_postfix({
                    "Loss": f"{loss.item():.4f}"
                })

            # runnig_avg_loss = sum_loss / len(self.train_loader)
            avg_train_loss, train_metric = self.train_eval()

            if self.val_loader:
                avg_val_loss, val_metric = self.validation_eval()

                # If early stopping is enabled, check for improvement
                if self.use_early_stopping:
                    # Determine the value to monitor
                    if self.early_stopping_mode == "min":
                        current_metric = avg_val_loss
                    else:
                        current_metric = val_metric

                    if best_metric is None:
                        best_metric = current_metric
                    else:
                        if self.early_stopping_mode == "min":
                            improvement = (best_metric - current_metric) 
                        else:
                            improvement = (current_metric - best_metric)
                        
                        if improvement > self.early_stopping_delta:
                            best_metric = current_metric
                            early_stop_counter = 0
                            if self.save_best_checkpoint:
                                self.save_checkpoint(epoch, self.checkpoint_path + f"/{self.runID}_checkpoint_best.pth")
                        else:
                            early_stop_counter += 1
                        
                        earlystopperc = early_stop_counter/self.early_stopping_patience
                
                # Log on wandb
                log_data = {
                    "early_stop_%" : earlystopperc,
                    "epoch": epoch+1,
                    "train_loss": avg_train_loss,
                    f"train_{self.metric_name}": train_metric,
                    "val_loss": avg_val_loss,
                    f"val_{self.metric_name}": val_metric,
                }
                wandb.log(log_data, step=epoch+1)
            else:
                wandb.log({
                    "epoch": epoch+1,
                    "train_loss": avg_train_loss,
                    f"train_{self.metric_name}": train_metric,
                }, step=epoch+1)

            if self.scheduler is not None:
                monitor_value = val_metric if self.val_loader else train_metric
                self.scheduler.step(monitor_value)    
                # self.scheduler.step()   # Only for Step
                current_lr = self.scheduler.get_last_lr()[0]
                wandb.log({"epoch": epoch+1, "learning_rate": current_lr}, step=epoch+1)

            # If early stopping is enabled, check if it's time to stop
            if self.use_early_stopping and self.val_loader and early_stop_counter >= self.early_stopping_patience:
                self.log(Fore.RED + Style.BRIGHT + "Early stopping triggered!")
                break

            if self.delta_save_checkpoint > 0 and epoch % self.delta_save_checkpoint == 0 and epoch > self.delta_save_checkpoint - 1:
                self.save_checkpoint(epoch, self.checkpoint_path + f"/{self.runID}_checkpoint_{epoch}.pth")

            batch_bar.close()

    def train_eval(self):
        """
        Train evaluation.
        Returns: (avg_val_loss, val_metric)
        """
        self.model.eval()
        all_outputs = []
        all_labels = []
        sum_loss = 0
        with torch.no_grad():
            for batch in tqdm(self.train_loader, desc=Fore.YELLOW + f"  ┗━━ Train_ev: ", unit="batch", leave=False, ncols=150, position=1):
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                all_outputs.append(outputs)
                all_labels.append(labels)
                loss = self.loss_fn(outputs, labels)
                sum_loss += loss.item()

        avg_loss = sum_loss / len(self.train_loader)
        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        val_metric = self.metric_fn(all_outputs, all_labels)

        return avg_loss, val_metric

    def validation_eval(self):
        """
        Validation on the validation dataset.
        Returns: (avg_val_loss, val_metric)
        """
        self.model.eval()
        sum_loss = 0
        all_outputs = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=Fore.MAGENTA + f"  ┗━━━━━ Valid: ", unit="batch", leave=False, ncols=150, position=1):
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                sum_loss += loss.item()
                all_outputs.append(outputs)
                all_labels.append(labels)

        avg_loss = sum_loss / len(self.val_loader)
        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        val_metric = self.metric_fn(all_outputs, all_labels)

        return avg_loss, val_metric

    def test_eval(self):
        """
        Tests the model on the test set.
        Returns: (avg_test_loss, test_metric)
        """
        if self.save_best_checkpoint:
            self.model.load_state_dict(torch.load(self.checkpoint_path + f"/{self.runID}_checkpoint_best.pth", weights_only=True)["model_state_dict"])
            self.model.to(self.device)
            print(Fore.GREEN + Style.BRIGHT + f"Model loaded from the best checkpoint.")

        self.model.eval()
        test_loss = 0
        all_outputs = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc=Fore.RED + f"Test: ", unit="batch", leave=False, ncols=150):
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                test_loss += loss.item()
                all_outputs.append(outputs)
                all_labels.append(labels)

        avg_test_loss = test_loss / len(self.test_loader)
        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        test_metric = self.metric_fn(all_outputs, all_labels)

        self.log(Fore.CYAN + Style.BRIGHT + "\nTEST RESULT:\n----------------------------------------")
        self.log("Test Loss: " + Fore.CYAN + f"{avg_test_loss:.4f}")
        self.log(f"{self.metric_name.capitalize()}:" + Fore.CYAN + f" {test_metric:.2f}")
        wandb.log({"#parameters": self.num_parameters / 1000000, f"Test_{self.metric_name}": test_metric})
        return avg_test_loss, test_metric

    def log(self, logmsg):
        if self.logger != None:
            self.logger.log(logmsg)
        else:
            print(logmsg)
