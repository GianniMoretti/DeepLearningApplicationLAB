# DeepLearningApplicationLAB
This repo contains all the material for the Deep Learning Application Laboratory. 

**Aggiungi la spiegazione di come è strutturato il lab.** 


# BaseTrainer
My default base trainer, I will use it as a pattern for all my LAB projects.

## Overview

BaseTrainer is a modular and extensible framework designed to simplify the process of training machine learning models. It provides a structured approach to model training, evaluation, and logging, making it easier to manage experiments and reproduce results. The framework is particularly suited for deep learning tasks and integrates seamlessly with [Weights & Biases](https://wandb.ai/) for experiment tracking.

## Features

- **Modular Design**: Easily extendable components for models, metrics, and training logic.
- **Experiment Tracking**: Integrated with Weights & Biases (wandb) for logging metrics, losses, and hyperparameters.
- **Early Stopping**: Configurable early stopping to prevent overfitting.
- **Checkpointing**: Save and resume training from checkpoints.
- **Custom Metrics**: Support for custom evaluation metrics like F1-score, precision, recall, and AUC.
- **Multi-GPU Support**: Compatible with CUDA for GPU acceleration.
- **Configurable**: YAML-based configuration for hyperparameters and settings.

## Project Structure

```
BaseTrainer/
├── BaseTrainer/
│   ├── baseModelTrainer.py   # Core training logic and utilities
│   ├── myModels.py           # Placeholder for user-defined models
│   ├── myMetrics.py          # Custom metrics for evaluation
│   ├── trainLogger.py        # Logging utility for Markdown logs
│   ├── main.py               # Entry point for training
│   ├── config_files/         # YAML configuration files
│   │   └── config_default.yaml
├── wandb/                    # Weights & Biases experiment logs
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── LICENSE                   # License file
└── .gitignore                # Git ignore rules
```

## Classes and Functionalities

### `BaseTrainer` (in `baseModelTrainer.py`)
The `BaseTrainer` class handles the core training loop, validation, and testing. Key features include:
- **Training Loop**: Supports logging, checkpointing, and learning rate scheduling.
- **Validation and Testing**: Evaluates the model on validation and test datasets.
- **Early Stopping**: Monitors a metric (e.g., accuracy) to stop training when no improvement is observed.
- **Metrics Integration**: Uses custom metrics defined in `myMetrics.py`.

### `myModels.py`
This file serves as a placeholder for user-defined models. Users can define their own models here and import them into the training pipeline. The file includes an example model to demonstrate the structure of a PyTorch model.

#### Example Model
The file contains a simple example model, `Mornet_light`, which is designed for image classification tasks. It consists of:
- Multiple convolutional layers with batch normalization and LeakyReLU activation.
- Dropout layers for regularization.
- A fully connected MLP head for classification.

Users can replace or extend this example with their own models.

### `TrainLogger` (in `trainLogger.py`)
A utility for logging training progress in Markdown format. Features include:
- ANSI-to-Markdown conversion for colored logs.
- Automatic creation of log files in a `trainlog` directory.

### Metrics (in `myMetrics.py`)
Custom metrics for evaluating model performance:
- **Accuracy**: Percentage of correct predictions.
- **F1-Score**: Harmonic mean of precision and recall.
- **Precision**: Ratio of true positives to predicted positives.
- **Recall**: Ratio of true positives to actual positives.
- **AUC**: Area Under the Curve for binary classification.

### Configuration (in `config_files/config_default.yaml`)
The YAML configuration file allows you to customize:
- Model and optimizer settings.
- Dataset paths and batch sizes.
- Early stopping parameters.
- Logging and checkpointing options.

## Dependencies and Installation

This project uses several Python libraries for its proper functioning. Below is a brief description of the packages used:

- **wandb**: Used for experiment tracking and real-time logging via [Weights & Biases](https://wandb.ai/).
- **colorama**: Enables colored output in the console, improving log readability.
- **tqdm**: Provides progress bars to monitor the progress of training loops and other iterative processes.
- **PyYAML**: Used for loading configuration from YAML files.

**Important Note:**  
The project also uses **torch** and **torchvision** for deep learning tasks. However, these packages are **not** included in the `requirements.txt` file because their versions may vary depending on your hardware configuration (CPU, GPU, CUDA, etc.).  
It is recommended to install **torch** and **torchvision** by following the official instructions available at this [link](https://pytorch.org/get-started/locally/).

### Installing Dependencies

To install the listed dependencies, run the following command in the project directory:

```bash
pip install -r requirements.txt
```

## Usage

1. **Prepare the Dataset**: Place your dataset in the path specified in the configuration file (`database_path`).
2. **Modify Configuration**: Update `config_files/config_default.yaml` with your desired settings.
3. **Define Your Model**: Add your custom model to `myModels.py` or use the provided example.
4. **Run Training**: Execute the `main.py` script to start training:
   ```bash
   python BaseTrainer/main.py
   ```
5. **Monitor Progress**: Use the console logs or Weights & Biases dashboard to monitor training.

## Example Configuration

Below is an example of a YAML configuration file:

```yaml
model_name: "Mornet_light"
database_name: "CIFAR10"
use_wandb: True
wand_project_name: "CVMR_CIFAR10"
print_model: True
database_path: "../dataset/CIFAR10/"
seed: -1
num_loader_workers: 8
validation_size: 20
metric: "accuracy"
early_stopping_mode: "Max"
early_stopping_patience: 10
early_stopping_delta: 0.005
save_best_checkpoint: False
delta_save_checkpoint: -1
resume_checkpoint_path: "None"
optim_name: "SGD"
learning_rate: 0.005
batch_size: 128
epochs: 50
weight_decay: 0.0001
momentum: 0.9
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Weights & Biases](https://wandb.ai/) for providing an excellent experiment tracking platform.
- [PyTorch](https://pytorch.org/) for being the backbone of this project.