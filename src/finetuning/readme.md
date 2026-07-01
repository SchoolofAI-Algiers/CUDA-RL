## Finetuning Source Code 
In this directory within the repository, we include the source code used for the finetuning, the directory includes the following : 
- **config.yaml:** Contains all configuration parameters for fine-tuning, including model selection, dataset paths, hyperparameters, and training settings.
- **datasetLoader.py:** Responsible for loading and formatting the dataset into the required structure for training.
- **gpuStats.py:** Retrieves GPU information during training to monitor resource usage and detect memory or compute bottlenecks.
- **modelLoader.py:** Handles model initialization and loading based on the specifications defined in the configuration file.
- **trainer.py:** Builds and configures the supervised fine-tuning (SFT) trainer, including training arguments and optimization setup.
- **train.py:** The main entry point of the pipeline. It loads the configuration, initializes the dataset and model, and launches the fine-tuning process.

We have loaded the finetuning directory within HPC and run it and got the following results : <br>

![beforelogging](./execution/Capture%20d’écran%202026-04-12%20à%201.54.07 PM.png)
