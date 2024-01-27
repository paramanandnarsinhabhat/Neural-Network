# Neural Network with RMSProp Optimization
This repository contains an implementation of a simple neural network in Python that uses RMSProp optimization. The neural network is built using NumPy for matrix operations and Matplotlib for visualizing the training process.
## Project Structure
- `notebook/`: A directory containing the Jupyter notebook with the implementation.
  - `neuralnetwork.ipynb`: The Jupyter notebook that contains the neural network code and visualizations.
- `source/`: A directory containing the source code in Python.
  - `neural.py`: The Python script with the neural network implementation.
- `myenv/`: A directory that may contain a virtual environment for the project.
- `.gitignore`: Configuration file for Git to ignore files that should not be committed.
- `LICENSE`: The license document for the project.
- `README.md`: This file, which provides an overview and instructions for the project.
- `requirements.txt`: A file listing all the Python dependencies required to run the project.
## Getting Started
To run this project, you will need to install its dependencies.
### Prerequisites
Ensure you have the following installed:

- Python 3.x
- pip (Python package manager)
### Installation
To install the required Python packages, run the following command in your terminal:

pip install -r requirements.txt

The `requirements.txt` file includes the following libraries:

numpy
matplotlib

## Running the Neural Network

Navigate to the `source` directory and run the `neural.py` script to execute the neural network:

python neural.py

You can also open and run the `neuralnetwork.ipynb` notebook within the `notebook` directory if you prefer an interactive environment provided by Jupyter.

## Understanding the Neural Network

The neural network follows these steps:

1. **Importing Libraries**: Uses NumPy for numerical computations and Matplotlib for plotting.
2. **Loading the Dataset**: Initializes the input (`X`) and output (`y`) data.
3. **Defining the Sigmoid Function**: Utilizes the sigmoid function as the activation function.
4. **Setting Hyperparameters**: Defines the learning rate, number of neurons in each layer, and number of epochs for training.
5. **Initializing Weights**: Sets up initial weights for the input to hidden and hidden to output layers.
6. **Training with RMSProp**: Implements the forward and backward propagation with RMSProp optimization during training.
7. **Visualizing Results**: Plots the error over each epoch and the learning rate adjustments.

## Visualizations

After training, the code generates two plots:

1. The error over each training epoch.
2. The learning rates for the hidden and output layers over each epoch.

These visualizations help in understanding the learning process and the effectiveness of the RMSProp optimization method.

## Contributing

We welcome contributions to this project. If you have suggestions or improvements, feel free to make a pull request or open an issue.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
