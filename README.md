This project is aiming at achieving respectable accuracy on the cifar10 dataset using Convolutional Neural Networks.

## Installation

To set up the environment and install all the necessary dependencies, follow these steps:

1. **Clone the Repository:**

    ```sh
    git clone https://github.com/pelagecha/CNN
    cd CNN
    ```

2. **Create a Virtual Environment:**

    Create a virtual environment to isolate your project's dependencies. You can use `venv` for this:

    ```sh
    python -m venv myenv
    ```

    Replace `myenv` with your preferred name for the virtual environment.

3. **Activate the Virtual Environment:**

    - **On Windows:**

        ```sh
        myenv\Scripts\activate
        ```

    - **On macOS and Linux:**

        ```sh
        source myenv/bin/activate
        ```

4. **Install Dependencies:**

    Install the packages listed in `requirements.txt`. Pip will only download and install packages that are not already present in your virtual environment:

    ```sh
    pip install --upgrade -r requirements.txt
    ```

    This command ensures that you have the required packages, upgrading them if necessary, and only installing those that are missing.

By following these steps, you will set up your environment and ensure that all dependencies are properly installed.

## Usage

1. **Configure the Training:**

    - **Dataset Selection:** Modify the `dataset_name` variable in `training.py` to choose between `"CIFAR10"` or `"MNIST"`.

        ```python
        dataset_name = "CIFAR10"  # Change to "MNIST" for the MNIST dataset
        ```

    - **Adjust Hyperparameters:** The batch size, learning rate, and number of epochs are specified in `datasets.json`. You can adjust these values directly in `datasets.json` to fit your needs. Example `datasets.json` configuration:

        ```json

        "CIFAR10": {
            ...
            "batch_size": 256,
            "learning_rate": 0.005,
            ...
            }
        }
        ```

        In `training.py`, these parameters are loaded as follows:

        ```python
        batch_size = settings["batch_size"]
        lr = settings["learning_rate"]
        num_epochs = 5
        ```

2. **Run Training:**

    Execute the script to start training the model:

    ```sh
    python3 train.py
    ```

    The script will handle data loading, model training, and saving the trained model.
