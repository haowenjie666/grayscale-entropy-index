# grayscale-entropy-index

# Image Entropy Prediction using ResNet-50

This project implements a regression model using a modified ResNet-50 architecture in PyTorch. The primary goal is to predict a continuous "entropy value" from single-channel, 16-bit TIF images. This solution is well-suited for scientific or industrial applications that require quantitative feature extraction from specialized image formats.

## ✨ Features

  - **Core Model**: Utilizes a ResNet-50 model adapted for a regression task.
  - **Input Compatibility**: Natively handles single-channel (grayscale) 16-bit TIF images, including normalization and data preprocessing.
  - **Custom Data Handling**: Includes a custom PyTorch `Dataset` class for efficiently loading images from a directory and corresponding labels from an Excel file.
  - **Optimized Training**: The training script automatically evaluates the model on a test set after each epoch and saves the weights of the model with the lowest validation loss.
  - **Result Visualization**: Automatically plots and displays the training and validation loss curves upon completion of the training process.
  - **Standalone Inference**: Provides a separate script to load a pre-trained model and perform inference on a single image.

## ⚙️ Requirements

This project requires the following Python libraries. It is highly recommended to use a virtual environment.

  - Python (\>= 3.8)
  - PyTorch (\>= 1.10)
  - Torchvision
  - Pandas
  - NumPy
  - Matplotlib
  - Pillow
  - openpyxl (for reading .xlsx files)

You can save the following content as `requirements.txt` and install all dependencies at once.

```txt
torch
torchvision
pandas
numpy
matplotlib
Pillow
openpyxl
```

## 🚀 Installation

1.  **Clone the Repository**

    ```bash
    git clone https://your-repository-url.git
    cd your-project-directory
    ```

2.  **Create and Activate a Virtual Environment (Recommended)**

    ```bash
    # Create the environment
    python -m venv venv

    # Activate on Windows
    .\venv\Scripts\activate

    # Activate on macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

## 📁 Data Preparation

To run the scripts correctly, please organize your project directory as follows:

```
your_project_root/
│
├── data/
│   ├── images/  # Store all .tif images here
│   │   ├── image_001.tif
│   │   ├── image_002.tif
│   │   └── ...
│   │
│   └── labels.xlsx  # Excel file with image IDs and labels
│
├── weights/  # Directory for storing trained model weights
│
├── train.py         # Training script
├── test_image.py    # Single image inference script
└── README.md
```

**`labels.xlsx` File Format:**

Ensure your Excel file contains at least the following two columns:

  - `ImageID`: The filename of the image (e.g., `image_001.tif`).
  - `EntropyValue`: The corresponding numerical label.

| ImageID       | EntropyValue |
|---------------|--------------|
| image\_001.tif | 12.345       |
| image\_002.tif | 15.678       |
| ...           | ...          |

## ▶️ How to Run

### 1\. Train the Model

The training script (`train.py`) will execute the complete training and validation pipeline, saving the best-performing model at the end.

1.  **Configure Paths**: Open `train.py` and modify the following path variables to match your directory structure:

    ```python
    # Path to data and Excel file
    image_dir = r'data/images'  # Points to your image folder
    excel_file = r'data/labels.xlsx' # Points to your labels file

    # ...

    # Path to save the best model
    best_model_save_path = r'weights/resnet50_best_model.pth'
    ```

2.  **Start Training**: Run the following command in your terminal with the virtual environment activated:

    ```bash
    python train.py
    ```

After training, the best model will be saved to the `weights/` directory, and a plot showing the loss curves will be displayed.

### 2\. Test a Single Image

The inference script (`test_image.py`) loads a saved model to predict the entropy value for any given TIF image.

1.  **Configure Paths**: Open `test_image.py` and modify the following path variables:

    ```python
    # Define model and image paths
    BEST_MODEL_PATH = r'weights/resnet50_best_model.pth' # Ensure this path is correct
    IMAGE_TO_TEST = r'path/to/your/single/image.tif' # Replace with the full path to your test image
    ```

2.  **Run Inference**: Execute the script from your terminal:

    ```bash
    python test_image.py
    ```

The predicted entropy value for the specified image will be printed to the console.

-----

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## © Copyright

Copyright (c) 2025 Wenjie Hao. All Rights Reserved.
