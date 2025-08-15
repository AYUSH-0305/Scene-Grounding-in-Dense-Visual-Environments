visual_grounding/
│
├── data/
│   └── flickr30k_entities/
│       ├── annotations/
│       ├── images/
│       └── Sentences/
│
├── outputs/
│   └── weights/
│       └── new_sota_train/
│           └── new_sota_model_epoch_13.pth
│
├── scripts/
│   ├── explore_data.py
│   └── ...
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── model_advanced.py
│   └── ...
│
├── run_gui.py
├── train_new_sota.py
└── README.md

1. Setup and Installation
Prerequisites
Python 3.8+

NVIDIA GPU with CUDA support (recommended for training)

Installation
Clone the repository:

git clone <your-repo-url>
cd visual_grounding

Create a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install the required libraries:

pip install torch torchvision transformers timm Pillow

2. Dataset Setup
Download the Dataset: This project uses the Flickr30k Entities dataset. You will need to acquire it and extract its contents.

Organize the Data: Place the dataset folders into the data/flickr30k_entities/ directory as shown in the project structure above.

Update Paths: Open the following files and ensure that the BASE_DATA_DIR variable points to the correct absolute path of your flickr30k_entities folder:

train_new_sota.py

run_gui.py

Example path: C:\projects\AIMS_TASK\visual_grounding\data\flickr30k_entities

3. How to Use the Project
Training the Model
To train the state-of-the-art model from scratch, run the advanced training script from the root directory:

python train_new_sota.py

The script will automatically use a GPU if available.

Model weights will be saved in the outputs/weights/new_sota_train/ directory after each epoch.

Running the GUI
To use the interactive GUI with the trained model, you first need to update the model path.

Update Model Path: Open run_gui.py and ensure the MODEL_WEIGHTS_PATH variable points to your best-trained model (e.g., the one from epoch 13).

Launch the GUI:

python run_gui.py

Usage:

Enter an Image ID from the Flickr30k dataset (e.g., 36979).

Enter a Text Prompt describing an object in the image (e.g., a green table).

Click "Run Prediction" to see the original image and the model's predicted bounding box.
