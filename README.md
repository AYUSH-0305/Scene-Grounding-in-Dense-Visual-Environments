# Visual Grounding with Flickr30k Entities

This project implements a **visual grounding** system using the Flickr30k Entities dataset.  
It allows you to train a state-of-the-art model for grounding natural language phrases in images and interactively test predictions via a GUI.

---

## 📂 Project Structure

```
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
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python **3.8+**
- NVIDIA GPU with CUDA (recommended for training)

### Installation Steps
```bash
# Clone the repository
git clone <your-repo-url>
cd visual_grounding

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install torch torchvision transformers timm Pillow
```

---

## 📦 Dataset Setup

This project uses the **Flickr30k Entities** dataset.

1. **Download Dataset**  
   Get the dataset from the official source and extract it.

2. **Organize Data**  
   Place files in:
   ```
   data/flickr30k_entities/
       ├── annotations/
       ├── images/
       └── Sentences/
   ```

3. **Update Paths**  
   In both:
   - `train_new_sota.py`
   - `run_gui.py`

   Set:
   ```python
   BASE_DATA_DIR = r"C:\projects\AIMS_TASK\visual_grounding\data\flickr30k_entities"
   ```
   *(Replace with your actual absolute path)*

---

## 🚀 Training the Model

To train from scratch:
```bash
python train_new_sota.py
```

- Uses GPU if available.
- Model weights are saved in:
  ```
  outputs/weights/new_sota_train/
  ```

---

## 🖥 Running the GUI

1. **Set Model Path** in `run_gui.py`:
```python
MODEL_WEIGHTS_PATH = r"outputs/weights/new_sota_train/new_sota_model_epoch_13.pth"
```

2. **Run GUI**:
```bash
python run_gui.py
```

3. **Usage**:
   - Enter **Image ID** from Flickr30k (e.g., `36979`).
   - Enter a **Text Prompt** describing the object (e.g., `a green table`).
   - Click **Run Prediction** to see:
     - Original image
     - Predicted bounding box

---

## 📌 Notes
- GPU acceleration is recommended for faster training and inference.
- Hyperparameters (batch size, learning rate) can be adjusted in `train_new_sota.py`.

---

## 📜 License
This project is for research and educational purposes. Please verify dataset licensing before use.
