import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw
import os
import sys
import torch
from transformers import BertTokenizer

# --- Add src directory to path to allow local imports ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from model_advanced import GroundingAdvancedModel # Use the SOTA model

class GroundingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Visual Grounding Inference")
        self.root.geometry("1100x600")

        # --- Configuration ---
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.MODEL_WEIGHTS_PATH = r"outputs\weights\new_sota_train\new_sota_model_epoch_13.pth"
        self.BASE_IMAGE_DIR = r"C:\projects\AIMS_TASK\visual_grounding\data\flickr30k_entities\images"

        # --- Load Model and Tokenizer ---
        self.model = self.load_model()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # --- UI Layout ---
        self.create_widgets()

    def load_model(self):
        try:
            print("Loading SOTA model...")
            model = GroundingAdvancedModel().to(self.DEVICE)
            model.load_state_dict(torch.load(self.MODEL_WEIGHTS_PATH, map_location=self.DEVICE))
            model.eval()
            print("Model loaded successfully.")
            return model
        except Exception as e:
            messagebox.showerror("Model Loading Error", f"Failed to load model weights:\n{e}")
            self.root.destroy()
            return None

    def create_widgets(self):
        # --- Main Frame ---
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Input Frame ---
        input_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        input_frame.pack(fill=tk.X, pady=5)

        ttk.Label(input_frame, text="Image ID:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.image_id_entry = ttk.Entry(input_frame, width=20)
        self.image_id_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.image_id_entry.insert(0, "36979") # Example ID

        ttk.Label(input_frame, text="Text Prompt:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.prompt_entry = ttk.Entry(input_frame, width=50)
        self.prompt_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.prompt_entry.insert(0, "A group of friends") # Example prompt

        self.predict_button = ttk.Button(input_frame, text="Run Prediction", command=self.run_prediction)
        self.predict_button.grid(row=0, column=2, rowspan=2, padx=10, pady=5, ipady=10)

        input_frame.columnconfigure(1, weight=1)

        # --- Image Display Frame ---
        image_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        image_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.original_panel = ttk.Label(image_frame, text="Original Image", anchor="center")
        self.original_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.boxed_panel = ttk.Label(image_frame, text="Predicted Box", anchor="center")
        self.boxed_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

    def run_prediction(self):
        image_id = self.image_id_entry.get().strip()
        prompt = self.prompt_entry.get().strip()

        if not image_id or not prompt:
            messagebox.showwarning("Input Error", "Please provide both an Image ID and a Text Prompt.")
            return

        image_path = os.path.join(self.BASE_IMAGE_DIR, f"{image_id}.jpg")
        if not os.path.exists(image_path):
            messagebox.showerror("File Error", f"Image not found at:\n{image_path}")
            return

        # --- Load and Display Original Image ---
        original_image = Image.open(image_path).convert("RGB")
        img_w, img_h = original_image.size
        
        # Resize for display
        display_img = original_image.copy()
        display_img.thumbnail((500, 500))
        self.original_photo = ImageTk.PhotoImage(display_img)
        self.original_panel.config(image=self.original_photo)

        # --- Preprocess and Predict ---
        image_tensor = self.model.visual_processor(images=original_image, return_tensors="pt")['pixel_values'].to(self.DEVICE)
        tokenized = self.tokenizer(prompt, padding='max_length', truncation=True, max_length=32, return_tensors="pt")
        input_ids = tokenized['input_ids'].to(self.DEVICE)
        attention_mask = tokenized['attention_mask'].to(self.DEVICE)

        with torch.no_grad():
            pred_logits, pred_boxes = self.model(image_tensor, input_ids, attention_mask)

        # --- Post-process and Draw Box ---
        best_box_idx = pred_logits.squeeze().argmax()
        best_box = pred_boxes.squeeze()[best_box_idx].cpu().numpy()

        box_unnormalized = [
            (best_box[0] - best_box[2] / 2) * img_w,
            (best_box[1] - best_box[3] / 2) * img_h,
            (best_box[0] + best_box[2] / 2) * img_w,
            (best_box[1] + best_box[3] / 2) * img_h,
        ]

        boxed_image = original_image.copy()
        draw = ImageDraw.Draw(boxed_image)
        draw.rectangle(box_unnormalized, outline="lime", width=4)

        # --- Display Boxed Image ---
        boxed_image.thumbnail((500, 500))
        self.boxed_photo = ImageTk.PhotoImage(boxed_image)
        self.boxed_panel.config(image=self.boxed_photo)

if __name__ == "__main__":
    root = tk.Tk()
    app = GroundingApp(root)
    root.mainloop()