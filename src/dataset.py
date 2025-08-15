import os
import re
import random
import xml.etree.ElementTree as ET
from PIL import Image

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, RobertaTokenizer
import torchvision.transforms as transforms

class CustomFlickrDataset(Dataset):
    def __init__(self, base_data_dir, tokenizer_name='bert-base-uncased', split='train'):
        super().__init__()
        self.image_dir = os.path.join(base_data_dir, 'images')
        self.annotation_dir = os.path.join(base_data_dir, 'annotations')
        self.sentence_dir = os.path.join(base_data_dir, 'Sentences')
        self.annotated_image_ids = [f.split('.')[0] for f in os.listdir(self.annotation_dir) if f.endswith('.xml')]
        
        # Train/val transforms
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
        # Handle both BERT and RoBERTa tokenizers
        if 'roberta' in tokenizer_name:
            self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

        num_images = len(self.annotated_image_ids)
        num_train = int(num_images * 0.9)
        
        # Split train/val
        if split == 'train':
            self.image_ids = self.annotated_image_ids[:num_train]
        else:
            self.image_ids = self.annotated_image_ids[num_train:]
        print(f"Initialized dataset for '{split}' split with {len(self.image_ids)} samples using '{tokenizer_name}'.")

    def __len__(self):
        return len(self.image_ids)
    
    def _parse_data_for_id(self, image_id):
        id_to_text = {}
        sentence_file_path = os.path.join(self.sentence_dir, f"{image_id}.txt")
        if os.path.exists(sentence_file_path):
            with open(sentence_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    phrases_in_line = re.findall(r'\[/EN#(\d+)/.*? (.*?)\]', line)
                    for phrase_id, phrase_text in phrases_in_line:
                        id_to_text[phrase_id] = phrase_text.strip()
        all_phrases, all_boxes = [], []
        xml_file_path = os.path.join(self.annotation_dir, f"{image_id}.xml")
        if not os.path.exists(xml_file_path): return None, None, None
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            phrase_id = obj.find('name').text
            phrase_text = id_to_text.get(phrase_id)
            if phrase_text:
                for box_elem in obj.findall('bndbox'):
                    xmin, ymin, xmax, ymax = int(box_elem.find('xmin').text), int(box_elem.find('ymin').text), int(box_elem.find('xmax').text), int(box_elem.find('ymax').text)
                    all_boxes.append([xmin, ymin, xmax, ymax])
                    all_phrases.append(phrase_text)
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image = Image.open(image_path).convert("RGB")
        return image, all_phrases, all_boxes

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image, phrases, boxes = self._parse_data_for_id(image_id)
        if not phrases: return self.__getitem__((idx + 1) % len(self))
        random_index = random.randint(0, len(phrases) - 1)
        selected_phrase, selected_box = phrases[random_index], boxes[random_index]
        
        original_w, original_h = image.size
        image_tensor = self.transform(image)
        
        tokenized_text = self.tokenizer(selected_phrase, padding='max_length', truncation=True, max_length=32, return_tensors="pt")
        input_ids, attention_mask = tokenized_text['input_ids'].squeeze(0), tokenized_text['attention_mask'].squeeze(0)

        xmin, ymin, xmax, ymax = selected_box
        w, h = xmax - xmin, ymax - ymin
        cx, cy = xmin + w / 2, ymin + h / 2
        normalized_box = [cx / original_w, cy / original_h, w / original_w, h / original_h]
        box_tensor = torch.tensor(normalized_box, dtype=torch.float32)

        return {'image': image_tensor, 'input_ids': input_ids, 'attention_mask': attention_mask, 'box':
