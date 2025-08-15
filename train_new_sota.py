import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torchvision.ops import generalized_box_iou_loss, box_convert
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from dataset import CustomFlickrDataset
# --- IMPORTANT: We now import the new model from the new file ---
from model_advanced import GroundingAdvancedModel

# --- Configuration ---
EPOCHS = 15
BATCH_SIZE = 16 
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
BASE_DATA_DIR = r"C:\projects\AIMS_TASK\visual_grounding\data\flickr30k_entities"
MODEL_SAVE_DIR = os.path.join("outputs", "weights", "new_sota_train")

class Matcher:
    """Matches the best predicted box to the ground truth box."""
    def __init__(self, cost_l1: float = 2.0, cost_giou: float = 5.0):
        self.cost_l1 = cost_l1
        self.cost_giou = cost_giou

    @torch.no_grad()
    def __call__(self, pred_boxes, gt_boxes):
        gt_boxes_unsqueezed = gt_boxes.unsqueeze(1)
        cost_l1 = torch.cdist(pred_boxes, gt_boxes_unsqueezed, p=1).squeeze(-1)
        pred_boxes_xyxy = box_convert(pred_boxes, in_fmt='cxcywh', out_fmt='xyxy')
        gt_boxes_xyxy = box_convert(gt_boxes_unsqueezed, in_fmt='cxcywh', out_fmt='xyxy')
        cost_giou = (1 - generalized_box_iou_loss(pred_boxes_xyxy, gt_boxes_xyxy, reduction='none')).squeeze(-1)
        cost_matrix = self.cost_l1 * cost_l1 + self.cost_giou * cost_giou
        best_match_indices = cost_matrix.argmin(dim=1)
        return best_match_indices

class CombinedSetLoss(nn.Module):
    """Computes the final loss by combining alignment, L1, and GIoU losses."""
    def __init__(self, matcher, weight_l1=2.0, weight_giou=5.0, weight_align=1.0):
        super().__init__()
        self.matcher = matcher
        self.weight_l1 = weight_l1
        self.weight_giou = weight_giou
        self.weight_align = weight_align
        self.alignment_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred_logits, pred_boxes, gt_boxes):
        match_indices = self.matcher(pred_boxes, gt_boxes)
        target_labels = torch.zeros_like(pred_logits)
        target_labels.scatter_(1, match_indices.unsqueeze(1), 1.0)
        loss_align = self.alignment_loss(pred_logits, target_labels)

        matched_pred_boxes = torch.gather(pred_boxes, 1, match_indices.view(-1, 1, 1).expand(-1, 1, 4)).squeeze(1)
        loss_l1 = F.l1_loss(matched_pred_boxes, gt_boxes)
        
        pred_xyxy = box_convert(matched_pred_boxes, 'cxcywh', 'xyxy')
        gt_xyxy = box_convert(gt_boxes, 'cxcywh', 'xyxy')
        loss_giou = generalized_box_iou_loss(pred_xyxy, gt_xyxy).mean()

        total_loss = (self.weight_align * loss_align) + (self.weight_l1 * loss_l1) + (self.weight_giou * loss_giou)
        return total_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = CustomFlickrDataset(base_data_dir=BASE_DATA_DIR, split='train')
    val_dataset = CustomFlickrDataset(base_data_dir=BASE_DATA_DIR, split='val')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    model = GroundingAdvancedModel().to(device)
    matcher = Matcher()
    criterion = CombinedSetLoss(matcher=matcher).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    print("\n--- Starting New SOTA Model Training ---")
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    for epoch in range(EPOCHS):
        model.train()
        for i, batch in enumerate(train_loader):
            images, ids, mask, boxes = batch['image'].to(device), batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['box'].to(device)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                pred_logits, pred_boxes = model(images, ids, mask)
                loss = criterion(pred_logits, pred_boxes, boxes)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if (i+1) % 100 == 0: print(f"  Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}, Train Loss: {loss.item():.4f}")
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images, ids, mask, boxes = batch['image'].to(device), batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['box'].to(device)
                pred_logits, pred_boxes = model(images, ids, mask)
                val_loss += criterion(pred_logits, pred_boxes, boxes).item()
        
        print(f"\nEpoch {epoch+1} complete. Avg Val Loss: {val_loss / len(val_loader):.4f}")
        scheduler.step()
        
        epoch_save_path = os.path.join(MODEL_SAVE_DIR, f"new_sota_model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), epoch_save_path)
        print(f"Model saved to: {epoch_save_path}")

    print("\n--- All Training Epochs Complete ---")

if __name__ == '__main__':
    main()