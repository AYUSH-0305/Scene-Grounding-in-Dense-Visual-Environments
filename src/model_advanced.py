import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor

class GroundingAdvancedModel(nn.Module):
    
    def __init__(self, num_queries=20, hidden_dim=256, 
                 text_encoder_name="bert-base-uncased", 
                 visual_encoder_name="facebook/dinov2-base"):
        super().__init__()

        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # 1. Load SOTA Pre-trained Encoders
        print(f"Loading text encoder: {text_encoder_name}")
        self.text_encoder = AutoModel.from_pretrained(text_encoder_name)
        
        print(f"Loading visual encoder: {visual_encoder_name}")
        self.visual_encoder = AutoModel.from_pretrained(visual_encoder_name)
        self.visual_processor = AutoProcessor.from_pretrained(visual_encoder_name)

        # Freeze encoders for stable fine-tuning
        for param in self.text_encoder.parameters():
            param.requires_grad_(False)
        for param in self.visual_encoder.parameters():
            param.requires_grad_(False)

        # 2. Projection Layers to Unify Dimensions
        text_feature_dim = self.text_encoder.config.hidden_size
        visual_feature_dim = self.visual_encoder.config.hidden_size
        self.text_proj = nn.Linear(text_feature_dim, hidden_dim)
        self.visual_proj = nn.Linear(visual_feature_dim, hidden_dim)

        # 3. DETR-style Cross-Modal Decoder for Fusion
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        # 4. Learnable Object Queries
        self.object_queries = nn.Embedding(num_queries, hidden_dim)

        # 5. Prediction Heads
        self.bbox_head = nn.Linear(hidden_dim, 4)  # (cx, cy, w, h)
        self.alignment_head = nn.Linear(hidden_dim, 1)

    def forward(self, images, input_ids, attention_mask):
        visual_outputs = self.visual_encoder(pixel_values=images).last_hidden_state
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state

        # Project features to common dimension
        visual_features_proj = self.visual_proj(visual_outputs)
        text_features_proj = self.text_proj(text_features)

        # Perform cross-modal fusion
        batch_size = visual_features_proj.shape[0]
        query_embed = self.object_queries.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        fused_features = self.decoder(
            tgt=query_embed,                 
            memory=torch.cat([visual_features_proj, text_features_proj], dim=1),
        )

        # Generate final predictions
        pred_logits = self.alignment_head(fused_features)
        pred_boxes = self.bbox_head(fused_features).sigmoid() 

        return pred_logits.squeeze(-1), pred_boxes
