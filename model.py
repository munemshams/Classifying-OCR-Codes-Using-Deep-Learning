import torch
import torch.nn as nn


class OCRModel(nn.Module):
    """
    Multimodal model:
      - Image branch (CNN) for OCR-like code images
      - Type branch (MLP) for one-hot insurance type vector (length 5)
      - Fusion (concat) + classifier head for binary classification:
          0 = primary_id
          1 = secondary_id
    """

    def __init__(self, type_dim: int = 5):
        super().__init__()

        # CNN feature extractor for grayscale images (C=1)
        self.image_layer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # (B,16,H,W)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (B,16,H/2,W/2)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # (B,32,H/2,W/2)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (B,32,H/4,W/4)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (B,64,H/4,W/4)
            nn.ReLU(),

            # Make it robust to unknown H/W by pooling to fixed size
            nn.AdaptiveAvgPool2d((4, 4)),                 # (B,64,4,4)
            nn.Flatten()                                  # (B,64*4*4)= (B,1024)
        )

        # Process the one-hot type vector
        self.type_layer = nn.Sequential(
            nn.Linear(type_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU()
        )

        # Fusion head (image_features + type_features)
        self.classifier = nn.Sequential(
            nn.Linear(1024 + 16, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)  # logits (use BCEWithLogitsLoss)
        )

    def forward(self, image: torch.Tensor, type_vec: torch.Tensor) -> torch.Tensor:
        img_feat = self.image_layer(image)
        type_feat = self.type_layer(type_vec)
        fused = torch.cat([img_feat, type_feat], dim=1)
        logits = self.classifier(fused)
        return logits
