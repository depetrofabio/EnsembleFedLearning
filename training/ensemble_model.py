import torch
import torch.nn as nn
from typing import List, Optional

class EnsembleModel(nn.Module):
    def __init__(self, models: List[nn.Module], feature_dim: int, num_classes: int = 10):
        """
        Ensemble of models with a shared dense layer.
        
        Args:
            models: List of base models (feature extractors). 
                    Assumes these models output a feature vector.
            feature_dim: Dimension of the output feature vector of a single model.
            num_classes: Number of output classes.
        """
        super().__init__()
        self.models = nn.ModuleList(models) 
        self.num_models = len(models) # num_models = num_clusters. num_clusters < num_clients
        self.feature_dim = feature_dim # 512 per resnet
        
        # The classifier takes concatenated features from all models
        # Input dim = num_models * feature_dim
        self.classifier = nn.Linear(self.num_models * self.feature_dim, num_classes)
        
    def forward(self, x: torch.Tensor, active_indices: Optional[List[int]] = None) -> torch.Tensor:
        """
        Forward pass with optional masking of models.
        
        Args:
            x: Input tensor.
            active_indices: List of indices of models to use. 
                            If None, use all models.
                            Models not in this list will have their outputs zeroed.
        """
        batch_size = x.size(0)
        
        # 1. Get features from all models
        # If we only run active ones, we need to construct the full tensor with zeros.
        
        if active_indices is None:
            active_indices = list(range(self.num_models))
            
        # Optimization: Only run forward on active models
        # Construct a tensor of zeros for the full feature set
        all_features = torch.zeros(batch_size, self.num_models * self.feature_dim, device=x.device)
        
        num_active = len(active_indices)
        if num_active == 0:
            # Should not happen in practice, but return zeros or handle
            return self.classifier(all_features)
            
        for idx in active_indices:
            # Run model[idx]
            # We assume the base models are feature extractors (no head)
            # or we take the output before the head if they are full models.
            # Here we assume they return features.
            features = self.models[idx](x)
            
            # Place in the correct slice of all_features
            start_idx = idx * self.feature_dim
            end_idx = (idx + 1) * self.feature_dim
            all_features[:, start_idx:end_idx] = features
            
        # 2. Normalize by number of active clients
        # "Input values need to be divided by the number of active clients"
        all_features = all_features / num_active
        
        # 3. Pass through classifier
        output = self.classifier(all_features)
        
        return output

class ResNetFeatureExtractor(nn.Module):
    """Wrapper to use ResNet18 as a feature extractor."""
    def __init__(self, original_model: nn.Module):
        super().__init__()
        # Copy layers from original model, excluding the final fc
        self.features = nn.Sequential(
            original_model.conv1,
            original_model.bn1,
            original_model.relu,
            original_model.maxpool,
            original_model.layer1,
            original_model.layer2,
            original_model.layer3,
            original_model.layer4,
            original_model.avgpool
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x
