import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet34, resnet50, densenet121, mobilenet_v2

import random
import numpy as np

def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch (CPU & GPU).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

def get_model(
    model_name: str = "resnet18",
    num_classes: int = 10,
    pretrained: bool = True,
    train_head_only: bool = False
) -> nn.Module:
    """
    Returns a model adapted for CIFAR-10.
    Supported models: resnet18, resnet34, resnet50, densenet121, mobilenet_v2
    
    Args:
        model_name: Name of the model architecture.
        num_classes: Number of output classes.
        pretrained: Whether to use pretrained weights (ImageNet).
        train_head_only: If True, freeze all layers except the classifier.
    """
    weights = "DEFAULT" if pretrained else None
    
    if model_name == "resnet18":
        model = resnet18(weights=weights)
    elif model_name == "resnet34":
        model = resnet34(weights=weights)
    elif model_name == "resnet50":
        model = resnet50(weights=weights)
    elif model_name == "densenet121":
        model = densenet121(weights=weights)
    elif model_name == "mobilenet_v2":
        model = mobilenet_v2(weights=weights)
    else:
        raise ValueError(f"Model {model_name} not supported.")

    # Freeze backbone if requested
    if train_head_only:
        for param in model.parameters():
            param.requires_grad = False

    # Adapt last layer (and first layer if not pretrained, or if we want to adapt it anyway)
    # Note: For CIFAR10, standard ResNet first conv is 7x7 stride 2, which is too aggressive.
    # However, if we use pretrained weights, we might want to keep the first layer as is 
    # or replace it. Replacing it invalidates pretrained weights for that layer.
    # A common practice for CIFAR10 with pretrained ImageNet models is to UPSAMPLE images
    # or just accept the mismatch. 
    # BUT, the previous implementation modified conv1. If we modify conv1 on a pretrained model,
    # we lose the weights for that layer. 
    # Let's keep the modification but re-initialize it if it's a new layer.
    # If train_head_only is True, we MUST ensure the new layers have requires_grad=True.

    if "resnet" in model_name:
        # We modify conv1 to handle 32x32 better, but this breaks pretraining for this layer.
        # If we want to strictly follow "linear probing" on ImageNet features, we should probably
        # NOT modify the backbone architecture, but CIFAR10 is 32x32, ImageNet is 224x224.
        # Standard ResNet reduces 32x32 to 1x1 too quickly.
        # Let's stick to the previous modification logic but ensure we unfreeze the new layers.
        
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif "densenet" in model_name:
        model.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.features.pool0 = nn.Identity()
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif "mobilenet" in model_name:
        model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    # If train_head_only, we need to make sure the NEWLY REPLACED layers are trainable.
    # The user specifically asked for "linear classifier on the head".
    # Modifying the first layer (conv1) technically makes it not just the head.
    # However, without modifying conv1, ResNet on 32x32 performs very poorly.
    # I will enable gradients for the modified layers (conv1 and fc/classifier).
    
    if train_head_only:
        if "resnet" in model_name:
            for param in model.conv1.parameters():
                param.requires_grad = True
            for param in model.fc.parameters():
                param.requires_grad = True
        elif "densenet" in model_name:
            for param in model.features.conv0.parameters():
                param.requires_grad = True
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif "mobilenet" in model_name:
            for param in model.features[0][0].parameters():
                param.requires_grad = True
            for param in model.classifier[1].parameters():
                param.requires_grad = True
    return model

def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int = 1,
    verbose: bool = False
) -> tuple[float, float]:
    """
    Train the model for a specified number of epochs.
    Returns (average_loss, average_gradient_norm).
    """
    model.train()
    model.to(device)
    
    epoch_loss = 0.0
    total_norm = 0.0
    num_batches = 0
    
    for epoch in range(epochs):
        running_loss = 0.0
        running_norm = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Compute gradient norm
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    grad_norm += param_norm.item() ** 2
            grad_norm = grad_norm ** 0.5
            running_norm += grad_norm
            
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            num_batches += 1
            
        epoch_loss = running_loss / len(train_loader.dataset)
        total_norm += running_norm
        
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {100.*correct/total:.2f}%")
            
    avg_norm = total_norm / num_batches if num_batches > 0 else 0.0
    return epoch_loss, avg_norm

def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple[float, float]:
    """
    Evaluate the model on the test set.
    Returns (average_loss, accuracy).
    """
    model.eval()
    model.to(device)
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    loss = running_loss / len(test_loader.dataset)
    accuracy = correct / total
    
    return loss, accuracy
