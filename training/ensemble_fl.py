import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

from training.utils import get_model, train, evaluate
from training.ensemble_model import EnsembleModel, ResNetFeatureExtractor
from data.loader import _extract_targets

class EnsembleFedAvg:
    def __init__(
        self,
        train_subsets: List[Subset],
        test_set: Subset,
        num_clients: int,
        device: torch.device,
        model_name: str = "resnet18",
        pretrained: bool = False,
        num_clusters: int = 5, # Default, can be changed
        batch_size: int = 32,
        lr: float = 0.01,
        seed: int = 42,
    ):
        self.seed = seed
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        self.train_subsets = train_subsets
        self.test_set = test_set
        self.num_clients = num_clients
        self.device = device
        self.model_name = model_name
        self.pretrained = pretrained
        self.num_clusters = num_clusters
        self.batch_size = batch_size
        self.lr = lr
        
        # Initialize global model for warm-up
        self.global_model = get_model(
            model_name=self.model_name,
            pretrained=self.pretrained
        ).to(self.device)
        
        self.test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        
        # State variables
        self.client_clusters = None # Mapping client_idx -> cluster_idx
        self.cluster_models = [] # List of models, one per cluster
        self.ensemble_classifier = None # The shared FC layer
        self.feature_dim = 512 # ResNet18 default
        self.warmup_gradients = {} # Store gradients during warmup for clustering
        self.warmup_layer_gradients = {} # Store layer-wise gradients during warmup
    
    # * IMPORTANT: the warm up that we want is with fedavg or only training for 2 epochs from scratch ?
    def run_warmup(self, use_fedavg: bool = True, rounds: int = 1, local_epochs: int = 2, 
                   use_weight_diff: bool = False):
        """
        Run the warmup phase.
        
        Args:
            use_fedavg: If True, runs standard FedAvg for `rounds` rounds.
                        If False, runs single client finetuning for `local_epochs` epochs.
            rounds: Number of rounds for FedAvg warmup.
            local_epochs: Number of local epochs for training.
            use_weight_diff: If True, use weight differences (Δw = w_before - w_after) for clustering.
                           If False, use actual gradients from backpropagation.
        """
        self.warmup_gradients = {} # Reset gradients
        self.warmup_layer_gradients = {} # Reset layer-wise gradients
        
        if use_fedavg:
            print(f"Starting FedAvg Warmup for {rounds} rounds...")
            for r in range(1, rounds + 1):
                self.warmup_round(r, fraction=1.0, local_epochs=local_epochs)
        else:
            print(f"Starting Single Client Finetuning Warmup for {local_epochs} epochs...")
            # In this mode, we just train each client from the initial global model
            # and collect gradients. We do NOT update the global model.
            global_weights = self.global_model.state_dict()
            
            for client_idx in range(self.num_clients):
                local_model = copy.deepcopy(self.global_model)
                local_model.load_state_dict(global_weights)
                
                optimizer = optim.SGD(local_model.parameters(), lr=self.lr, momentum=0.9)
                criterion = nn.CrossEntropyLoss()
                train_loader = DataLoader(self.train_subsets[client_idx], batch_size=self.batch_size, shuffle=True)
                
                # Initialize gradient list for this client
                if client_idx not in self.warmup_gradients:
                    self.warmup_gradients[client_idx] = []
                if client_idx not in self.warmup_layer_gradients:
                    self.warmup_layer_gradients[client_idx] = []
                
                if use_weight_diff:
                    # Use weight difference method
                    initial_weights = copy.deepcopy(local_model.state_dict())
                    
                    # Train for local_epochs
                    for epoch in range(local_epochs):
                        local_model.train()
                        for inputs, labels in train_loader:
                            inputs, labels = inputs.to(self.device), labels.to(self.device)
                            optimizer.zero_grad()
                            outputs = local_model(inputs)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            optimizer.step()
                    
                    # Compute weight differences as proxy for gradients
                    final_weights = local_model.state_dict()
                    
                    # Flatten weight differences for selected layers
                    diff_vec = []
                    layer_diffs = {}
                    for key in initial_weights:
                        if 'num_batches_tracked' in key:
                            continue
                        if initial_weights[key].dtype == torch.float:
                            if 'layer4' in key or 'fc' in key:  # Same layer selection
                                diff = (initial_weights[key] - final_weights[key]).view(-1)
                                diff_vec.append(diff)
                                layer_diffs[key] = diff.cpu().numpy()
                    
                    if diff_vec:
                        flat_diff = torch.cat(diff_vec).cpu().numpy()
                        self.warmup_gradients[client_idx].append(flat_diff)
                    
                    if layer_diffs:
                        self.warmup_layer_gradients[client_idx].append(layer_diffs)
                
                else:
                    # Use actual gradient method (original)
                    # Train epoch by epoch to collect gradients for each epoch
                    local_model.train()
                    for epoch in range(local_epochs):
                        epoch_grad_accumulator = None
                        epoch_layer_grads = {}  # Layer-wise gradients for this epoch
                        num_batches = 0
                        
                        for inputs, labels in train_loader:
                            inputs, labels = inputs.to(self.device), labels.to(self.device)
                            
                            optimizer.zero_grad()
                            outputs = local_model(inputs)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            
                            # Accumulate gradients (flat) - LAYER4 + FC for feature-level clustering
                            current_batch_grad = []
                            for name, p in local_model.named_parameters():
                                # Include layer4 and FC to capture both feature and task-level heterogeneity
                                if p.grad is not None and ('layer4' in name or 'fc' in name):
                                    current_batch_grad.append(p.grad.view(-1))
                            
                            if current_batch_grad:
                                flat_batch_grad = torch.cat(current_batch_grad)
                                if epoch_grad_accumulator is None:
                                    epoch_grad_accumulator = flat_batch_grad
                                else:
                                    epoch_grad_accumulator += flat_batch_grad
                            
                            # Accumulate layer-wise gradients - LAYER4 + FC
                            for name, param in local_model.named_parameters():
                                if param.grad is not None and ('layer4' in name or 'fc' in name):
                                    grad_flat = param.grad.view(-1).clone()
                                    if name not in epoch_layer_grads:
                                        epoch_layer_grads[name] = grad_flat
                                    else:
                                        epoch_layer_grads[name] += grad_flat
                            
                            optimizer.step()
                            num_batches += 1
                        
                        # Average and store gradient for this epoch
                        if epoch_grad_accumulator is not None and num_batches > 0:
                            avg_epoch_grad = epoch_grad_accumulator / num_batches
                            self.warmup_gradients[client_idx].append(avg_epoch_grad.cpu().numpy())
                        
                        # Average and store layer-wise gradients for this epoch
                        if epoch_layer_grads and num_batches > 0:
                            avg_layer_grads = {}
                            for name, grad in epoch_layer_grads.items():
                                avg_layer_grads[name] = (grad / num_batches).cpu().numpy()
                            self.warmup_layer_gradients[client_idx].append(avg_layer_grads)
                
                print(f"Client {client_idx} finetuning complete. Collected {len(self.warmup_gradients[client_idx])} gradients.")
            
            print(f"Single Client Finetuning Warmup Complete. Gradients collected.")

    def save_warmup_model(self, path: str):
        """
        Save the global model state dict to the specified path.
        Useful for saving the model after warmup.
        """
        if self.global_model:
            torch.save(self.global_model.state_dict(), path)
            print(f"Warmup model saved to {path}")
        else:
            print("No global model to save (it might have been deleted after ensemble initialization).")

    
    def _compute_client_gradients(self) -> List[np.ndarray]:
        """
        Compute gradients for all clients.
        Returns list of flattened gradient vectors.
        """
        gradients = []
        global_weights = self.global_model.state_dict()
        
        for client_idx in range(self.num_clients):
            local_model = copy.deepcopy(self.global_model)
            local_model.load_state_dict(global_weights)
            local_model.to(self.device)
            local_model.train()
            
            train_loader = DataLoader(self.train_subsets[client_idx], batch_size=self.batch_size, shuffle=True)
            
            optimizer = optim.SGD(local_model.parameters(), lr=self.lr)
            criterion = nn.CrossEntropyLoss()
            
            # Train for 1 epoch and take the difference in weights as proxy for "average gradient"
            initial_weights = copy.deepcopy(local_model.state_dict())
            train(local_model, train_loader, optimizer, criterion, self.device, epochs=1)
            final_weights = local_model.state_dict()
            
            # Flatten weight difference
            diff_vec = []
            for key in initial_weights:
                if 'num_batches_tracked' in key: continue # Skip batchnorm stats
                if initial_weights[key].dtype == torch.float: # Only float params
                    diff = (initial_weights[key] - final_weights[key]).view(-1)
                    diff_vec.append(diff)
            
            flat_diff = torch.cat(diff_vec).cpu().numpy()
            gradients.append(flat_diff)
        
        return gradients
    
    def compute_feature_representations(self, num_samples: int = 100) -> np.ndarray:
        """
        Compute average feature representations for each client.
        Better for clustering when labels are IID but features are non-IID.
        
        Args:
            num_samples: Number of samples to use per client for feature extraction
            
        Returns:
            Array of shape (num_clients, feature_dim) with average features per client
        """
        feature_extractor = ResNetFeatureExtractor(self.global_model)
        feature_extractor.eval()
        
        client_features = []
        
        for client_idx in range(self.num_clients):
            # Get a subset of client data
            subset_size = min(num_samples, len(self.train_subsets[client_idx]))
            indices = np.random.choice(len(self.train_subsets[client_idx]), subset_size, replace=False)
            
            features_list = []
            
            with torch.no_grad():
                for idx in indices:
                    data_idx = self.train_subsets[client_idx].indices[idx]
                    x, _ = self.train_subsets[client_idx].dataset[data_idx]
                    
                    # Apply transforms if needed
                    if hasattr(self.train_subsets[client_idx].dataset, 'transform') and \
                       self.train_subsets[client_idx].dataset.transform is not None:
                        x = self.train_subsets[client_idx].dataset.transform(x)
                    
                    x = x.unsqueeze(0).to(self.device)
                    features = feature_extractor(x)
                    features_list.append(features.cpu().numpy())
            
            # Average features for this client
            avg_features = np.mean(features_list, axis=0).flatten()
            client_features.append(avg_features)
            
            if client_idx % 10 == 0:
                print(f"Extracted features for client {client_idx}/{self.num_clients}")
        
        return np.array(client_features)
    
    def get_client_gradients(self, average_across_rounds: bool = True) -> Dict[int, np.ndarray]:
        """
        Return the gradients computed during warmup for each client.
        
        Args:
            average_across_rounds: If True, return averaged gradients across all warmup rounds.
                                  If False, return list of gradients for each round.
        
        Returns:
            Dictionary mapping client_idx -> gradient array (or list of arrays if not averaged)
        """
        if not self.warmup_gradients:
            raise ValueError("No warmup gradients available. Run warmup_round() first.")
        
        result = {}
        for client_idx in range(self.num_clients):
            if client_idx in self.warmup_gradients:
                if average_across_rounds:
                    # Return averaged gradient across all warmup rounds
                    result[client_idx] = np.mean(self.warmup_gradients[client_idx], axis=0)
                else:
                    # Return list of gradients from each round
                    result[client_idx] = self.warmup_gradients[client_idx]
            else:
                print(f"Warning: Client {client_idx} has no warmup gradients")
        
        return result

    def get_client_layer_gradients(self, average_across_epochs: bool = True) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Return the layer-wise gradients computed during warmup for each client.
        
        Args:
            average_across_epochs: If True, return averaged gradients across all epochs.
                                  If False, return list of gradient dicts for each epoch.
        
        Returns:
            Dictionary mapping client_idx -> {layer_name: gradient_array} (or list of dicts if not averaged)
        """
        if not self.warmup_layer_gradients:
            raise ValueError("No layer-wise warmup gradients available. Run run_warmup(use_fedavg=False) first.")
        
        result = {}
        for client_idx in range(self.num_clients):
            if client_idx in self.warmup_layer_gradients:
                if average_across_epochs:
                    # Average gradients across all epochs for each layer
                    layer_grads = self.warmup_layer_gradients[client_idx]
                    if layer_grads:
                        avg_layer_grads = {}
                        layer_names = layer_grads[0].keys()
                        for name in layer_names:
                            # Stack gradients from all epochs and average
                            stacked = np.stack([epoch_grads[name] for epoch_grads in layer_grads])
                            avg_layer_grads[name] = np.mean(stacked, axis=0)
                        result[client_idx] = avg_layer_grads
                    else:
                        result[client_idx] = {}
                else:
                    # Return list of gradient dicts from each epoch
                    result[client_idx] = self.warmup_layer_gradients[client_idx]
            else:
                print(f"Warning: Client {client_idx} has no layer-wise warmup gradients")
        
        return result

    def _initialize_ensemble(self):
        """Initialize cluster-specific models and the global ensemble classifier."""
        # 1. Create feature extractors for each cluster, initialized from global model
        self.cluster_models = []
        base_feature_extractor = ResNetFeatureExtractor(self.global_model)
        
        for _ in range(self.num_clusters):
            # Deep copy the feature extractor
            self.cluster_models.append(copy.deepcopy(base_feature_extractor))
            
        # 2. Initialize Global Classifier
        # The global model has a 'fc' layer. We want to expand it.
        # Old FC: 512 -> 10
        # New FC: (512 * K) -> 10
        # We copy the weights K times.
        
        old_fc = self.global_model.fc
        new_fc = nn.Linear(self.feature_dim * self.num_clusters, 10)
        
        with torch.no_grad():
            for i in range(self.num_clusters):
                start_col = i * self.feature_dim
                end_col = (i + 1) * self.feature_dim
                # Copy weights
                new_fc.weight[:, start_col:end_col] = old_fc.weight
            
            # Bias: We can divide bias by K or just keep it?
            # If y = Wx + b. Now y = sum(Wx/K) + b_new.
            # sum(Wx/K) = Wx. So b_new should be b.
            new_fc.bias.copy_(old_fc.bias)
            
        self.ensemble_classifier = new_fc.to(self.device)
        
        # Clean up global model to save memory
        del self.global_model
        
    def train_ensemble_round(self, round_num: int, fraction: float = 0.1, local_epochs: int = 1) -> Tuple[float, float]:
        """
        Train using the ensemble strategy.
        Clients train their specific cluster model + the global classifier (masked).
        """
        m = max(int(fraction * self.num_clients), 1)
        selected_clients = random.sample(range(self.num_clients), m)
        
        cluster_updates = {i: [] for i in range(self.num_clusters)} # List of (weights, size)
        classifier_updates = [] # List of (weights, size)
        
        print(f"Ensemble Round {round_num}: Training on {m} clients...")
        
        global_classifier_weights = self.ensemble_classifier.state_dict()
        
        for client_idx in selected_clients:
            cluster_id = self.client_clusters[client_idx]
            
            # Prepare Client Model
            # Client gets: FeatureExtractor[cluster_id] AND GlobalClassifier
            # But we need to wrap them in a way that the client can train.
            # We can create a temporary "ClientEnsemble" model.
            
            feature_extractor = copy.deepcopy(self.cluster_models[cluster_id])
            classifier = copy.deepcopy(self.ensemble_classifier)
            classifier.load_state_dict(global_classifier_weights)
            
            # We need a model that:
            # 1. Takes x
            # 2. Extracts features using feature_extractor
            # 3. Pads features to full size (zeros for other clusters)
            # 4. Normalizes (divide by 1, since only 1 active)
            # 5. Runs classifier
            
            client_model = ClientEnsembleWrapper(
                feature_extractor, 
                classifier, 
                cluster_id, 
                self.num_clusters, 
                self.feature_dim
            ).to(self.device)
            
            optimizer = optim.SGD(client_model.parameters(), lr=self.lr, momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            train_loader = DataLoader(self.train_subsets[client_idx], batch_size=self.batch_size, shuffle=True)
            
            train(client_model, train_loader, optimizer, criterion, self.device, epochs=local_epochs)
            
            # Collect updates
            # 1. Feature Extractor
            cluster_updates[cluster_id].append((
                client_model.feature_extractor.state_dict(),
                len(self.train_subsets[client_idx])
            ))
            
            # 2. Classifier
            classifier_updates.append((
                client_model.classifier.state_dict(),
                len(self.train_subsets[client_idx])
            ))
            
        # Aggregate
        # 1. Cluster Models
        for cid in range(self.num_clusters):
            updates = cluster_updates[cid]
            if updates:
                weights = [u[0] for u in updates]
                sizes = [u[1] for u in updates]
                avg_weights = self._aggregate(weights, sizes)
                self.cluster_models[cid].load_state_dict(avg_weights)
                
        # 2. Global Classifier
        if classifier_updates:
            weights = [u[0] for u in classifier_updates]
            sizes = [u[1] for u in classifier_updates]
            avg_weights = self._aggregate(weights, sizes)
            self.ensemble_classifier.load_state_dict(avg_weights)
            
        # Evaluate Ensemble
        loss, acc = self.evaluate_ensemble()
        print(f"Ensemble Round {round_num} Result: Loss: {loss:.4f}, Accuracy: {acc:.4f}")
        return loss, acc

    def evaluate_ensemble(self) -> Tuple[float, float]:
        """Evaluate the full ensemble on test set."""
        # Construct full ensemble model
        full_model = EnsembleModel(
            self.cluster_models, 
            self.feature_dim, 
            10
        )
        # Set the classifier
        full_model.classifier = self.ensemble_classifier
        full_model.to(self.device)
        
        return evaluate(full_model, self.test_loader, nn.CrossEntropyLoss(), self.device)

    def _aggregate(self, weights: List[Dict], sizes: List[int]) -> Dict:
        """Compute weighted average of weights."""
        total_size = sum(sizes)
        avg_weights = copy.deepcopy(weights[0])
        
        for key in avg_weights.keys():
            avg_weights[key] = torch.zeros_like(avg_weights[key], dtype=torch.float)
            for i in range(len(weights)):
                avg_weights[key] += weights[i][key] * sizes[i]
            avg_weights[key] = torch.div(avg_weights[key], total_size)
            
        return avg_weights

class ClientEnsembleWrapper(nn.Module):
    """
    Helper model for client training.
    Represents the path through one specific cluster model + the global classifier.
    """
    def __init__(self, feature_extractor, classifier, cluster_id, num_clusters, feature_dim):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.cluster_id = cluster_id
        self.num_clusters = num_clusters
        self.feature_dim = feature_dim
        
    def forward(self, x):
        features = self.feature_extractor(x) # (B, feature_dim)
        
        # Construct full feature vector
        batch_size = x.size(0)
        full_features = torch.zeros(batch_size, self.num_clusters * self.feature_dim, device=x.device)
        
        start_idx = self.cluster_id * self.feature_dim
        end_idx = (self.cluster_id + 1) * self.feature_dim
        
        full_features[:, start_idx:end_idx] = features
        
        # Normalize (Active clients = 1)
        # full_features = full_features / 1.0 
        
        output = self.classifier(full_features)
        return output
