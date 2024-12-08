import torch
import torch.nn as nn
import numpy as np

class Pruner:
    def __init__(self, model, pruning_ratio=0.5, method='magnitude'):
        """
        Initialize the pruner
        Args:
            model: The neural network model to prune
            pruning_ratio: Percentage of weights to prune (0-1)
            method: Pruning method ('magnitude', 'random', or 'structured')
        """
        self.model = model
        self.pruning_ratio = pruning_ratio
        self.method = method
        self.masks = {}
        
    def compute_masks(self):
        """Compute binary masks for pruning"""
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                if self.method == 'magnitude':
                    # Magnitude-based pruning
                    threshold = np.percentile(abs(param.data.cpu().numpy()), 
                                           self.pruning_ratio * 100)
                    mask = torch.abs(param.data) > threshold
                    
                elif self.method == 'random':
                    # Random pruning
                    mask = torch.rand(param.shape) > self.pruning_ratio
                    
                elif self.method == 'structured':
                    # Structured pruning (remove entire filters/channels)
                    if len(param.shape) == 4:  # Conv layers
                        importance = torch.norm(param.data.view(param.shape[0], -1), p=1, dim=1)
                        num_keep = int(importance.shape[0] * (1 - self.pruning_ratio))
                        threshold = torch.sort(importance)[0][num_keep]
                        mask = importance.unsqueeze(1).unsqueeze(2).unsqueeze(3) > threshold
                    else:
                        mask = torch.ones_like(param.data, dtype=torch.bool)
                
                self.masks[name] = mask.to(param.device)
    
    def apply_masks(self):
        """Apply the computed masks to the model parameters"""
        for name, param in self.model.named_parameters():
            if 'weight' in name and name in self.masks:
                param.data *= self.masks[name].float()
    
    def get_model_size_stats(self, model, masks=None):
        total_params = 0
        nonzero_params = 0
        size_in_bytes = 0
        
        for name, param in model.named_parameters():
            param_size = param.numel()
            total_params += param_size
            
            if masks and name in masks:
                nonzero_params += torch.sum(masks[name]).item()
            else:
                nonzero_params += param_size
                
            # Calculate memory usage (assuming float32)
            size_in_bytes += param_size * 4  # 4 bytes per parameter
        
        return {
            'total_params': total_params,
            'nonzero_params': nonzero_params,
            'compression_ratio': total_params / nonzero_params if nonzero_params > 0 else 0,
            'size_mb': size_in_bytes / (1024 * 1024),
            'pruned_size_mb': (nonzero_params * 4) / (1024 * 1024)
        }


    def prune(self):
        """Compute and apply pruning masks"""
        print("\nBefore pruning:")
        before_stats = self.get_model_size_stats(self.model)
        print(f"Model size: {before_stats['size_mb']:.2f} MB")
        
        self.compute_masks()
        self.apply_masks()
        
        print("\nAfter pruning:")
        after_stats = self.get_model_size_stats(self.model, self.masks)
        print(f"Model size: {after_stats['pruned_size_mb']:.2f} MB")
        print(f"Compression ratio: {after_stats['compression_ratio']:.2f}x")
        print(f"Memory saved: {before_stats['size_mb'] - after_stats['pruned_size_mb']:.2f} MB")
        
    def get_sparsity(self):
        """Calculate the current sparsity ratio of the model"""
        total_params = 0
        zero_params = 0
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                total_params += param.numel()
                zero_params += (param.data == 0).sum().item()
        return zero_params / total_params if total_params > 0 else 0