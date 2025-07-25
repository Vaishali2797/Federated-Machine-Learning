#!/usr/bin/env python3
"""
Basic Federated Learning Example

This example demonstrates the fundamental concepts of federated learning using
a simple neural network on synthetic data. It's designed to be educational
and easy to understand.

Author: Federated Learning Framework Team
License: MIT
"""

import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
import time
import json

# Import federated learning framework components
# Note: These would be actual imports in a real implementation
class FLConfig:
    """Configuration class for federated learning parameters."""
    
    def __init__(self, **kwargs):
        # Core FL parameters
        self.num_clients = kwargs.get('num_clients', 10)
        self.num_rounds = kwargs.get('num_rounds', 20)
        self.local_epochs = kwargs.get('local_epochs', 5)
        self.client_selection_ratio = kwargs.get('client_selection_ratio', 0.5)
        
        # Model parameters
        self.model_config = kwargs.get('model_config', {
            'input_size': 784,
            'hidden_layers': [128, 64],
            'output_size': 10,
            'activation': 'relu'
        })
        
        # Training parameters
        self.learning_rate = kwargs.get('learning_rate', 0.01)
        self.batch_size = kwargs.get('batch_size', 32)
        
        # Data parameters
        self.data_distribution = kwargs.get('data_distribution', 'iid')
        self.alpha = kwargs.get('alpha', 0.5)  # For Dirichlet distribution
        
        # Logging
        self.verbose = kwargs.get('verbose', True)

class SimpleNeuralNetwork:
    """Simple neural network implementation for demonstration."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.weights = self._initialize_weights()
        
    def _initialize_weights(self) -> List[np.ndarray]:
        """Initialize network weights randomly."""
        weights = []
        layer_sizes = [self.config['input_size']] + self.config['hidden_layers'] + [self.config['output_size']]
        
        for i in range(len(layer_sizes) - 1):
            # Xavier initialization
            w = np.random.normal(0, np.sqrt(2.0 / layer_sizes[i]), 
                               (layer_sizes[i], layer_sizes[i + 1]))
            b = np.zeros((1, layer_sizes[i + 1]))
            weights.extend([w, b])
        
        return weights
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through the network."""
        activation = X
        
        for i in range(0, len(self.weights), 2):
            w, b = self.weights[i], self.weights[i + 1]
            z = np.dot(activation, w) + b
            
            # Apply activation function (ReLU for hidden layers, softmax for output)
            if i < len(self.weights) - 2:  # Hidden layers
                activation = np.maximum(0, z)  # ReLU
            else:  # Output layer
                activation = self._softmax(z)
        
        return activation
    
    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """Softmax activation function."""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, learning_rate: float) -> Dict[str, float]:
        """Train the neural network."""
        losses = []
        accuracies = []
        
        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(X)
            
            # Compute loss (cross-entropy)
            loss = self._cross_entropy_loss(predictions, y)
            losses.append(loss)
            
            # Compute accuracy
            accuracy = self._compute_accuracy(predictions, y)
            accuracies.append(accuracy)
            
            # Backward pass and update weights
            self._backward_pass(X, y, predictions, learning_rate)
        
        return {
            'loss': np.mean(losses),
            'accuracy': np.mean(accuracies),
            'final_loss': losses[-1],
            'final_accuracy': accuracies[-1]
        }
    
    def _cross_entropy_loss(self, predictions: np.ndarray, y: np.ndarray) -> float:
        """Compute cross-entropy loss."""
        n_samples = predictions.shape[0]
        y_one_hot = np.eye(self.config['output_size'])[y]
        loss = -np.sum(y_one_hot * np.log(predictions + 1e-15)) / n_samples
        return loss
    
    def _compute_accuracy(self, predictions: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy."""
        predicted_classes = np.argmax(predictions, axis=1)
        return np.mean(predicted_classes == y)
    
    def _backward_pass(self, X: np.ndarray, y: np.ndarray, predictions: np.ndarray, learning_rate: float):
        """Backward pass and weight update."""
        n_samples = X.shape[0]
        y_one_hot = np.eye(self.config['output_size'])[y]
        
        # This is a simplified backward pass for demonstration
        # In practice, you would implement full backpropagation
        
        # Output layer gradient
        output_error = predictions - y_one_hot
        
        # Update output layer weights (simplified)
        if len(self.weights) >= 2:
            # Get the last hidden layer activation (simplified)
            last_hidden = X  # This is oversimplified
            
            # Update weights and biases
            self.weights[-2] -= learning_rate * np.dot(last_hidden.T, output_error) / n_samples
            self.weights[-1] -= learning_rate * np.mean(output_error, axis=0, keepdims=True)
    
    def get_weights(self) -> List[np.ndarray]:
        """Get current model weights."""
        return [w.copy() for w in self.weights]
    
    def set_weights(self, weights: List[np.ndarray]):
        """Set model weights."""
        self.weights = [w.copy() for w in weights]

class FLClient:
    """Federated learning client."""
    
    def __init__(self, client_id: str, data: Tuple[np.ndarray, np.ndarray], model_config: Dict[str, Any]):
        self.client_id = client_id
        self.X_train, self.y_train = data
        self.model = SimpleNeuralNetwork(model_config)
        self.training_history = []
        
    def train_local_model(self, global_weights: List[np.ndarray], epochs: int, learning_rate: float) -> Dict[str, Any]:
        """Train local model on client's private data."""
        # Set global weights
        self.model.set_weights(global_weights)
        
        # Train locally
        training_metrics = self.model.train(self.X_train, self.y_train, epochs, learning_rate)
        
        # Store training history
        self.training_history.append(training_metrics)
        
        return {
            'client_id': self.client_id,
            'weights': self.model.get_weights(),
            'num_samples': len(self.X_train),
            'metrics': training_metrics
        }
    
    def get_data_size(self) -> int:
        """Get size of local dataset."""
        return len(self.X_train)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate local model on test data."""
        predictions = self.model.forward(X_test)
        accuracy = self.model._compute_accuracy(predictions, y_test)
        loss = self.model._cross_entropy_loss(predictions, y_test)
        
        return {'accuracy': accuracy, 'loss': loss}

class FLServer:
    """Federated learning server."""
    
    def __init__(self, model_config: Dict[str, Any]):
        self.global_model = SimpleNeuralNetwork(model_config)
        self.global_weights_history = []
        self.round_metrics = []
        
    def aggregate_updates(self, client_updates: List[Dict[str, Any]]) -> List[np.ndarray]:
        """Aggregate client updates using FedAvg algorithm."""
        # Calculate total number of samples
        total_samples = sum(update['num_samples'] for update in client_updates)
        
        # Initialize aggregated weights
        aggregated_weights = []
        for i, weight in enumerate(client_updates[0]['weights']):
            aggregated_weights.append(np.zeros_like(weight))
        
        # Weighted average of client weights
        for update in client_updates:
            weight_factor = update['num_samples'] / total_samples
            for i, weight in enumerate(update['weights']):
                aggregated_weights[i] += weight_factor * weight
        
        return aggregated_weights
    
    def get_global_weights(self) -> List[np.ndarray]:
        """Get current global model weights."""
        return self.global_model.get_weights()
    
    def update_global_model(self, aggregated_weights: List[np.ndarray]):
        """Update global model with aggregated weights."""
        self.global_model.set_weights(aggregated_weights)
        self.global_weights_history.append([w.copy() for w in aggregated_weights])
    
    def evaluate_global_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate global model on test data."""
        predictions = self.global_model.forward(X_test)
        accuracy = self.global_model._compute_accuracy(predictions, y_test)
        loss = self.global_model._cross_entropy_loss(predictions, y_test)
        
        return {'accuracy': accuracy, 'loss': loss}

class FederatedLearning:
    """Main federated learning orchestrator."""
    
    def __init__(self, config: FLConfig):
        self.config = config
        self.server = FLServer(config.model_config)
        self.clients = []
        self.training_results = {}
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        level = logging.INFO if self.config.verbose else logging.WARNING
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
    
    def add_client(self, client: FLClient):
        """Add a client to the federated learning system."""
        self.clients.append(client)
        self.logger.info(f"Added client {client.client_id} with {client.get_data_size()} samples")
    
    def create_mock_client(self, client_id: str) -> FLClient:
        """Create a mock client with synthetic data."""
        # Generate synthetic data for demonstration
        np.random.seed(hash(client_id) % 2**32)  # Deterministic but different per client
        
        n_samples = np.random.randint(100, 500)  # Random number of samples per client
        X = np.random.randn(n_samples, self.config.model_config['input_size'])
        
        # Create labels with some pattern
        weights = np.random.randn(self.config.model_config['input_size'])
        scores = np.dot(X, weights)
        y = (scores > np.median(scores)).astype(int)
        
        # Add some noise to make it more realistic
        noise_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
        y[noise_indices] = np.random.randint(0, self.config.model_config['output_size'], 
                                           size=len(noise_indices))
        
        return FLClient(client_id, (X, y), self.config.model_config)
    
    def select_clients(self, round_num: int) -> List[FLClient]:
        """Select clients for training round."""
        num_selected = max(1, int(len(self.clients) * self.config.client_selection_ratio))
        
        # Random selection for simplicity
        np.random.seed(round_num)  # Deterministic selection per round
        selected_indices = np.random.choice(len(self.clients), size=num_selected, replace=False)
        selected_clients = [self.clients[i] for i in selected_indices]
        
        self.logger.info(f"Round {round_num}: Selected {len(selected_clients)} clients: "
                        f"{[c.client_id for c in selected_clients]}")
        
        return selected_clients
    
    def start_training(self) -> Dict[str, Any]:
        """Start the federated learning training process."""
        self.logger.info("Starting federated learning training...")
        self.logger.info(f"Configuration: {self.config.num_clients} clients, "
                        f"{self.config.num_rounds} rounds, "
                        f"{self.config.local_epochs} local epochs")
        
        start_time = time.time()
        round_metrics = []
        
        # Create test data for evaluation
        X_test, y_test = self._create_test_data()
        
        for round_num in range(1, self.config.num_rounds + 1):
            self.logger.info(f"\n--- Round {round_num}/{self.config.num_rounds} ---")
            
            # Select clients for this round
            selected_clients = self.select_clients(round_num)
            
            # Get current global weights
            global_weights = self.server.get_global_weights()
            
            # Train selected clients
            client_updates = []
            for client in selected_clients:
                update = client.train_local_model(
                    global_weights, 
                    self.config.local_epochs, 
                    self.config.learning_rate
                )
                client_updates.append(update)
                
                self.logger.info(f"Client {client.client_id}: "
                               f"Loss={update['metrics']['final_loss']:.4f}, "
                               f"Accuracy={update['metrics']['final_accuracy']:.4f}")
            
            # Aggregate updates
            aggregated_weights = self.server.aggregate_updates(client_updates)
            
            # Update global model
            self.server.update_global_model(aggregated_weights)
            
            # Evaluate global model
            global_metrics = self.server.evaluate_global_model(X_test, y_test)
            
            # Store round metrics
            round_metrics.append({
                'round': round_num,
                'global_accuracy': global_metrics['accuracy'],
                'global_loss': global_metrics['loss'],
                'num_participants': len(selected_clients),
                'client_metrics': [update['metrics'] for update in client_updates]
            })
            
            self.logger.info(f"Global model: Loss={global_metrics['loss']:.4f}, "
                           f"Accuracy={global_metrics['accuracy']:.4f}")
        
        training_time = time.time() - start_time
        
        # Final evaluation
        final_metrics = self.server.evaluate_global_model(X_test, y_test)
        
        self.training_results = {
            'final_accuracy': final_metrics['accuracy'],
            'final_loss': final_metrics['loss'],
            'training_time': training_time,
            'round_metrics': round_metrics,
            'config': self.config.__dict__
        }
        
        self.logger.info(f"\nTraining completed!")
        self.logger.info(f"Final global accuracy: {final_metrics['accuracy']:.4f}")
        self.logger.info(f"Final global loss: {final_metrics['loss']:.4f}")
        self.logger.info(f"Total training time: {training_time:.2f} seconds")
        
        return self.training_results
    
    def _create_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create test data for evaluation."""
        np.random.seed(42)  # Fixed seed for consistent test data
        
        n_test = 1000
        X_test = np.random.randn(n_test, self.config.model_config['input_size'])
        
        # Create test labels with the same pattern as training data
        weights = np.random.randn(self.config.model_config['input_size'])
        scores = np.dot(X_test, weights)
        y_test = (scores > np.median(scores)).astype(int)
        
        return X_test, y_test
    
    def plot_training_progress(self, save_path: str = None):
        """Plot training progress visualization."""
        if not self.training_results:
            self.logger.warning("No training results available for plotting")
            return
        
        rounds = [m['round'] for m in self.training_results['round_metrics']]
        accuracies = [m['global_accuracy'] for m in self.training_results['round_metrics']]
        losses = [m['global_loss'] for m in self.training_results['round_metrics']]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot accuracy
        ax1.plot(rounds, accuracies, 'b-', marker='o', linewidth=2, markersize=4)
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Global Accuracy')
        ax1.set_title('Federated Learning: Global Model Accuracy')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Plot loss
        ax2.plot(rounds, losses, 'r-', marker='o', linewidth=2, markersize=4)
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Global Loss')
        ax2.set_title('Federated Learning: Global Model Loss')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training progress plot saved to {save_path}")
        
        plt.show()
    
    def save_results(self, filepath: str):
        """Save training results to JSON file."""
        if not self.training_results:
            self.logger.warning("No training results available to save")
            return
        
        with open(filepath, 'w') as f:
            json.dump(self.training_results, f, indent=2, default=str)
        
        self.logger.info(f"Training results saved to {filepath}")

def main():
    """Main function to run the basic federated learning example."""
    parser = argparse.ArgumentParser(description='Basic Federated Learning Example')
    parser.add_argument('--num_clients', type=int, default=10, 
                       help='Number of clients (default: 10)')
    parser.add_argument('--num_rounds', type=int, default=20, 
                       help='Number of training rounds (default: 20)')
    parser.add_argument('--local_epochs', type=int, default=5, 
                       help='Local epochs per round (default: 5)')
    parser.add_argument('--learning_rate', type=float, default=0.01, 
                       help='Learning rate (default: 0.01)')
    parser.add_argument('--client_selection_ratio', type=float, default=0.5, 
                       help='Fraction of clients selected per round (default: 0.5)')
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable verbose logging')
    parser.add_argument('--plot', action='store_true', 
                       help='Generate training progress plots')
    parser.add_argument('--save_results', type=str, 
                       help='Path to save training results JSON file')
    
    args = parser.parse_args()
    
    # Create configuration
    config = FLConfig(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        learning_rate=args.learning_rate,
        client_selection_ratio=args.client_selection_ratio,
        verbose=args.verbose,
        model_config={
            'input_size': 784,  # MNIST-like input size
            'hidden_layers': [128, 64],
            'output_size': 2,  # Binary classification for simplicity
            'activation': 'relu'
        }
    )
    
    # Initialize federated learning system
    fl_system = FederatedLearning(config)
    
    # Create mock clients
    print(f"Creating {config.num_clients} mock clients...")
    for i in range(config.num_clients):
        client = fl_system.create_mock_client(f"client_{i:03d}")
        fl_system.add_client(client)
    
    # Start training
    print("\nStarting federated learning training...")
    results = fl_system.start_training()
    
    # Display results
    print(f"\n{'='*50}")
    print("TRAINING COMPLETED")
    print(f"{'='*50}")
    print(f"Final Global Accuracy: {results['final_accuracy']:.4f}")
    print(f"Final Global Loss: {results['final_loss']:.4f}")
    print(f"Training Time: {results['training_time']:.2f} seconds")
    print(f"Average Time per Round: {results['training_time']/config.num_rounds:.2f} seconds")
    
    # Generate plots if requested
    if args.plot:
        fl_system.plot_training_progress()
    
    # Save results if requested
    if args.save_results:
        fl_system.save_results(args.save_results)
    
    print(f"\nExample completed successfully!")

if __name__ == "__main__":
    main()