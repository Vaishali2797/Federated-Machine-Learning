# Federated Learning API Documentation

## Table of Contents

1. [Overview](#overview)
2. [Core APIs](#core-apis)
3. [Client APIs](#client-apis)
4. [Server APIs](#server-apis)
5. [Model APIs](#model-apis)
6. [Communication APIs](#communication-apis)
7. [Security APIs](#security-apis)
8. [Utility Functions](#utility-functions)
9. [Examples](#examples)
10. [Error Handling](#error-handling)

## Overview

This document provides comprehensive documentation for the Federated Learning framework APIs. The framework is designed to enable distributed machine learning across multiple clients while preserving data privacy.

### Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FL Client 1   │    │   FL Client 2   │    │   FL Client N   │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ Local Model │ │    │ │ Local Model │ │    │ │ Local Model │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   FL Server     │
                    │                 │
                    │ ┌─────────────┐ │
                    │ │Global Model │ │
                    │ └─────────────┘ │
                    └─────────────────┘
```

## Core APIs

### FederatedLearning

Main orchestrator class for federated learning operations.

#### `class FederatedLearning`

**Description**: Central coordinator for federated learning processes.

**Constructor**:
```python
def __init__(self, config: FLConfig, logger: Optional[Logger] = None)
```

**Parameters**:
- `config` (FLConfig): Configuration object containing FL parameters
- `logger` (Optional[Logger]): Custom logger instance

**Example**:
```python
from federated_learning import FederatedLearning, FLConfig

config = FLConfig(
    num_clients=10,
    num_rounds=100,
    local_epochs=5,
    learning_rate=0.01
)

fl_system = FederatedLearning(config)
```

#### Methods

##### `start_training()`

**Description**: Initiates the federated learning training process.

**Signature**:
```python
def start_training(self) -> TrainingResults
```

**Returns**: 
- `TrainingResults`: Object containing training metrics and final model

**Raises**:
- `FLConfigError`: If configuration is invalid
- `ClientConnectionError`: If unable to connect to required clients

**Example**:
```python
results = fl_system.start_training()
print(f"Final accuracy: {results.final_accuracy}")
print(f"Training time: {results.training_time}")
```

##### `add_client()`

**Description**: Registers a new client to the federated learning system.

**Signature**:
```python
def add_client(self, client: FLClient) -> bool
```

**Parameters**:
- `client` (FLClient): Client instance to add

**Returns**: 
- `bool`: True if client was successfully added

**Example**:
```python
client = FLClient(client_id="client_001", data_path="/path/to/data")
success = fl_system.add_client(client)
```

## Client APIs

### FLClient

Represents a federated learning client that participates in training.

#### `class FLClient`

**Description**: Individual client in the federated learning network.

**Constructor**:
```python
def __init__(self, client_id: str, data_path: str, model_config: ModelConfig)
```

**Parameters**:
- `client_id` (str): Unique identifier for the client
- `data_path` (str): Path to local training data
- `model_config` (ModelConfig): Configuration for the local model

#### Methods

##### `train_local_model()`

**Description**: Trains the local model on client's private data.

**Signature**:
```python
def train_local_model(self, global_weights: np.ndarray, epochs: int = 1) -> LocalTrainingResult
```

**Parameters**:
- `global_weights` (np.ndarray): Current global model weights
- `epochs` (int): Number of local training epochs

**Returns**: 
- `LocalTrainingResult`: Training results including updated weights and metrics

**Example**:
```python
result = client.train_local_model(global_weights, epochs=5)
print(f"Local loss: {result.loss}")
print(f"Local accuracy: {result.accuracy}")
```

##### `get_data_size()`

**Description**: Returns the size of local dataset.

**Signature**:
```python
def get_data_size(self) -> int
```

**Returns**: 
- `int`: Number of samples in local dataset

## Server APIs

### FLServer

Manages the global model and coordinates client training.

#### `class FLServer`

**Description**: Central server that orchestrates federated learning.

**Constructor**:
```python
def __init__(self, model_config: ModelConfig, aggregation_strategy: str = "fedavg")
```

**Parameters**:
- `model_config` (ModelConfig): Global model configuration
- `aggregation_strategy` (str): Strategy for aggregating client updates

#### Methods

##### `aggregate_updates()`

**Description**: Aggregates model updates from clients.

**Signature**:
```python
def aggregate_updates(self, client_updates: List[ClientUpdate]) -> np.ndarray
```

**Parameters**:
- `client_updates` (List[ClientUpdate]): List of client model updates

**Returns**: 
- `np.ndarray`: Aggregated global model weights

**Example**:
```python
updates = [client1.get_update(), client2.get_update()]
new_global_weights = server.aggregate_updates(updates)
```

##### `select_clients()`

**Description**: Selects subset of clients for next training round.

**Signature**:
```python
def select_clients(self, selection_ratio: float = 0.1) -> List[str]
```

**Parameters**:
- `selection_ratio` (float): Fraction of clients to select

**Returns**: 
- `List[str]`: List of selected client IDs

## Model APIs

### FLModel

Base class for federated learning models.

#### `class FLModel`

**Description**: Abstract base class for models used in federated learning.

**Methods**:

##### `get_weights()`

**Description**: Retrieves current model weights.

**Signature**:
```python
def get_weights(self) -> np.ndarray
```

**Returns**: 
- `np.ndarray`: Current model weights

##### `set_weights()`

**Description**: Sets model weights.

**Signature**:
```python
def set_weights(self, weights: np.ndarray) -> None
```

**Parameters**:
- `weights` (np.ndarray): New model weights

##### `train()`

**Description**: Trains the model on provided data.

**Signature**:
```python
def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1) -> TrainingMetrics
```

**Parameters**:
- `X` (np.ndarray): Training features
- `y` (np.ndarray): Training labels
- `epochs` (int): Number of training epochs

**Returns**: 
- `TrainingMetrics`: Training metrics (loss, accuracy, etc.)

### Neural Network Models

#### `class FLNeuralNetwork(FLModel)`

**Description**: Neural network implementation for federated learning.

**Constructor**:
```python
def __init__(self, layers: List[int], activation: str = "relu", optimizer: str = "adam")
```

**Parameters**:
- `layers` (List[int]): List of layer sizes
- `activation` (str): Activation function
- `optimizer` (str): Optimization algorithm

**Example**:
```python
model = FLNeuralNetwork(
    layers=[784, 128, 64, 10],
    activation="relu",
    optimizer="adam"
)
```

## Communication APIs

### CommunicationManager

Handles communication between clients and server.

#### `class CommunicationManager`

**Description**: Manages network communication in federated learning.

**Constructor**:
```python
def __init__(self, protocol: str = "grpc", encryption: bool = True)
```

**Parameters**:
- `protocol` (str): Communication protocol ("grpc", "http", "socket")
- `encryption` (bool): Whether to encrypt communications

#### Methods

##### `send_model_update()`

**Description**: Sends model update from client to server.

**Signature**:
```python
def send_model_update(self, client_id: str, update: ModelUpdate) -> bool
```

**Parameters**:
- `client_id` (str): ID of sending client
- `update` (ModelUpdate): Model update to send

**Returns**: 
- `bool`: True if update was successfully sent

##### `broadcast_global_model()`

**Description**: Broadcasts global model to selected clients.

**Signature**:
```python
def broadcast_global_model(self, model_weights: np.ndarray, client_ids: List[str]) -> Dict[str, bool]
```

**Parameters**:
- `model_weights` (np.ndarray): Global model weights
- `client_ids` (List[str]): List of target client IDs

**Returns**: 
- `Dict[str, bool]`: Success status for each client

## Security APIs

### PrivacyManager

Implements differential privacy and secure aggregation.

#### `class PrivacyManager`

**Description**: Handles privacy-preserving mechanisms in federated learning.

**Constructor**:
```python
def __init__(self, privacy_budget: float = 1.0, noise_multiplier: float = 1.0)
```

**Parameters**:
- `privacy_budget` (float): Total privacy budget (epsilon)
- `noise_multiplier` (float): Noise scaling factor

#### Methods

##### `add_noise()`

**Description**: Adds differential privacy noise to model updates.

**Signature**:
```python
def add_noise(self, weights: np.ndarray, sensitivity: float) -> np.ndarray
```

**Parameters**:
- `weights` (np.ndarray): Original model weights
- `sensitivity` (float): Sensitivity of the query

**Returns**: 
- `np.ndarray`: Noisy model weights

**Example**:
```python
privacy_manager = PrivacyManager(privacy_budget=1.0)
noisy_weights = privacy_manager.add_noise(weights, sensitivity=0.1)
```

##### `secure_aggregation()`

**Description**: Performs secure aggregation of client updates.

**Signature**:
```python
def secure_aggregation(self, encrypted_updates: List[EncryptedUpdate]) -> np.ndarray
```

**Parameters**:
- `encrypted_updates` (List[EncryptedUpdate]): List of encrypted client updates

**Returns**: 
- `np.ndarray`: Aggregated weights without revealing individual updates

## Utility Functions

### Data Utilities

#### `load_federated_data()`

**Description**: Loads and partitions data for federated learning.

**Signature**:
```python
def load_federated_data(
    data_path: str, 
    num_clients: int, 
    distribution: str = "iid"
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]
```

**Parameters**:
- `data_path` (str): Path to dataset
- `num_clients` (int): Number of clients to partition data for
- `distribution` (str): Data distribution strategy ("iid", "non_iid")

**Returns**: 
- `Dict[str, Tuple[np.ndarray, np.ndarray]]`: Client ID mapped to (X, y) data

**Example**:
```python
client_data = load_federated_data(
    data_path="./data/mnist.npz",
    num_clients=10,
    distribution="non_iid"
)
```

#### `evaluate_global_model()`

**Description**: Evaluates global model performance.

**Signature**:
```python
def evaluate_global_model(
    model: FLModel, 
    test_data: Tuple[np.ndarray, np.ndarray]
) -> EvaluationMetrics
```

**Parameters**:
- `model` (FLModel): Model to evaluate
- `test_data` (Tuple[np.ndarray, np.ndarray]): Test dataset (X, y)

**Returns**: 
- `EvaluationMetrics`: Evaluation results (accuracy, loss, F1-score, etc.)

### Configuration Utilities

#### `class FLConfig`

**Description**: Configuration container for federated learning parameters.

**Attributes**:
- `num_clients` (int): Total number of clients
- `num_rounds` (int): Number of federated learning rounds
- `local_epochs` (int): Local training epochs per round
- `learning_rate` (float): Learning rate for local training
- `batch_size` (int): Batch size for local training
- `client_selection_ratio` (float): Fraction of clients selected per round

**Example**:
```python
config = FLConfig(
    num_clients=100,
    num_rounds=50,
    local_epochs=3,
    learning_rate=0.001,
    batch_size=32,
    client_selection_ratio=0.1
)
```

## Examples

### Basic Federated Learning Setup

```python
from federated_learning import FederatedLearning, FLConfig, FLClient, FLNeuralNetwork

# 1. Create configuration
config = FLConfig(
    num_clients=10,
    num_rounds=20,
    local_epochs=5,
    learning_rate=0.01,
    client_selection_ratio=0.5
)

# 2. Initialize federated learning system
fl_system = FederatedLearning(config)

# 3. Create and add clients
for i in range(10):
    model = FLNeuralNetwork(layers=[784, 128, 10])
    client = FLClient(
        client_id=f"client_{i}",
        data_path=f"./data/client_{i}",
        model_config=model.get_config()
    )
    fl_system.add_client(client)

# 4. Start training
results = fl_system.start_training()

# 5. Evaluate results
print(f"Final global accuracy: {results.final_accuracy}")
print(f"Training completed in {results.training_time} seconds")
```

### Custom Aggregation Strategy

```python
from federated_learning import FLServer, CustomAggregator

class WeightedAverageAggregator(CustomAggregator):
    def aggregate(self, client_updates):
        total_samples = sum(update.num_samples for update in client_updates)
        weighted_sum = np.zeros_like(client_updates[0].weights)
        
        for update in client_updates:
            weight = update.num_samples / total_samples
            weighted_sum += weight * update.weights
            
        return weighted_sum

# Use custom aggregator
server = FLServer(
    model_config=model_config,
    aggregation_strategy=WeightedAverageAggregator()
)
```

### Differential Privacy Example

```python
from federated_learning import PrivacyManager, FLClient

# Initialize privacy manager
privacy_manager = PrivacyManager(
    privacy_budget=1.0,
    noise_multiplier=0.5
)

# Client with differential privacy
class PrivateClient(FLClient):
    def get_update(self):
        update = super().get_update()
        # Add noise to preserve privacy
        update.weights = privacy_manager.add_noise(
            update.weights, 
            sensitivity=0.1
        )
        return update
```

### Non-IID Data Distribution

```python
from federated_learning.utils import create_non_iid_distribution

# Create non-IID data distribution
client_data = create_non_iid_distribution(
    dataset="cifar10",
    num_clients=20,
    classes_per_client=2,  # Each client has data from only 2 classes
    alpha=0.5  # Dirichlet distribution parameter
)

# Initialize clients with non-IID data
for client_id, (X, y) in client_data.items():
    client = FLClient(client_id, data=(X, y))
    fl_system.add_client(client)
```

## Error Handling

### Exception Classes

#### `FLError`
Base exception class for federated learning errors.

#### `FLConfigError`
Raised when configuration parameters are invalid.

#### `ClientConnectionError`
Raised when unable to establish connection with client.

#### `ModelConvergenceError`
Raised when model fails to converge within specified rounds.

#### `PrivacyBudgetExhaustedError`
Raised when differential privacy budget is exhausted.

### Error Handling Examples

```python
try:
    fl_system.start_training()
except FLConfigError as e:
    print(f"Configuration error: {e}")
    # Handle configuration issues
except ClientConnectionError as e:
    print(f"Client connection failed: {e}")
    # Retry or remove problematic clients
except ModelConvergenceError as e:
    print(f"Model convergence failed: {e}")
    # Adjust learning parameters
```

### Logging and Monitoring

```python
import logging
from federated_learning import FLLogger

# Setup logging
logger = FLLogger(
    level=logging.INFO,
    log_file="fl_training.log",
    include_metrics=True
)

# Monitor training progress
fl_system = FederatedLearning(config, logger=logger)

# Training metrics will be automatically logged
results = fl_system.start_training()
```

## Performance Optimization

### Parallel Client Training

```python
from federated_learning import ParallelTrainingManager

# Enable parallel client training
parallel_manager = ParallelTrainingManager(max_workers=4)
fl_system.set_training_manager(parallel_manager)
```

### Model Compression

```python
from federated_learning.compression import ModelCompressor

# Compress model updates to reduce communication overhead
compressor = ModelCompressor(compression_ratio=0.1)
fl_system.set_compressor(compressor)
```

### Asynchronous Updates

```python
from federated_learning import AsyncFederatedLearning

# Use asynchronous federated learning
async_fl = AsyncFederatedLearning(
    config=config,
    staleness_threshold=5  # Maximum allowed staleness
)
```

## Testing and Validation

### Unit Testing

```python
import unittest
from federated_learning.testing import FLTestCase

class TestFederatedLearning(FLTestCase):
    def test_client_training(self):
        client = self.create_test_client()
        result = client.train_local_model(self.global_weights)
        self.assertIsInstance(result, LocalTrainingResult)
        
    def test_server_aggregation(self):
        server = self.create_test_server()
        updates = self.create_test_updates()
        aggregated = server.aggregate_updates(updates)
        self.assertEqual(aggregated.shape, self.global_weights.shape)
```

### Integration Testing

```python
from federated_learning.testing import FLIntegrationTest

class TestFLIntegration(FLIntegrationTest):
    def test_end_to_end_training(self):
        fl_system = self.create_test_fl_system()
        results = fl_system.start_training()
        self.assertGreater(results.final_accuracy, 0.8)
```

---

**Note**: This documentation covers the complete API surface for a comprehensive federated learning framework. For implementation details and advanced usage patterns, refer to the individual module documentation and examples in the `/examples` directory.