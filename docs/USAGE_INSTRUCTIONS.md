# Usage Instructions and Examples

## Table of Contents

1. [Quick Start Guide](#quick-start-guide)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Advanced Configurations](#advanced-configurations)
5. [Real-World Examples](#real-world-examples)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)
8. [Performance Optimization](#performance-optimization)

## Quick Start Guide

This guide will help you get started with the Federated Learning framework in under 10 minutes.

### Prerequisites

- Python 3.8 or higher
- NumPy, TensorFlow/PyTorch (optional)
- Network connectivity between clients and server

### 1. Basic Setup

```bash
# Install the federated learning framework
pip install federated-learning-framework

# Or install from source
git clone https://github.com/your-org/federated-learning
cd federated-learning
pip install -e .
```

### 2. Simple Example

```python
from federated_learning import FederatedLearning, FLConfig

# Create a simple configuration
config = FLConfig(
    num_clients=5,
    num_rounds=10,
    local_epochs=3,
    learning_rate=0.01
)

# Initialize the federated learning system
fl_system = FederatedLearning(config)

# Add some mock clients (in practice, these would be real distributed clients)
for i in range(5):
    client = fl_system.create_mock_client(f"client_{i}")
    fl_system.add_client(client)

# Start training
results = fl_system.start_training()
print(f"Training completed! Final accuracy: {results.final_accuracy:.4f}")
```

### 3. Expected Output

```
Round 1/10: Global accuracy: 0.6234, Loss: 1.2456
Round 2/10: Global accuracy: 0.7123, Loss: 0.9876
...
Round 10/10: Global accuracy: 0.8945, Loss: 0.4321
Training completed! Final accuracy: 0.8945
```

## Installation

### Standard Installation

```bash
# Install from PyPI (recommended)
pip install federated-learning-framework

# Install with additional dependencies
pip install federated-learning-framework[full]
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/your-org/federated-learning
cd federated-learning

# Create virtual environment
python -m venv fl_env
source fl_env/bin/activate  # On Windows: fl_env\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Docker Installation

```bash
# Pull the official Docker image
docker pull federated-learning:latest

# Run a container
docker run -it --name fl-container federated-learning:latest
```

### Verification

```python
import federated_learning
print(f"Federated Learning Framework version: {federated_learning.__version__}")

# Run basic tests
federated_learning.run_self_test()
```

## Basic Usage

### Setting Up a Federated Learning Experiment

#### 1. Configuration

```python
from federated_learning import FLConfig

# Basic configuration
config = FLConfig(
    # Core parameters
    num_clients=20,                    # Total number of clients
    num_rounds=50,                     # Training rounds
    local_epochs=5,                    # Local training epochs per round
    client_selection_ratio=0.3,        # Fraction of clients per round
    
    # Model parameters
    model_type="neural_network",
    model_config={
        "layers": [784, 128, 64, 10],
        "activation": "relu",
        "optimizer": "adam"
    },
    
    # Training parameters
    learning_rate=0.001,
    batch_size=32,
    
    # Data parameters
    data_distribution="non_iid",
    alpha=0.5  # For Dirichlet distribution
)
```

#### 2. Data Preparation

```python
from federated_learning.data import load_federated_data, DataPartitioner

# Load and partition data
partitioner = DataPartitioner(strategy="dirichlet", alpha=0.5)
client_data = partitioner.partition_data(
    dataset_path="./data/mnist.npz",
    num_clients=config.num_clients
)

# Verify data distribution
for client_id, (X, y) in client_data.items():
    print(f"Client {client_id}: {len(X)} samples, {len(np.unique(y))} classes")
```

#### 3. Client Setup

```python
from federated_learning import FLClient

clients = []
for client_id, (X_train, y_train) in client_data.items():
    client = FLClient(
        client_id=client_id,
        data=(X_train, y_train),
        model_config=config.model_config
    )
    clients.append(client)
```

#### 4. Server Setup and Training

```python
from federated_learning import FLServer, FederatedLearning

# Create server
server = FLServer(
    model_config=config.model_config,
    aggregation_strategy="fedavg"
)

# Create federated learning system
fl_system = FederatedLearning(
    config=config,
    server=server,
    clients=clients
)

# Start training
results = fl_system.start_training()
```

### Working with Real Datasets

#### MNIST Example

```python
from federated_learning.datasets import load_mnist_federated

# Load MNIST with federated partitioning
train_data, test_data = load_mnist_federated(
    num_clients=10,
    distribution="non_iid",
    alpha=0.1
)

# Create clients
clients = []
for client_id, (X, y) in train_data.items():
    client = FLClient(
        client_id=client_id,
        data=(X, y),
        model_config={
            "input_shape": (28, 28, 1),
            "num_classes": 10,
            "architecture": "cnn"
        }
    )
    clients.append(client)
```

#### CIFAR-10 Example

```python
from federated_learning.datasets import load_cifar10_federated

# Load CIFAR-10
train_data, test_data = load_cifar10_federated(
    num_clients=20,
    distribution="pathological",
    classes_per_client=2
)

# Create clients with CNN model
for client_id, (X, y) in train_data.items():
    client = FLClient(
        client_id=client_id,
        data=(X, y),
        model_config={
            "input_shape": (32, 32, 3),
            "num_classes": 10,
            "architecture": "resnet18"
        }
    )
    clients.append(client)
```

### Custom Models

#### Using PyTorch Models

```python
import torch
import torch.nn as nn
from federated_learning.models import PyTorchFLModel

class CustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Wrap in FL model
fl_model = PyTorchFLModel(
    model_class=CustomCNN,
    model_args={"num_classes": 10},
    optimizer="adam",
    loss_function="cross_entropy"
)
```

#### Using TensorFlow/Keras Models

```python
import tensorflow as tf
from federated_learning.models import TensorFlowFLModel

def create_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Wrap in FL model
fl_model = TensorFlowFLModel(
    model_fn=create_cnn_model,
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

## Advanced Configurations

### Communication Protocols

#### gRPC Communication

```python
from federated_learning.communication import gRPCCommunicationManager

# Configure gRPC communication
comm_config = {
    "protocol": "grpc",
    "server_address": "localhost:50051",
    "max_message_size": "100MB",
    "compression": "gzip",
    "tls_enabled": True,
    "certificate_path": "/path/to/cert.pem"
}

comm_manager = gRPCCommunicationManager(comm_config)
fl_system.set_communication_manager(comm_manager)
```

#### HTTP/REST Communication

```python
from federated_learning.communication import HTTPCommunicationManager

# Configure HTTP communication
comm_config = {
    "protocol": "http",
    "server_url": "https://fl-server.example.com",
    "api_version": "v1",
    "timeout": 30,
    "retry_attempts": 3
}

comm_manager = HTTPCommunicationManager(comm_config)
fl_system.set_communication_manager(comm_manager)
```

### Security and Privacy

#### Differential Privacy

```python
from federated_learning.privacy import DifferentialPrivacyManager

# Configure differential privacy
privacy_manager = DifferentialPrivacyManager(
    epsilon=1.0,           # Privacy budget
    delta=1e-5,           # Failure probability
    noise_multiplier=1.1,  # Noise scaling
    max_grad_norm=1.0     # Gradient clipping
)

# Apply to clients
for client in clients:
    client.set_privacy_manager(privacy_manager)
```

#### Secure Aggregation

```python
from federated_learning.security import SecureAggregationProtocol

# Configure secure aggregation
secure_agg = SecureAggregationProtocol(
    threshold=5,           # Minimum clients for aggregation
    dropout_resilience=2   # Maximum client dropouts tolerated
)

server.set_aggregation_protocol(secure_agg)
```

#### Encryption

```python
from federated_learning.security import EncryptionManager

# Configure end-to-end encryption
encryption_manager = EncryptionManager(
    algorithm="AES-256",
    key_rotation_interval=100  # Rotate keys every 100 rounds
)

fl_system.set_encryption_manager(encryption_manager)
```

### Custom Aggregation Strategies

#### Weighted Aggregation

```python
from federated_learning.aggregation import WeightedAggregator

class DataSizeWeightedAggregator(WeightedAggregator):
    def compute_weights(self, client_updates):
        total_samples = sum(update.num_samples for update in client_updates)
        weights = [update.num_samples / total_samples for update in client_updates]
        return weights

# Use custom aggregator
server.set_aggregation_strategy(DataSizeWeightedAggregator())
```

#### Robust Aggregation

```python
from federated_learning.aggregation import RobustAggregator

# Configure robust aggregation against Byzantine clients
robust_agg = RobustAggregator(
    strategy="krum",
    byzantine_tolerance=2,  # Tolerate up to 2 Byzantine clients
    selection_parameter=5   # Select top 5 clients for Krum
)

server.set_aggregation_strategy(robust_agg)
```

### Asynchronous Federated Learning

```python
from federated_learning import AsyncFederatedLearning

# Configure asynchronous FL
async_config = FLConfig(
    num_clients=50,
    staleness_threshold=5,  # Maximum rounds a client can be behind
    async_buffer_size=10,   # Buffer for storing async updates
    mixing_parameter=0.9    # Weight for mixing async updates
)

async_fl = AsyncFederatedLearning(async_config)
```

## Real-World Examples

### Healthcare: Federated Medical Image Classification

```python
from federated_learning import FederatedLearning, FLConfig
from federated_learning.datasets import MedicalImageDataset
from federated_learning.models import MedicalCNN

# Configuration for medical image classification
config = FLConfig(
    num_clients=5,  # 5 hospitals
    num_rounds=100,
    local_epochs=10,
    client_selection_ratio=1.0,  # All hospitals participate
    
    # Privacy settings for healthcare
    differential_privacy=True,
    epsilon=1.0,
    delta=1e-5,
    
    # Security settings
    encryption_enabled=True,
    secure_aggregation=True,
    
    # Model configuration
    model_config={
        "architecture": "medical_cnn",
        "input_shape": (224, 224, 3),
        "num_classes": 5,  # 5 disease categories
        "pretrained": True
    }
)

# Load medical datasets from different hospitals
hospital_data = {}
for hospital_id in ["hospital_A", "hospital_B", "hospital_C", "hospital_D", "hospital_E"]:
    dataset = MedicalImageDataset(
        data_path=f"/data/{hospital_id}/images",
        labels_path=f"/data/{hospital_id}/labels.csv",
        anonymization=True  # Remove patient identifiers
    )
    hospital_data[hospital_id] = dataset

# Create federated learning system
fl_system = FederatedLearning(config)

# Add hospital clients
for hospital_id, dataset in hospital_data.items():
    client = fl_system.create_client(
        client_id=hospital_id,
        dataset=dataset,
        compliance_checks=True  # Enable HIPAA compliance checks
    )
    fl_system.add_client(client)

# Start training
results = fl_system.start_training()

# Evaluate on global test set
global_test_data = MedicalImageDataset("/data/global_test")
evaluation = fl_system.evaluate(global_test_data)
print(f"Global model accuracy: {evaluation.accuracy:.4f}")
print(f"Per-class F1 scores: {evaluation.f1_scores}")
```

### Finance: Fraud Detection

```python
from federated_learning import FederatedLearning, FLConfig
from federated_learning.models import TabularNN

# Configuration for fraud detection
config = FLConfig(
    num_clients=10,  # 10 banks
    num_rounds=50,
    local_epochs=5,
    
    # Imbalanced data handling
    class_weights={0: 1.0, 1: 10.0},  # Weight fraud cases higher
    sampling_strategy="smote",
    
    # Privacy for financial data
    differential_privacy=True,
    epsilon=0.5,  # Strict privacy budget
    
    model_config={
        "architecture": "tabular_nn",
        "input_features": 50,
        "hidden_layers": [128, 64, 32],
        "output_classes": 2,  # Fraud vs. Normal
        "dropout_rate": 0.3
    }
)

# Load transaction data from different banks
bank_data = {}
for bank_id in range(10):
    # Each bank has different transaction patterns
    data = load_bank_transactions(
        bank_id=f"bank_{bank_id}",
        features=["amount", "merchant_category", "time_of_day", "location", ...],
        anonymize_pii=True
    )
    bank_data[f"bank_{bank_id}"] = data

# Create federated system
fl_system = FederatedLearning(config)

# Add bank clients
for bank_id, data in bank_data.items():
    client = fl_system.create_client(
        client_id=bank_id,
        data=data,
        data_validation=True,  # Validate data quality
        anomaly_detection=True  # Detect data poisoning
    )
    fl_system.add_client(client)

# Train the global fraud detection model
results = fl_system.start_training()

# Evaluate performance
test_metrics = fl_system.evaluate_on_test_set()
print(f"Fraud detection accuracy: {test_metrics.accuracy:.4f}")
print(f"Precision: {test_metrics.precision:.4f}")
print(f"Recall: {test_metrics.recall:.4f}")
print(f"AUC-ROC: {test_metrics.auc_roc:.4f}")
```

### IoT: Predictive Maintenance

```python
from federated_learning import FederatedLearning, FLConfig
from federated_learning.models import TimeSeriesLSTM

# Configuration for IoT predictive maintenance
config = FLConfig(
    num_clients=100,  # 100 manufacturing plants
    num_rounds=200,
    local_epochs=3,
    client_selection_ratio=0.2,  # Select 20% of plants per round
    
    # Handle intermittent connectivity
    asynchronous_updates=True,
    staleness_threshold=10,
    
    # Resource constraints for edge devices
    model_compression=True,
    compression_ratio=0.1,
    
    model_config={
        "architecture": "lstm",
        "sequence_length": 100,  # 100 time steps
        "input_features": 20,    # 20 sensor readings
        "lstm_units": 64,
        "output_classes": 3      # Normal, Warning, Critical
    }
)

# Load sensor data from manufacturing plants
plant_data = {}
for plant_id in range(100):
    # Each plant has different equipment and operating conditions
    sensor_data = load_sensor_data(
        plant_id=f"plant_{plant_id}",
        sensors=["temperature", "vibration", "pressure", "current", ...],
        time_window="30_days",
        preprocessing=True
    )
    plant_data[f"plant_{plant_id}"] = sensor_data

# Create federated system
fl_system = FederatedLearning(config)

# Add plant clients
for plant_id, data in plant_data.items():
    client = fl_system.create_edge_client(
        client_id=plant_id,
        data=data,
        resource_constraints={
            "max_memory": "512MB",
            "max_compute_time": "5min"
        }
    )
    fl_system.add_client(client)

# Train predictive maintenance model
results = fl_system.start_training()

# Deploy model to edge devices
for client in fl_system.get_clients():
    client.deploy_model(results.final_model)
    client.start_real_time_inference()
```

### Mobile: Keyboard Next-Word Prediction

```python
from federated_learning import FederatedLearning, FLConfig
from federated_learning.models import LanguageModel

# Configuration for mobile keyboard prediction
config = FLConfig(
    num_clients=10000,  # 10,000 mobile devices
    num_rounds=1000,
    local_epochs=1,     # Single epoch due to battery constraints
    client_selection_ratio=0.01,  # 1% of devices per round
    
    # Mobile-specific constraints
    max_update_size="1MB",
    battery_aware_training=True,
    wifi_only_updates=True,
    
    # Privacy for personal text data
    differential_privacy=True,
    epsilon=8.0,  # Moderate privacy for utility
    local_dp=True,  # Apply DP locally on device
    
    model_config={
        "architecture": "transformer",
        "vocab_size": 50000,
        "embedding_dim": 256,
        "num_heads": 8,
        "num_layers": 6,
        "max_sequence_length": 128
    }
)

# Simulate mobile clients with different text patterns
mobile_clients = []
for user_id in range(10000):
    # Each user has personalized text patterns
    client = MobileClient(
        user_id=f"user_{user_id}",
        text_data_generator=PersonalizedTextGenerator(),
        privacy_level="high",
        device_constraints={
            "max_training_time": "30s",
            "max_memory_usage": "100MB"
        }
    )
    mobile_clients.append(client)

# Create federated system
fl_system = FederatedLearning(config)

# Add mobile clients
for client in mobile_clients:
    fl_system.add_client(client)

# Train keyboard prediction model
results = fl_system.start_training()

# Deploy to app store
fl_system.package_model_for_deployment(
    platform="mobile",
    optimization="quantization",
    target_size="5MB"
)
```

## Best Practices

### 1. Data Preparation

#### Ensure Data Quality
```python
from federated_learning.data import DataValidator

# Validate data before training
validator = DataValidator()
for client in clients:
    validation_result = validator.validate_client_data(client.get_data())
    if not validation_result.is_valid:
        print(f"Data issues in {client.client_id}: {validation_result.issues}")
        # Fix data issues or exclude client
```

#### Handle Data Heterogeneity
```python
# Monitor data distribution across clients
from federated_learning.analysis import DataDistributionAnalyzer

analyzer = DataDistributionAnalyzer()
distribution_report = analyzer.analyze_client_distributions(clients)

# Adjust training strategy based on heterogeneity
if distribution_report.heterogeneity_score > 0.8:
    config.local_epochs = 10  # More local training for non-IID data
    config.learning_rate *= 0.5  # Reduce learning rate
```

### 2. Model Configuration

#### Start Simple
```python
# Begin with a simple model and gradually increase complexity
simple_config = {
    "architecture": "linear",
    "input_features": X.shape[1],
    "output_classes": num_classes
}

# Gradually increase complexity if needed
complex_config = {
    "architecture": "deep_nn",
    "layers": [512, 256, 128, 64],
    "dropout_rate": 0.3,
    "batch_normalization": True
}
```

#### Monitor Overfitting
```python
# Use validation sets to monitor overfitting
for client in clients:
    train_data, val_data = client.split_data(validation_split=0.2)
    client.set_validation_data(val_data)

# Monitor validation metrics during training
results = fl_system.start_training(monitor_validation=True)
```

### 3. Communication Efficiency

#### Model Compression
```python
from federated_learning.compression import ModelCompressor

# Compress model updates to reduce communication overhead
compressor = ModelCompressor(
    method="quantization",
    bits=8,  # 8-bit quantization
    sparsity_threshold=0.01  # Remove small weights
)

fl_system.set_model_compressor(compressor)
```

#### Update Scheduling
```python
# Schedule updates to avoid network congestion
from federated_learning.scheduling import UpdateScheduler

scheduler = UpdateScheduler(
    strategy="round_robin",
    batch_size=5,  # Update 5 clients at a time
    interval=60    # 60 seconds between batches
)

fl_system.set_update_scheduler(scheduler)
```

### 4. Privacy and Security

#### Implement Defense Mechanisms
```python
from federated_learning.security import ByzantineDefense

# Protect against malicious clients
defense = ByzantineDefense(
    detection_method="anomaly_detection",
    threshold=2.0,  # Standard deviations from mean
    quarantine_duration=10  # Rounds to quarantine suspicious clients
)

server.set_byzantine_defense(defense)
```

#### Regular Security Audits
```python
from federated_learning.security import SecurityAuditor

# Perform regular security audits
auditor = SecurityAuditor()
audit_report = auditor.audit_system(fl_system)

if audit_report.has_vulnerabilities:
    print("Security vulnerabilities found:")
    for vulnerability in audit_report.vulnerabilities:
        print(f"- {vulnerability.description}")
        print(f"  Severity: {vulnerability.severity}")
        print(f"  Recommendation: {vulnerability.recommendation}")
```

### 5. Monitoring and Debugging

#### Comprehensive Logging
```python
import logging
from federated_learning.logging import FLLogger

# Configure detailed logging
logger = FLLogger(
    level=logging.INFO,
    log_file="fl_training.log",
    include_metrics=True,
    include_system_info=True
)

fl_system.set_logger(logger)
```

#### Performance Monitoring
```python
from federated_learning.monitoring import PerformanceMonitor

# Monitor system performance
monitor = PerformanceMonitor(
    metrics=["cpu_usage", "memory_usage", "network_bandwidth", "training_time"],
    sampling_interval=10  # seconds
)

monitor.start_monitoring()
results = fl_system.start_training()
performance_report = monitor.stop_monitoring()
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Client Connection Issues

**Problem**: Clients failing to connect to server

**Solutions**:
```python
# Check network connectivity
from federated_learning.diagnostics import NetworkDiagnostics

diagnostics = NetworkDiagnostics()
connectivity_report = diagnostics.check_client_connectivity(clients)

for client_id, status in connectivity_report.items():
    if not status.connected:
        print(f"Client {client_id} connection issue: {status.error}")
        
        # Retry with different configuration
        if status.error == "timeout":
            client.set_timeout(60)  # Increase timeout
        elif status.error == "firewall":
            client.use_proxy(proxy_config)
```

#### 2. Model Convergence Issues

**Problem**: Model not converging or converging slowly

**Solutions**:
```python
# Adjust learning parameters
if results.convergence_rate < 0.01:
    config.learning_rate *= 2  # Increase learning rate
    config.local_epochs += 2   # More local training
    
# Check data quality
data_quality_report = analyzer.analyze_data_quality(clients)
if data_quality_report.average_quality < 0.7:
    # Implement data cleaning
    for client in clients:
        client.apply_data_cleaning()
```

#### 3. Memory Issues

**Problem**: Out of memory errors during training

**Solutions**:
```python
# Reduce batch size
config.batch_size = min(config.batch_size, 16)

# Enable gradient accumulation
config.gradient_accumulation_steps = 4

# Use model parallelism for large models
if model_size > 100_000_000:  # 100M parameters
    config.model_parallelism = True
    config.num_gpus = 2
```

#### 4. Privacy Budget Exhaustion

**Problem**: Differential privacy budget exhausted too quickly

**Solutions**:
```python
# Implement adaptive privacy budget allocation
from federated_learning.privacy import AdaptivePrivacyManager

adaptive_privacy = AdaptivePrivacyManager(
    total_budget=1.0,
    allocation_strategy="decreasing",  # Allocate more budget to early rounds
    min_budget_per_round=0.001
)

fl_system.set_privacy_manager(adaptive_privacy)
```

### Debugging Tools

#### 1. Training Visualization

```python
from federated_learning.visualization import TrainingVisualizer

# Visualize training progress
visualizer = TrainingVisualizer()
visualizer.plot_training_metrics(results)
visualizer.plot_client_participation(results)
visualizer.plot_model_convergence(results)
```

#### 2. Data Distribution Analysis

```python
from federated_learning.analysis import DataAnalyzer

# Analyze data distribution across clients
analyzer = DataAnalyzer()
distribution_plot = analyzer.plot_data_distribution(clients)
heterogeneity_metrics = analyzer.compute_heterogeneity_metrics(clients)
```

#### 3. Communication Analysis

```python
from federated_learning.analysis import CommunicationAnalyzer

# Analyze communication patterns
comm_analyzer = CommunicationAnalyzer()
bandwidth_usage = comm_analyzer.analyze_bandwidth_usage(results)
latency_analysis = comm_analyzer.analyze_communication_latency(results)
```

## Performance Optimization

### 1. Computational Optimization

#### GPU Acceleration
```python
# Enable GPU acceleration
config.device = "cuda"
config.mixed_precision = True  # Use FP16 for faster training

# Multi-GPU training
config.data_parallel = True
config.num_gpus = 4
```

#### Model Optimization
```python
# Optimize model architecture
from federated_learning.optimization import ModelOptimizer

optimizer = ModelOptimizer()
optimized_model = optimizer.optimize_for_federated_learning(
    model=base_model,
    target_latency=100,  # milliseconds
    target_memory=512,   # MB
    optimization_techniques=["pruning", "quantization", "distillation"]
)
```

### 2. Communication Optimization

#### Compression Techniques
```python
# Advanced compression
from federated_learning.compression import AdvancedCompressor

compressor = AdvancedCompressor(
    techniques=["gradient_compression", "error_feedback", "local_updates"],
    compression_ratio=0.01,  # 1% of original size
    error_compensation=True
)
```

#### Hierarchical Aggregation
```python
# Use hierarchical aggregation for large-scale FL
from federated_learning.aggregation import HierarchicalAggregator

hierarchical_agg = HierarchicalAggregator(
    hierarchy_levels=2,
    regional_servers=["us-east", "us-west", "europe", "asia"],
    aggregation_strategy="fedavg"
)

fl_system.set_aggregation_strategy(hierarchical_agg)
```

### 3. System-Level Optimization

#### Resource Management
```python
# Optimize resource usage
from federated_learning.resources import ResourceManager

resource_manager = ResourceManager(
    cpu_limit=80,     # 80% CPU usage limit
    memory_limit=8,   # 8GB memory limit
    priority_clients=["high_data_quality_clients"],
    load_balancing=True
)

fl_system.set_resource_manager(resource_manager)
```

#### Caching and Persistence
```python
# Enable intelligent caching
from federated_learning.caching import SmartCache

cache = SmartCache(
    cache_size="1GB",
    cache_strategy="lru",
    persistent_cache=True,
    cache_location="/tmp/fl_cache"
)

fl_system.set_cache_manager(cache)
```

---

This comprehensive usage guide provides practical instructions for implementing federated learning in various scenarios, from basic setups to complex real-world applications. Remember to always consider privacy, security, and performance requirements specific to your use case.