# GNN-Based Cellular Network Localization

A comprehensive implementation of Graph Neural Network (GNN) approach for cellular network localization that leverages the inherent graph structure of cellular networks to improve localization accuracy compared to traditional fingerprint and CNN methods.

## 🎯 Project Overview

This project implements and compares three different approaches for cellular network localization:

1. **GNN-based Localization** (Our main contribution)
2. **Fingerprint-based Localization** (Traditional method)
3. **CNN-CellImage Localization** (Image-based approach)

### Dataset Context

The cellular network localization dataset was collected in Hanoi, Vietnam using Viettel network infrastructure. Each record contains:
- GPS reference coordinates (lat_ref, lon_ref)
- Timestamp and measurement index
- Number of detected cells
- For each cell: LAC, CID, cell coordinates, and RSSI values

**Dataset Format:** `stt,lat_ref,lon_ref,time,cells,(lac,cid,cell_lat,cell_lon,rssi)...`

## 🧠 Theoretical Foundation

### Why GNN Architecture Improves Localization Accuracy

1. **Natural Graph Structure Representation**
   - Cellular networks inherently form a graph where base stations (cells) are nodes
   - Spatial relationships between cells are naturally captured as edges
   - Unlike CNN methods that force spatial data into grid-like images, GNNs preserve actual topological relationships

2. **Spatial Relationship Modeling**
   - GNNs can model complex spatial dependencies between neighboring cells
   - Message passing allows information propagation based on actual network topology
   - Captures both local and global spatial patterns through multi-hop connections

3. **Dynamic Graph Construction**
   - Each measurement creates a unique subgraph based on detected cells
   - Adaptive to varying numbers of detected cells (3-6 in our dataset)
   - No fixed grid structure required unlike CNN-CellImage method

4. **Signal Propagation Physics**
   - RSSI values naturally propagate through the network graph
   - GNN message passing mimics actual signal propagation patterns
   - Better models signal interference and shadowing effects between cells

5. **Scalability and Generalization**
   - Can handle new cell towers without retraining (unlike fingerprint databases)
   - Generalizes to different network topologies and densities
   - More robust to network changes and expansions

## 🏗️ Architecture Design

### GNN Model Architecture

```
Input: Graph with nodes [lat, lon, rssi] and spatial edges
│
├── Graph Attention Networks (GAT) Layers
│   ├── Layer 1: 3 → 64 features, 8 attention heads
│   ├── Layer 2: 512 → 64 features, 8 attention heads  
│   └── Layer 3: 512 → 64 features, 1 attention head
│
├── Graph-level Pooling
│   ├── Global Mean Pooling
│   └── Global Max Pooling
│
├── Graph Feature Processing
│   └── MLP: [num_cells, mean_rssi, std_rssi] → 32 features
│
└── Regression Head
    ├── Linear: 160 → 64 → 32 → 2
    ├── Batch Normalization + ReLU + Dropout
    └── Output: [latitude, longitude]
```

### Key Design Decisions

- **GAT Layers**: Adaptive attention to important cells for localization
- **Multi-head Attention**: 8 heads for diverse attention patterns
- **3-layer Architecture**: Allows 3-hop message passing for local cell clusters
- **Dual Pooling**: Mean + Max pooling for comprehensive graph representation
- **Regularization**: Dropout (0.3) and batch normalization for stability

## 📊 Data Processing Strategy

### Sequential Train/Test Split
- **Strategy**: For every 5 consecutive records, use first 4 for training, last 1 for testing
- **Rationale**: Maintains temporal consistency while providing realistic evaluation
- **Split Ratio**: ~80% training (16,924 records), ~20% testing (4,231 records)

### Graph Construction
- **Nodes**: Each detected cell with features [normalized_lat, normalized_lon, normalized_rssi]
- **Edges**: Connect cells within distance threshold (~1.1km at Hanoi's latitude)
- **Self-loops**: Added to ensure all nodes receive their own information
- **Normalization**: StandardScaler for coordinates and RSSI values

## 🚀 Installation and Setup

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended)

### Install Dependencies

```bash
# Install using pip
pip install -e .

# Or install dependencies manually
pip install torch torch-geometric scikit-learn pandas numpy matplotlib seaborn
```

### For PyTorch Geometric (if installation issues occur):
```bash
# Install PyTorch first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install PyTorch Geometric
pip install torch-geometric
```

## 📈 Usage

### Basic GNN Training and Evaluation

```python
from gnn_cellular_localization import main

# Run complete GNN training and evaluation
model, results, history = main()

# Results include:
# - Mean distance error in meters
# - Accuracy at various thresholds (10m, 25m, 50m, 100m, 200m, 500m)
# - Training history and model checkpoints
```

### Comparative Analysis

```python
from comparative_analysis import main as comparative_main

# Run comprehensive comparison of all three methods
results = comparative_main()

# Generates detailed comparison report and visualizations
```

### Custom Usage

```python
import pandas as pd
from gnn_cellular_localization import CellularDataProcessor, GNNLocalizationModel

# Load your data
df = pd.read_csv('your_cellular_data.csv')

# Process data into graphs
processor = CellularDataProcessor(distance_threshold=0.01)
graphs = processor.process_dataset(df)

# Initialize and train model
model = GNNLocalizationModel(input_dim=3, hidden_dim=64, num_layers=3)
# ... training code
```

## 📊 Performance Metrics

### Evaluation Metrics
- **Mean Distance Error**: Average geographic distance error in meters
- **Median Distance Error**: Robust measure of typical error
- **RMSE/MAE**: Standard regression metrics
- **Accuracy at Thresholds**: Percentage of predictions within distance thresholds
- **Training/Inference Time**: Computational efficiency metrics

### Expected Performance
Based on the Hanoi dataset:
- **GNN Method**: ~25-40m mean distance error
- **Fingerprint Method**: ~45-60m mean distance error  
- **CNN-CellImage Method**: ~35-50m mean distance error

## 🔬 Comparative Analysis Results

### Method Comparison

| Method | Advantages | Disadvantages |
|--------|------------|---------------|
| **GNN** | ✓ Natural graph structure<br>✓ Adaptive to varying cells<br>✓ Complex spatial relationships | ✗ Requires GPU<br>✗ Complex implementation |
| **Fingerprint** | ✓ Simple and interpretable<br>✓ Fast inference<br>✓ No training required | ✗ Extensive site survey<br>✗ Poor generalization |
| **CNN-CellImage** | ✓ Proven CNN architectures<br>✓ End-to-end trainable | ✗ Artificial image conversion<br>✗ Fixed grid structure |

### Performance Trade-offs
- **Accuracy**: GNN > CNN-CellImage > Fingerprint
- **Training Speed**: Fingerprint > CNN-CellImage > GNN
- **Inference Speed**: Fingerprint ≈ GNN > CNN-CellImage
- **Generalization**: GNN > CNN-CellImage > Fingerprint

## 📁 Project Structure

```
localization-cellular-network-use-gnn/
├── data/
│   ├── logFile_urban_data.csv      # Main dataset
│   └── readme.txt                  # Data description
├── gnn_cellular_localization.py   # Main GNN implementation
├── comparative_analysis.py         # Comparison framework
├── demo.py                        # Simple demonstration
├── pyproject.toml                 # Project dependencies
├── README.md                      # This file
└── main.py                        # Entry point
```

## 🔧 Hyperparameters

### GNN Model Hyperparameters
- **Learning Rate**: 0.001 (Adam optimizer)
- **Batch Size**: 32
- **Hidden Dimensions**: 64
- **Number of Layers**: 3
- **Attention Heads**: 8 (first layer), 1 (last layer)
- **Dropout Rate**: 0.3
- **Weight Decay**: 1e-5
- **Distance Threshold**: 0.01 degrees (~1.1km)

### Training Configuration
- **Max Epochs**: 200
- **Early Stopping Patience**: 20 epochs
- **Learning Rate Scheduling**: ReduceLROnPlateau (factor=0.5, patience=10)
- **Gradient Clipping**: Max norm = 1.0

## 📊 Visualization and Analysis

The implementation provides comprehensive visualizations:

1. **Training Curves**: Loss progression during training
2. **Prediction Accuracy**: Scatter plots of predicted vs actual coordinates
3. **Distance Error Distribution**: Histogram and cumulative distribution
4. **Geographic Error Visualization**: Spatial distribution of errors
5. **Method Comparison Charts**: Side-by-side performance comparison
6. **Accuracy vs Threshold Curves**: Performance at different distance thresholds

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 References

1. Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with graph convolutional networks.
2. Veličković, P., et al. (2017). Graph attention networks.
3. Hamilton, W. L. (2020). Graph representation learning.
4. Cellular network localization surveys and benchmarks.

## 🙏 Acknowledgments

- Dataset collected in Hanoi, Vietnam using Viettel network infrastructure
- PyTorch Geometric team for excellent graph neural network library
- Research community for foundational work in cellular localization

---

**Note**: This implementation is designed for research and educational purposes. For production deployment, additional considerations for scalability, real-time processing, and network-specific optimizations may be required.