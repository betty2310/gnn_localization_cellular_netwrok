# GNN-Based Cellular Network Localization
A implementation of Graph Neural Network approach for cellular network localization that leverages the inherent graph structure of cellular networks to improve localization accuracy.

### Dataset Context
The cellular network localization dataset was collected in Hanoi, Vietnam using Viettel network infrastructure. Each record contains:
- GPS reference coordinates (lat_ref, lon_ref)
- Timestamp and measurement index
- Number of detected cells
- For each cell: LAC, CID, cell coordinates, and RSSI values

**Dataset Format:** `stt,lat_ref,lon_ref,time,cells,(lac,cid,cell_lat,cell_lon,rssi)...`

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

## Architecture

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

## Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended)

### Install Dependencies

TODO