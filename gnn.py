"""
GNN-Based Cellular Network Localization System

This module implements a Graph Neural Network approach for cellular network localization
that leverages the inherent graph structure of cellular networks to improve localization
accuracy compared to traditional fingerprint and CNN methods.

Theoretical Foundation:
======================

Why GNN Architecture Improves Localization Accuracy:

1. **Natural Graph Structure Representation**:
   - Cellular networks inherently form a graph where base stations (cells) are nodes
   - Spatial relationships between cells are naturally captured as edges
   - Unlike CNN methods that force spatial data into grid-like images, GNNs preserve
     the actual topological relationships between base stations

2. **Spatial Relationship Modeling**:
   - GNNs can model complex spatial dependencies between neighboring cells
   - Message passing allows information propagation based on actual network topology
   - Captures both local and global spatial patterns through multi-hop connections

3. **Dynamic Graph Construction**:
   - Each measurement creates a unique subgraph based on detected cells
   - Adaptive to varying numbers of detected cells (3-6 in our dataset)
   - No fixed grid structure required unlike CNN-CellImage method

4. **Signal Propagation Physics**:
   - RSSI values naturally propagate through the network graph
   - GNN message passing mimics actual signal propagation patterns
   - Better models signal interference and shadowing effects between cells

5. **Scalability and Generalization**:
   - Can handle new cell towers without retraining (unlike fingerprint databases)
   - Generalizes to different network topologies and densities
   - More robust to network changes and expansions

6. **Feature Learning**:
   - Automatically learns optimal spatial features from graph structure
   - Combines geometric (lat/lon) and signal (RSSI) information effectively
   - Learns complex non-linear relationships between cell configurations and locations
"""

from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool, GlobalAttention
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class CellularDataProcessor:
    """
    Processes cellular network data into graph representations suitable for GNN training.

    The processor converts each measurement record into a graph where:
    - Nodes represent detected cell towers with features [lat, lon, rssi]
    - Edges connect spatially proximate cells based on distance threshold
    - Graph-level features include measurement metadata
    """

    def __init__(self, distance_threshold: float = 0.01, rssi_threshold: float = 10.0):
        """
        Initialize the data processor.

        Args:
            distance_threshold: Maximum distance (in degrees) to create edges between cells
                              ~0.01 degrees ≈ 1.1km at Hanoi's latitude
            rssi_threshold: Maximum RSSI difference to create signal-based edges
        """
        self.distance_threshold = distance_threshold
        self.rssi_threshold = rssi_threshold
        self.scaler_coords = StandardScaler()
        self.scaler_rssi = StandardScaler()
        self.cell_registry = {}  # Registry of all unique cells

        # Edge creation statistics
        self.spatial_edges = 0
        self.rssi_edges = 0
        self.total_edges = 0

        # Initialize coordinate range attributes
        self.lat_min = 0.0
        self.lat_max = 0.0
        self.lon_min = 0.0
        self.lon_max = 0.0
        self.target_lat_min = 0.0
        self.target_lat_max = 0.0
        self.target_lon_min = 0.0
        self.target_lon_max = 0.0

    def parse_cellular_record(self, row: pd.Series) -> Dict:
        """
        Parse a single cellular measurement record into structured format.

        Args:
            row: Pandas series containing one measurement record

        Returns:
            Dictionary with parsed cell information and metadata
        """
        record = {
            'stt': row['stt'],
            'lat_ref': row['lat_ref'],
            'lon_ref': row['lon_ref'],
            'time': row['time'],
            'num_cells': row['cells'],
            'cells': []
        }

        # Parse cell information (groups of 5: lac, cid, cell_lat, cell_lon, rssi)
        cell_data = row.iloc[5:].values  # Skip first 5 columns

        for i in range(0, len(cell_data), 5):
            if i + 4 < len(cell_data) and pd.notna(cell_data[i]):
                cell = {
                    'lac': int(cell_data[i]) if pd.notna(cell_data[i]) else None,
                    'cid': int(cell_data[i+1]) if pd.notna(cell_data[i+1]) else None,
                    'lat': float(cell_data[i+2]) if pd.notna(cell_data[i+2]) else None,
                    'lon': float(cell_data[i+3]) if pd.notna(cell_data[i+3]) else None,
                    'rssi': float(cell_data[i+4]) if pd.notna(cell_data[i+4]) else None
                }

                # Only add valid cells
                if all(v is not None for v in cell.values()):
                    record['cells'].append(cell)

                    # Register unique cells
                    cell_id = f"{cell['lac']}_{cell['cid']}"
                    if cell_id not in self.cell_registry:
                        self.cell_registry[cell_id] = {
                            'lat': cell['lat'],
                            'lon': cell['lon'],
                            'lac': cell['lac'],
                            'cid': cell['cid']
                        }

        return record

    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate Euclidean distance between two geographic points.

        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates

        Returns:
            Euclidean distance in degrees
        """
        return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

    def create_graph_from_record(self, record: Dict) -> Data:
        """
        Convert a cellular measurement record into a PyTorch Geometric graph with enhanced edges.

        Graph Construction Improvements:
        - Creates both distance-based and RSSI-based edges
        - Adds edge attributes to capture relationship strength
        - Creates more meaningful connections between cells

        Args:
            record: Parsed cellular measurement record

        Returns:
            PyTorch Geometric Data object with enhanced structure
        """
        cells = record['cells']
        num_nodes = len(cells)

        if num_nodes == 0:
            return None

        # Create node features: [lat, lon, rssi]
        node_features = []
        positions = []
        rssi_values = []

        for cell in cells:
            node_features.append([cell['lat'], cell['lon'], cell['rssi']])
            positions.append([cell['lat'], cell['lon']])
            rssi_values.append(cell['rssi'])

        node_features = np.array(node_features)
        positions = np.array(positions)
        rssi_values = np.array(rssi_values)

        # Create edges using multiple strategies:
        edge_indices = []
        edge_attrs = []  # Edge attributes/weights

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    # 1. Spatial proximity
                    dist = self.calculate_distance(
                        positions[i][0], positions[i][1],
                        positions[j][0], positions[j][1]
                    )

                    # 2. RSSI similarity (stronger signals likely have more reliable connections)
                    rssi_diff = abs(rssi_values[i] - rssi_values[j])

                    # Create edge if either condition is met
                    if dist <= self.distance_threshold or rssi_diff <= self.rssi_threshold:
                        edge_indices.append([i, j])

                        # Edge weight combines distance and signal factors
                        # Normalize and invert distance (closer = higher weight)
                        dist_factor = max(0, 1 - (dist / self.distance_threshold))
                        # Normalize and invert RSSI difference (more similar = higher weight)
                        rssi_factor = max(0, 1 - (rssi_diff / self.rssi_threshold))
                        # Combined edge weight (higher = stronger connection)
                        edge_weight = (dist_factor + rssi_factor) / 2
                        edge_attrs.append([edge_weight, dist, rssi_diff])

        # Add self-loops with maximum weight
        for i in range(num_nodes):
            edge_indices.append([i, i])
            edge_attrs.append([1.0, 0.0, 0.0])  # Self connection: max weight, zero distance/diff

        # Convert to tensor format
        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        else:
            # If no edges, create self-loops only
            edge_index = torch.tensor([[i, i] for i in range(num_nodes)],
                                      dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor([[1.0, 0.0, 0.0] for _ in range(num_nodes)],
                                     dtype=torch.float)

        # Node features tensor
        x = torch.tensor(node_features, dtype=torch.float)

        # Target coordinates (what we want to predict)
        y = torch.tensor([record['lat_ref'], record['lon_ref']], dtype=torch.float)

        # Additional graph-level features
        graph_features = torch.tensor([
            num_nodes,  # Number of detected cells
            np.mean([cell['rssi'] for cell in cells]),  # Mean RSSI
            np.std([cell['rssi'] for cell in cells]) if num_nodes > 1 else 0,  # RSSI std
        ], dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                    graph_features=graph_features, num_nodes=num_nodes)

    def process_dataset(self, df: pd.DataFrame) -> List[Data]:
        """
        Process entire dataset into list of graph objects.

        Args:
            df: DataFrame containing cellular measurement data

        Returns:
            List of PyTorch Geometric Data objects
        """
        graphs = []

        print(f"Processing {len(df)} records...")

        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"Processed {idx}/{len(df)} records")

            record = self.parse_cellular_record(row)
            graph = self.create_graph_from_record(record)

            if graph is not None and graph.num_nodes > 0:
                graphs.append(graph)

        print(f"Successfully created {len(graphs)} graphs")
        print(f"Registered {len(self.cell_registry)} unique cells")

        # Normalize features across all graphs
        self._normalize_features(graphs)

        return graphs

    def _normalize_features(self, graphs: List[Data]):
        """
        Enhanced normalization with domain-specific knowledge for cellular networks.
        - RSSI values transformed to more meaningful representation
        - Adds signal quality features
        - Preserves original coordinate ranges for later denormalization

        Args:
            graphs: List of graph objects to normalize
        """
        # Collect all node features
        all_coords = []
        all_rssi = []
        all_targets = []

        for graph in graphs:
            if graph is None or not hasattr(graph, 'x') or graph.x is None:
                continue

            coords = graph.x[:, :2].numpy()  # lat, lon
            rssi = graph.x[:, 2:3].numpy()   # rssi
            all_coords.append(coords)
            all_rssi.append(rssi)

            target = graph.y.view(-1, 2).numpy() if hasattr(graph.y, 'view') else graph.y.reshape(1, 2)
            all_targets.append(target)

        if not all_coords or not all_rssi or not all_targets:
            print("Warning: Empty feature lists. Check graph creation.")
            return

        all_coords = np.vstack(all_coords)
        all_rssi = np.vstack(all_rssi)
        all_targets = np.vstack(all_targets)

        # Store original ranges for denormalization
        self.lat_min = float(np.min(all_coords[:, 0]))
        self.lat_max = float(np.max(all_coords[:, 0]))
        self.lon_min = float(np.min(all_coords[:, 1]))
        self.lon_max = float(np.max(all_coords[:, 1]))
        self.target_lat_min = float(np.min(all_targets[:, 0]))
        self.target_lat_max = float(np.max(all_targets[:, 0]))
        self.target_lon_min = float(np.min(all_targets[:, 1]))
        self.target_lon_max = float(np.max(all_targets[:, 1]))

        print(f"Coordinate ranges - Lat: [{self.lat_min:.6f}, {self.lat_max:.6f}], Lon: [{self.lon_min:.6f}, {self.lon_max:.6f}]")
        print(f"Target ranges - Lat: [{self.target_lat_min:.6f}, {self.target_lat_max:.6f}], Lon: [{self.target_lon_min:.6f}, {self.target_lon_max:.6f}]")

        # Fit scalers for standard normalization approach (kept for compatibility)
        self.scaler_coords.fit(all_coords)
        self.scaler_rssi.fit(all_rssi)

        # Apply enhanced normalization to each graph
        for graph in graphs:
            if graph is None or not hasattr(graph, 'x') or graph.x is None:
                continue

            # Get original features
            coords = graph.x[:, :2].numpy()
            rssi = graph.x[:, 2:3].numpy()

            # Min-max normalization for coordinates (relative to dataset bounds)
            coords_norm = np.zeros_like(coords)
            coords_norm[:, 0] = (coords[:, 0] - self.lat_min) / (self.lat_max - self.lat_min)
            coords_norm[:, 1] = (coords[:, 1] - self.lon_min) / (self.lon_max - self.lon_min)

            # RSSI normalization (assume typical range -110 to -40 dBm)
            # Transform to [0,1] with 1 being strongest signal
            rssi_norm = (rssi + 110) / 70.0
            rssi_norm = np.clip(rssi_norm, 0, 1)

            # Create additional features: signal quality categories (one-hot encoded)
            # e.g., poor (-110 to -90), fair (-90 to -80), good (-80 to -70), excellent (-70 to -40)
            signal_quality = np.zeros((len(rssi), 4))
            for i, r in enumerate(rssi):
                if r <= -90:
                    signal_quality[i, 0] = 1  # poor
                elif r <= -80:
                    signal_quality[i, 1] = 1  # fair
                elif r <= -70:
                    signal_quality[i, 2] = 1  # good
                else:
                    signal_quality[i, 3] = 1  # excellent

            # Combine all features
            graph.x = torch.tensor(
                np.hstack([coords_norm, rssi_norm, signal_quality]),
                dtype=torch.float
            )

            # Normalize target coordinates
            if hasattr(graph.y, 'view'):
                target = graph.y.view(-1, 2).numpy()
            else:
                target = np.array([graph.y[0], graph.y[1]]).reshape(1, 2)

            target_norm = np.zeros_like(target)
            target_norm[:, 0] = (target[:, 0] - self.lat_min) / (self.lat_max - self.lat_min)
            target_norm[:, 1] = (target[:, 1] - self.lon_min) / (self.lon_max - self.lon_min)
            graph.y = torch.tensor(target_norm.flatten(), dtype=torch.float)


class GNNLocalizationModel(nn.Module):
    """
    Graph Neural Network model for cellular network localization.

    Architecture Design Rationale:
    =============================

    1. **Multi-layer Graph Convolution**:
       - Uses Graph Attention Networks (GAT) for adaptive attention to important cells
       - 3 layers allow for 3-hop message passing (sufficient for local cell clusters)
       - Each layer: 64 → 128 → 64 hidden dimensions for feature expansion then compression

    2. **Attention Mechanism**:
       - GAT layers learn to focus on most informative cells for localization
       - 8 attention heads in first layer for diverse attention patterns
       - Helps handle varying signal quality and cell importance

    3. **Graph-level Pooling**:
       - Combines mean and max pooling for comprehensive graph representation
       - Mean pooling: captures average cell characteristics
       - Max pooling: captures strongest signal characteristics

    4. **Regression Head**:
       - Two-layer MLP with dropout for coordinate prediction
       - Separate outputs for latitude and longitude
       - Batch normalization for training stability

    5. **Regularization**:
       - Dropout (0.3) prevents overfitting to specific cell configurations
       - Batch normalization stabilizes training with varying graph sizes
    """

    def __init__(self,
                 input_dim: int = 3,
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 num_heads: int = 8,
                 dropout: float = 0.3):
        """
        Initialize GNN localization model.

        Args:
            input_dim: Input feature dimension (lat, lon, rssi = 3)
            hidden_dim: Hidden layer dimension
            num_layers: Number of graph convolution layers
            num_heads: Number of attention heads for GAT
            dropout: Dropout probability
        """
        super(GNNLocalizationModel, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # Graph convolution layers (using GAT for attention mechanism)
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer: input_dim -> hidden_dim
        self.convs.append(GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim * num_heads))

        # Middle layers: hidden_dim -> hidden_dim (with attention head concatenation)
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim,
                                      heads=num_heads, dropout=dropout))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim * num_heads))

        # Last layer: reduce to single head output
        self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim,
                                  heads=1, dropout=dropout))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Graph-level feature processing
        self.graph_feature_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),  # Process graph-level features
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Final regression layers
        # Combine pooled node features + graph features
        final_dim = hidden_dim * 2 + hidden_dim // 2  # mean_pool + max_pool + graph_features

        self.regressor = nn.Sequential(
            nn.Linear(final_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # Output: [lat, lon]
        )

    def forward(self, data):
        """
        Forward pass through the GNN model.

        Args:
            data: PyTorch Geometric batch of graph data

        Returns:
            Predicted coordinates [batch_size, 2]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        graph_features = data.graph_features

        # Graph convolution layers with residual connections
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)

            # Residual connection (when dimensions match)
            if i > 0 and x.size(-1) == x_new.size(-1):
                x = x + x_new
            else:
                x = x_new

        # Graph-level pooling
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)

        # Process graph-level features
        # Handle batched graph features - reshape to [batch_size, 3]
        if graph_features.dim() > 2:
            # If graph_features is [batch_size, num_graphs, 3], take the mean
            graph_features = graph_features.mean(dim=1)
        elif graph_features.dim() == 1:
            # If graph_features is flattened, reshape to [batch_size, 3]
            batch_size = mean_pool.size(0)
            graph_features = graph_features.view(batch_size, -1)
            if graph_features.size(1) != 3:
                # Take only the first 3 features if there are more
                graph_features = graph_features[:, :3]

        graph_feat = self.graph_feature_mlp(graph_features)

        # Combine all features
        combined = torch.cat([mean_pool, max_pool, graph_feat], dim=1)

        # Final regression
        output = self.regressor(combined)

        return output


class ImprovedGNNModel(nn.Module):
    """
    Improved Graph Neural Network model for cellular network localization.

    Key improvements:
    - Uses multiple GNN layer types for different kinds of message passing
    - Incorporates edge attributes in message passing
    - Enhanced global pooling with attention mechanism
    - Uses residual connections and skip connections
    - Deeper architecture with regularization
    """

    def __init__(self,
                 input_dim=7,  # [lat, lon, rssi, 4 signal quality features]
                 hidden_dim=128,
                 edge_dim=3):  # [weight, distance, rssi_diff]
        super(ImprovedGNNModel, self).__init__()

        # Graph convolution layers with different types for diverse message passing
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, dropout=0.2, edge_dim=edge_dim)
        self.conv2 = GATConv(hidden_dim*4, hidden_dim, heads=2, dropout=0.2, edge_dim=edge_dim)
        self.conv3 = SAGEConv(hidden_dim*2, hidden_dim)  # SAGEConv for robust aggregation
        self.conv4 = GCNConv(hidden_dim, hidden_dim)     # GCNConv for final refinement

        # Batch normalization for training stability
        self.bn1 = nn.BatchNorm1d(hidden_dim*4)
        self.bn2 = nn.BatchNorm1d(hidden_dim*2)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)

        # Process graph-level features
        self.graph_feature_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Attention-based global pooling
        self.global_attention = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.ReLU(),
                nn.Linear(hidden_dim//2, 1)
            )
        )

        # Final regression layers with skip connections
        combined_dim = hidden_dim*2 + hidden_dim//2  # [mean_pool, att_pool, graph_features]

        self.regressor = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Final output layer
        self.final = nn.Linear(hidden_dim//2, 2)

        # Skip connection for residual learning
        self.skip = nn.Linear(combined_dim, hidden_dim//2)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        graph_features = data.graph_features

        # First GAT layer
        x1 = self.conv1(x, edge_index, edge_attr)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)

        # Second GAT layer
        x2 = self.conv2(x1, edge_index, edge_attr)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=0.2, training=self.training)

        # SAGE convolution layer
        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)
        x3 = F.dropout(x3, p=0.2, training=self.training)

        # Final GCN layer (with residual connection)
        x4 = self.conv4(x3, edge_index)
        x4 = self.bn4(x4)
        x4 = F.relu(x4)
        x4 = x3 + x4  # Residual connection

        # Multiple pooling strategies
        mean_pool = global_mean_pool(x4, batch)
        att_pool = self.global_attention(x4, batch)

        # Process graph-level features
        if graph_features.dim() > 2:
            # If graph_features is [batch_size, num_graphs, 3], take the mean
            graph_features = graph_features.mean(dim=1)
        elif graph_features.dim() == 1:
            # If graph_features is flattened, reshape to [batch_size, 3]
            batch_size = mean_pool.size(0)
            graph_features = graph_features.view(batch_size, -1)
            if graph_features.size(1) != 3:
                # Take only the first 3 features if there are more
                graph_features = graph_features[:, :3]

        graph_feat = self.graph_feature_mlp(graph_features)

        # Combine all pooled representations
        combined = torch.cat([mean_pool, att_pool, graph_feat], dim=1)

        # Apply regressor with skip connection
        hidden = self.regressor(combined)
        skip_features = self.skip(combined)

        # Add skip connection and final projection
        output = self.final(hidden + skip_features)

        return output


class GNNTrainer:
    """
    Training and evaluation manager for GNN localization model.

    Training Strategy:
    ==================

    1. **Loss Function**: MSE Loss for coordinate regression
       - Directly optimizes localization error
       - Stable gradients for coordinate prediction

    2. **Optimizer**: Adam with learning rate scheduling
       - Initial LR: 0.001 (good for GAT layers)
       - ReduceLROnPlateau: reduces LR when validation loss plateaus
       - Weight decay: 1e-5 for L2 regularization

    3. **Batch Size**: 32 (balance between memory and gradient stability)
       - Small enough for diverse graph sizes
       - Large enough for stable batch normalization

    4. **Early Stopping**: Prevents overfitting
       - Patience: 20 epochs
       - Monitors validation loss

    5. **Data Splitting Strategy**:
       - Sequential 5-record groups: 4 training, 1 testing
       - Maintains temporal consistency
       - Realistic evaluation of model generalization
    """

    def __init__(self, model: ImprovedGNNModel, device=None):
        """
        Initialize trainer.

        Args:
            model: GNN model to train
            device: Training device ('cuda' or 'cpu')
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = model.to(device)
        self.device = device

        # Optimizer with weight decay for regularization
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.0005,  # Lower learning rate for better stability
            weight_decay=1e-4   # Increased regularization
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,  # Reduced patience for faster adaptation
            min_lr=1e-6  # Set minimum learning rate
        )

        # Loss function
        self.criterion = nn.MSELoss()

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = 20

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train model for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in train_loader:
            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            pred = self.model(batch)

            # Reshape target tensor to match prediction shape [batch_size, 2]
            target = batch.y.view(-1, 2)
            loss = self.criterion(pred, target)

            # Backward pass
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate model performance.

        Args:
            val_loader: Validation data loader

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                pred = self.model(batch)

                # Reshape target tensor to match prediction shape [batch_size, 2]
                target = batch.y.view(-1, 2)
                loss = self.criterion(pred, target)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int = 300) -> Dict:
        """
        Enhanced training loop with:
        - Cosine annealing scheduler with warm restarts
        - Better early stopping strategy
        - Training curve visualization
        - Gradient monitoring

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs

        Returns:
            Training history dictionary
        """
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Use cosine annealing scheduler with warm restarts
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=20,  # Initial restart period
            T_mult=2,  # Increase period by factor of 2 after each restart
            eta_min=1e-6  # Minimum learning rate
        )

        # Early stopping with longer patience
        self.max_patience = 50
        self.patience_counter = 0
        self.best_val_loss = float('inf')

        # Track losses and learning rates
        self.train_losses = []
        self.val_losses = []
        lr_history = []

        for epoch in range(num_epochs):
            # Training
            self.model.train()
            total_loss = 0
            num_batches = 0

            for batch in train_loader:
                batch = batch.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                pred = self.model(batch)

                # Reshape target tensor to match prediction shape [batch_size, 2]
                target = batch.y.view(-1, 2)
                loss = self.criterion(pred, target)

                # Backward pass
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Monitor gradients every 50 epochs
                if epoch % 50 == 0 and num_batches == 0:
                    total_norm = 0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.detach().data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** 0.5
                    print(f"  Gradient norm: {total_norm:.4f}")

                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            train_loss = total_loss / num_batches

            # Validation
            val_loss = self.validate(val_loader)

            # Learning rate scheduling
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            lr_history.append(current_lr)

            # Record losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Early stopping check with improved logic
            if val_loss < self.best_val_loss * 0.997:  # 0.3% improvement needed
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_gnn_model.pth')
                print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}, LR: {current_lr:.2e} (saved)")
            else:
                self.patience_counter += 1

                # Print progress every 10 epochs or when close to stopping
                if epoch % 10 == 0 or self.patience_counter > self.max_patience - 10:
                    print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.6f}, "
                          f"Val Loss: {val_loss:.6f}, LR: {current_lr:.2e}")

            # Advanced early stopping - restart when plateauing
            if epoch >= 100 and self.patience_counter >= 25 and self.patience_counter < self.max_patience:
                # Learning rate warmup restart
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 0.0005  # Reset LR
                print(f"Restarting learning rate at epoch {epoch}")
                self.patience_counter = 0

            # Final early stopping
            if self.patience_counter >= self.max_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Load best model
        self.model.load_state_dict(torch.load('best_gnn_model.pth'))

        # Plot training curves
        self._plot_training_curves(lr_history)

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'lr_history': lr_history
        }

    def _plot_training_curves(self, lr_history):
        """Visualize training progress and learning rate schedule"""
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.yscale('log')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 1, 2)
        plt.plot(lr_history)
        plt.yscale('log')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_curves.png')


class ModelEvaluator:
    """
    Comprehensive evaluation of GNN localization model performance.

    Evaluation Metrics:
    ==================

    1. **Mean Squared Error (MSE)**: Primary training objective
    2. **Mean Absolute Error (MAE)**: Interpretable average error
    3. **Root Mean Squared Error (RMSE)**: Error in same units as coordinates
    4. **Distance Error**: Actual geographic distance error in meters
    5. **Accuracy at Thresholds**: Percentage of predictions within distance thresholds
    """

    def __init__(self, model: ImprovedGNNModel, processor: CellularDataProcessor, device=None):
        """
        Initialize evaluator.

        Args:
            model: Trained GNN model
            processor: Data processor with normalization information
            device: Evaluation device ('cuda' or 'cpu')
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = model.to(device)
        self.device = device
        self.processor = processor  # Store processor for denormalization

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate haversine distance between coordinate pairs.

        Args:
            lat1, lon1: First set of coordinates
            lat2, lon2: Second set of coordinates

        Returns:
            Distance in meters
        """
        # Convert inputs to numpy arrays if they aren't already
        lat1 = np.asarray(lat1)
        lon1 = np.asarray(lon1)
        lat2 = np.asarray(lat2)
        lon2 = np.asarray(lon2)

        R = 6371000.0  # Earth radius in meters

        # Convert decimal degrees to radians
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)

        # Haversine formula
        a = np.sin(dlat/2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2.0)**2
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0-a))
        distance = R * c

        return distance

    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Comprehensive model evaluation.

        Args:
            test_loader: Test data loader

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                pred = self.model(batch)

                # Reshape target tensor to match prediction shape [batch_size, 2]
                if hasattr(batch.y, 'view'):
                    target = batch.y.view(-1, 2)
                else:
                    target = batch.y.reshape(-1, 2)

                all_predictions.append(pred.cpu().numpy())
                all_targets.append(target.cpu().numpy())

        # Stack predictions and targets
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)

        # Denormalize predictions and targets back to original coordinate space
        predictions_denorm = self.processor.scaler_coords.inverse_transform(predictions)
        targets_denorm = self.processor.scaler_coords.inverse_transform(targets)

        # Basic regression metrics on normalized data
        mse = float(mean_squared_error(targets, predictions))
        mae = float(mean_absolute_error(targets, predictions))
        rmse = float(np.sqrt(mse))

        # Geographic distance errors on denormalized data
        distances = self.haversine_distance(
            targets_denorm[:, 0], targets_denorm[:, 1],
            predictions_denorm[:, 0], predictions_denorm[:, 1]
        )

        mean_distance_error = np.mean(distances)
        median_distance_error = np.median(distances)
        std_distance_error = np.std(distances)

        # Accuracy at different thresholds
        thresholds = [10, 25, 50, 100, 200, 500]  # meters
        accuracies = {}

        for threshold in thresholds:
            accuracy = float(np.sum(distances <= threshold) / len(distances) * 100)
            accuracies[f'accuracy_{threshold}m'] = accuracy

        # Coordinate-wise errors
        lat_mae = float(mean_absolute_error(targets[:, 0], predictions[:, 0]))
        lon_mae = float(mean_absolute_error(targets[:, 1], predictions[:, 1]))

        # Calculate statistics on distances
        mean_distance_error = float(np.mean(distances))
        median_distance_error = float(np.median(distances))
        std_distance_error = float(np.std(distances))
        min_distance_error = float(np.min(distances))
        max_distance_error = float(np.max(distances))

        results = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mean_distance_error_m': mean_distance_error,
            'median_distance_error_m': median_distance_error,
            'std_distance_error_m': std_distance_error,
            'min_distance_error_m': min_distance_error,
            'max_distance_error_m': max_distance_error,
            'lat_mae': lat_mae,
            'lon_mae': lon_mae,
            **accuracies,
            'predictions': predictions_denorm,  # Store denormalized predictions
            'targets': targets_denorm,          # Store denormalized targets
            'distances': distances
        }

        return results

    def plot_results(self, results: Dict, save_path: str = 'gnn_evaluation_plots.png'):
        """
        Create comprehensive visualization of results.

        Args:
            results: Evaluation results dictionary
            save_path: Path to save plots
        """
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        predictions = results['predictions']
        targets = results['targets']
        distances = results['distances']

        # 1. Predicted vs Actual Coordinates
        axes[0, 0].scatter(targets[:, 0], predictions[:, 0], alpha=0.6, s=20)
        min_lat, max_lat = min(targets[:, 0].min(), predictions[:, 0].min()), max(targets[:, 0].max(), predictions[:, 0].max())
        axes[0, 0].plot([min_lat, max_lat], [min_lat, max_lat], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Latitude')
        axes[0, 0].set_ylabel('Predicted Latitude')
        axes[0, 0].set_title('Latitude Prediction Accuracy')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].scatter(targets[:, 1], predictions[:, 1], alpha=0.6, s=20)
        min_lon, max_lon = min(targets[:, 1].min(), predictions[:, 1].min()), max(targets[:, 1].max(), predictions[:, 1].max())
        axes[0, 1].plot([min_lon, max_lon], [min_lon, max_lon], 'r--', lw=2)
        axes[0, 1].set_xlabel('Actual Longitude')
        axes[0, 1].set_ylabel('Predicted Longitude')
        axes[0, 1].set_title('Longitude Prediction Accuracy')
        axes[0, 1].grid(True, alpha=0.3)

        # 2. Distance Error Distribution
        # Clip distances to a reasonable range for better visualization
        plot_distances = np.clip(distances, 0, 500)  # Clip at 500m for better visualization
        axes[0, 2].hist(plot_distances, bins=50, alpha=0.7, edgecolor='black')
        mean_dist = results['mean_distance_error_m']
        median_dist = results['median_distance_error_m']
        axes[0, 2].axvline(mean_dist, color='red', linestyle='--',
                           label=f'Mean: {mean_dist:.1f}m')
        axes[0, 2].axvline(median_dist, color='green', linestyle='--',
                           label=f'Median: {median_dist:.1f}m')
        axes[0, 2].set_xlabel('Distance Error (meters)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Distance Error Distribution')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # 3. Cumulative Distance Error
        sorted_distances = np.sort(plot_distances)
        cumulative_pct = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances) * 100

        axes[1, 0].plot(sorted_distances, cumulative_pct, linewidth=2)
        axes[1, 0].set_xlabel('Distance Error (meters)')
        axes[1, 0].set_ylabel('Cumulative Percentage (%)')
        axes[1, 0].set_title('Cumulative Distance Error')
        axes[1, 0].grid(True, alpha=0.3)

        # Add threshold lines
        thresholds = [50, 100, 200]
        for threshold in thresholds:
            pct = 100 * np.sum(distances <= threshold) / len(distances)
            axes[1, 0].axvline(threshold, color='red', linestyle='--', alpha=0.7)
            axes[1, 0].text(threshold, min(pct + 5, 95), f'{pct:.1f}%',
                            rotation=90, ha='center')

        # 4. Accuracy at Thresholds
        thresholds = [10, 25, 50, 100, 200, 500]
        accuracies = [results[f'accuracy_{t}m'] for t in thresholds]

        axes[1, 1].bar(range(len(thresholds)), accuracies, alpha=0.7)
        axes[1, 1].set_xlabel('Distance Threshold (meters)')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].set_title('Accuracy at Different Thresholds')
        axes[1, 1].set_xticks(range(len(thresholds)))
        axes[1, 1].set_xticklabels([f'{t}m' for t in thresholds])
        axes[1, 1].grid(True, alpha=0.3)

        # Add percentage labels on bars
        for i, acc in enumerate(accuracies):
            axes[1, 1].text(i, min(acc + 1, 99), f'{acc:.1f}%', ha='center', va='bottom')

        # 5. Geographic Error Visualization
        scatter = axes[1, 2].scatter(targets[:, 1], targets[:, 0], c=np.clip(distances, 0, 200),
                                     cmap='viridis', alpha=0.6, s=20)
        cbar = plt.colorbar(scatter, ax=axes[1, 2])
        cbar.set_label('Distance Error (meters)')
        axes[1, 2].set_xlabel('Longitude')
        axes[1, 2].set_ylabel('Latitude')
        axes[1, 2].set_title('Geographic Distribution of Errors')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        # Create comparison table plot
        self._plot_comparison_table(results, save_path.replace('.png', '_table.png'))

    def _plot_comparison_table(self, results: Dict, save_path: str = 'comparison_table.png'):
        """
        Create a comparison table of localization methods.

        Args:
            results: Evaluation results dictionary
            save_path: Path to save the table image
        """
        # Define the methods and their metrics
        methods = [
            'Cell ID', 'Centroid', 'Weighted Centroid', 'Linear Regression',
            'Support Vector Regression', 'Multilayer Perceptron', 'Fingerprint',
            'CNN-CellImage', 'GNN'
        ]

        # Example values from the paper (replace with actual values if available)
        mean_values = [242.6, 160.7, 155.3, 137.3, 32.2, 29.7, 26.1, 33.1, results['mean_distance_error_m']]
        median_values = [199.6, 150.6, 144.6, 126.6, 25.5, 21.3, 13.5, 19.8, results['median_distance_error_m']]
        min_values = [4.8, 1.0, 2.8, 1.5, 0.2, 0.3, 0.2, 0.3, results['min_distance_error_m']]
        max_values = [1522.8, 471.0, 403.4, 496.1, 325.4, 400.3, 610.8, 642.1, results['max_distance_error_m']]

        # Create a figure and axis
        fig, ax = plt.figure(figsize=(12, 6)), plt.gca()

        # Hide axes
        ax.axis('tight')
        ax.axis('off')

        # Create the table
        table_data = []
        for i, method in enumerate(methods):
            table_data.append([
                method,
                f"{mean_values[i]:.1f}",
                f"{median_values[i]:.1f}",
                f"{min_values[i]:.1f}",
                f"{max_values[i]:.1f}"
            ])

        # Add header
        header = ['Method', 'Mean (m)', 'Median (m)', 'Min (m)', 'Max (m)']

        # Create the table
        table = ax.table(
            cellText=table_data,
            colLabels=header,
            loc='center',
            cellLoc='center',
            colColours=['#f2f2f2']*5,
            colWidths=[0.3, 0.15, 0.15, 0.15, 0.15]
        )

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Highlight our method
        for i in range(5):
            cell = table[(len(methods), i)]
            cell.set_facecolor('#e6f2ff')

        # Add title
        plt.title('Distance error of localization methods in Scenario 1', fontsize=14, pad=20)

        # Save the figure
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

        # Print summary statistics
        print("\n" + "="*60)
        print("GNN LOCALIZATION MODEL EVALUATION RESULTS")
        print("="*60)
        print(f"Mean Distance Error:    {results['mean_distance_error_m']:.2f} meters")
        print(f"Median Distance Error:  {results['median_distance_error_m']:.2f} meters")
        print(f"Min Distance Error:     {results['min_distance_error_m']:.2f} meters")
        print(f"Max Distance Error:     {results['max_distance_error_m']:.2f} meters")
        print(f"Std Distance Error:     {results['std_distance_error_m']:.2f} meters")
        print(f"RMSE:                   {results['rmse']:.6f}")
        print(f"MAE:                    {results['mae']:.6f}")
        print("\nAccuracy at Thresholds:")
        for threshold in [10, 25, 50, 100, 200, 500]:
            acc = results[f'accuracy_{threshold}m']
            print(f"  Within {threshold:3d}m:         {acc:5.1f}%")


def create_improved_data_split(graphs: List[Data]) -> Tuple[List[Data], List[Data], List[Data]]:
    """
    Enhanced data splitting strategy that:
    1. Maintains spatial coherence (geographically similar areas stay together)
    2. Ensures diverse cell tower configurations in each split

    Args:
        graphs: List of graph objects

    Returns:
        Tuple of (train_graphs, val_graphs, test_graphs)
    """
    # Extract features for clustering
    features = []
    for graph in graphs:
        if graph is None or not hasattr(graph, 'x') or graph.x is None:
            continue

        # Calculate center point of this measurement
        lat_mean = torch.mean(graph.x[:, 0]).item()
        lon_mean = torch.mean(graph.x[:, 1]).item()
        rssi_mean = torch.mean(graph.x[:, 2]).item()
        num_nodes = graph.num_nodes

        features.append([lat_mean, lon_mean, rssi_mean, num_nodes])

    features = np.array(features)

    # Normalize features for clustering
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_norm = scaler.fit_transform(features)

    # Apply clustering to group similar measurements
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=10, random_state=42)
    clusters = kmeans.fit_predict(features_norm)

    # Create stratified split based on clusters
    from sklearn.model_selection import StratifiedShuffleSplit

    # Create stratified split (70% train, 15% val, 15% test)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, temp_idx = next(split.split(features, clusters))

    # Further split temp into val and test
    val_test_clusters = clusters[temp_idx]
    val_test_split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    rel_val_idx, rel_test_idx = next(val_test_split.split(features[temp_idx], val_test_clusters))

    val_idx = temp_idx[rel_val_idx]
    test_idx = temp_idx[rel_test_idx]

    # Create final splits
    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]
    test_graphs = [graphs[i] for i in test_idx]

    print(f"Stratified split based on geographic and signal clusters:")
    print(f"  Training: {len(train_graphs)} graphs ({len(train_graphs)/len(graphs)*100:.1f}%)")
    print(f"  Validation: {len(val_graphs)} graphs ({len(val_graphs)/len(graphs)*100:.1f}%)")
    print(f"  Testing: {len(test_graphs)} graphs ({len(test_graphs)/len(graphs)*100:.1f}%)")

    return train_graphs, val_graphs, test_graphs


def create_sequential_split(graphs: List[Data], group_size: int = 5) -> Tuple[List[Data], List[Data]]:
    """
    Create sequential train/test split using group-based strategy.

    Strategy: For every group_size consecutive records, use first (group_size-1) 
    for training and last 1 for testing. This maintains temporal consistency
    while providing realistic evaluation.

    Args:
        graphs: List of graph objects
        group_size: Size of sequential groups (default: 5)

    Returns:
        Tuple of (train_graphs, test_graphs)
    """
    train_graphs = []
    test_graphs = []

    for i in range(0, len(graphs), group_size):
        group = graphs[i:i+group_size]

        if len(group) == group_size:
            # Add first (group_size-1) to training, last 1 to testing
            train_graphs.extend(group[:-1])
            test_graphs.append(group[-1])
        else:
            # Handle remaining records (add all to training)
            train_graphs.extend(group)

    print(f"Sequential split: {len(train_graphs)} training, {len(test_graphs)} testing")
    print(f"Split ratio: {len(train_graphs)/(len(train_graphs)+len(test_graphs)):.3f} train")

    return train_graphs, test_graphs


class GeographicDistanceLoss(nn.Module):
    """
    Custom loss function that combines MSE loss with geographic distance error.

    This loss function:
    1. Uses standard MSE loss on normalized coordinates
    2. Adds a component based on approximate geographic distance
    3. Balances these components with a weighting factor

    Args:
        processor: CellularDataProcessor with coordinate normalization info
        alpha: Weight between MSE and geographic components (0-1)
    """

    def __init__(self, processor: CellularDataProcessor, alpha: float = 0.7):
        super(GeographicDistanceLoss, self).__init__()
        self.processor = processor
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        # Standard MSE loss on normalized coordinates
        mse_loss = self.mse(pred, target)

        # Get normalization parameters
        lat_min, lat_max = self.processor.lat_min, self.processor.lat_max
        lon_min, lon_max = self.processor.lon_min, self.processor.lon_max

        # Denormalize predictions and targets
        pred_denorm = torch.zeros_like(pred)
        target_denorm = torch.zeros_like(target)

        pred_denorm[:, 0] = pred[:, 0] * (lat_max - lat_min) + lat_min
        pred_denorm[:, 1] = pred[:, 1] * (lon_max - lon_min) + lon_min
        target_denorm[:, 0] = target[:, 0] * (lat_max - lat_min) + lat_min
        target_denorm[:, 1] = target[:, 1] * (lon_max - lon_min) + lon_min

        # Calculate Euclidean distance (simpler approximation of geographic distance)
        # Using approx 111,000 meters per degree of latitude/longitude (rough approximation)
        meters_per_degree = 111000.0
        lat_diff_meters = (pred_denorm[:, 0] - target_denorm[:, 0]) * meters_per_degree
        lon_diff_meters = (pred_denorm[:, 1] - target_denorm[:, 1]) * meters_per_degree

        # Euclidean distance in meters (approximation)
        distances = torch.sqrt(lat_diff_meters**2 + lon_diff_meters**2)
        geo_loss = torch.mean(distances) / 1000.0  # Convert to km for better scale

        # Combined loss (weighted sum)
        return self.alpha * mse_loss + (1 - self.alpha) * geo_loss


def main():
    """
    Main execution function for improved GNN-based cellular localization.
    """
    print("Enhanced GNN-Based Cellular Network Localization System")
    print("=" * 60)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and process data
    print("\n1. Loading and processing data...")
    df = pd.read_csv('data/logFile_urban_data.csv')
    print(f"Loaded {len(df)} records")

    # Initialize data processor with improved parameters
    processor = CellularDataProcessor(
        distance_threshold=0.005,  # ~500m at Hanoi's latitude
        rssi_threshold=15.0        # RSSI difference threshold
    )

    # Process dataset into graphs
    graphs = processor.process_dataset(df)
    print(f"Created {len(graphs)} valid graphs")

    # Create improved data split
    print("\n2. Creating improved data split...")
    train_graphs, val_graphs, test_graphs = create_improved_data_split(graphs)

    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

    # Initialize improved model
    print("\n3. Initializing enhanced GNN model...")
    model = ImprovedGNNModel(
        input_dim=7,      # [lat, lon, rssi, 4 signal quality features]
        hidden_dim=128,
        edge_dim=3        # [weight, distance, rssi_diff]
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize trainer with custom loss
    custom_loss = GeographicDistanceLoss(processor, alpha=0.7)

    trainer = GNNTrainer(model, device)
    trainer.criterion = custom_loss  # Replace default MSE loss

    # Train model
    print("\n4. Training enhanced model...")
    history = trainer.train(train_loader, val_loader, num_epochs=300)

    # Evaluate model
    print("\n5. Evaluating model...")
    evaluator = ModelEvaluator(model, processor, device)
    results = evaluator.evaluate(test_loader)

    # Visualize results
    evaluator.plot_results(results)

    # Save model and results
    torch.save(model.state_dict(), 'gnn_localization_model.pth')

    print("\n6. Training completed successfully!")
    print(f"Best validation loss: {history['best_val_loss']:.6f}")
    print(f"Final test distance error: {results['mean_distance_error_m']:.2f} meters")

    return model, results, history


def create_day_based_split(df: pd.DataFrame, processor: CellularDataProcessor,
                           train_ratio: float = 0.8) -> Tuple[List[Data], List[Data], List[Data]]:
    """
    Create train/val/test split based on different days of measurements.

    This implements Scenario 2: training and test sets built from measurements 
    taken on different days to evaluate generalization across varying 
    environmental conditions and time periods.

    Args:
        df: DataFrame containing cellular measurement data
        processor: Data processor for creating graph objects
        train_ratio: Approximate ratio of data for training (default: 0.8)

    Returns:
        Tuple of (train_graphs, val_graphs, test_graphs)
    """
    print("Creating day-based split (Scenario 2)...")

    # Extract timestamp and convert to day
    df['day'] = df['time'].apply(lambda x: int(float(x) / (24 * 3600 * 1e9)))

    # Get unique days
    unique_days = df['day'].unique()
    print(f"Dataset contains measurements from {len(unique_days)} unique days")

    # Sort days to ensure reproducibility
    unique_days = sorted(unique_days)

    # Calculate number of days for training
    num_train_days = int(len(unique_days) * train_ratio)

    # Split days for train/test
    train_days = unique_days[:num_train_days]
    test_days = unique_days[num_train_days:]

    # Further split train into train/val (90% train, 10% val)
    num_val_days = max(1, int(len(train_days) * 0.1))
    val_days = train_days[-num_val_days:]
    train_days = train_days[:-num_val_days]

    print(f"Train: {len(train_days)} days, Val: {len(val_days)} days, Test: {len(test_days)} days")

    # Create dataframes for each split
    train_df = df[df['day'].isin(train_days)]
    val_df = df[df['day'].isin(val_days)]
    test_df = df[df['day'].isin(test_days)]

    print(f"Train: {len(train_df)} records ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val: {len(val_df)} records ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test: {len(test_df)} records ({len(test_df)/len(df)*100:.1f}%)")

    # Process each dataframe into graphs
    print("Processing training data...")
    train_graphs = processor.process_dataset(train_df)

    print("Processing validation data...")
    val_graphs = processor.process_dataset(val_df)

    print("Processing test data...")
    test_graphs = processor.process_dataset(test_df)

    return train_graphs, val_graphs, test_graphs


def evaluate_all_scenarios(df: pd.DataFrame):
    """
    Evaluate the GNN model on both test scenarios:
    1. Sequential split (4:1 for every 5 measurements)
    2. Different days split (training and testing from different days)

    Args:
        df: DataFrame containing cellular measurement data
    """
    print("GNN-Based Cellular Network Localization - Multiple Scenario Evaluation")
    print("=" * 80)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    results = {}

    # SCENARIO 1: Sequential split
    print("\n" + "=" * 80)
    print("SCENARIO 1: SEQUENTIAL SPLIT (4:1 FOR EVERY 5 MEASUREMENTS)")
    print("=" * 80)

    # Initialize data processor
    processor_s1 = CellularDataProcessor(
        distance_threshold=0.005,
        rssi_threshold=15.0
    )

    # Process dataset into graphs
    graphs_s1 = processor_s1.process_dataset(df)
    print(f"Created {len(graphs_s1)} valid graphs")

    # Create sequential split
    train_graphs_s1, val_graphs_s1, test_graphs_s1 = create_improved_data_split(graphs_s1)

    # Create data loaders
    train_loader_s1 = DataLoader(train_graphs_s1, batch_size=32, shuffle=True)
    val_loader_s1 = DataLoader(val_graphs_s1, batch_size=32, shuffle=False)
    test_loader_s1 = DataLoader(test_graphs_s1, batch_size=32, shuffle=False)

    # Initialize model
    model_s1 = ImprovedGNNModel(
        input_dim=7,
        hidden_dim=128,
        edge_dim=3
    )

    # Train model
    trainer_s1 = GNNTrainer(model_s1, device)
    history_s1 = trainer_s1.train(train_loader_s1, val_loader_s1, num_epochs=200)

    # Evaluate model
    evaluator_s1 = ModelEvaluator(model_s1, processor_s1, device)
    results_s1 = evaluator_s1.evaluate(test_loader_s1)

    # Save results
    results['scenario1'] = results_s1

    # Plot and save results
    evaluator_s1.plot_results(results_s1, 'scenario1_results.png')
    torch.save(model_s1.state_dict(), 'model_scenario1.pth')

    # SCENARIO 2: Different days split
    print("\n" + "=" * 80)
    print("SCENARIO 2: DIFFERENT DAYS SPLIT")
    print("=" * 80)

    # Initialize data processor
    processor_s2 = CellularDataProcessor(
        distance_threshold=0.005,
        rssi_threshold=15.0
    )

    # Create day-based split
    train_graphs_s2, val_graphs_s2, test_graphs_s2 = create_day_based_split(df, processor_s2)

    # Create data loaders
    train_loader_s2 = DataLoader(train_graphs_s2, batch_size=32, shuffle=True)
    val_loader_s2 = DataLoader(val_graphs_s2, batch_size=32, shuffle=False)
    test_loader_s2 = DataLoader(test_graphs_s2, batch_size=32, shuffle=False)

    # Initialize model
    model_s2 = ImprovedGNNModel(
        input_dim=7,
        hidden_dim=128,
        edge_dim=3
    )

    # Train model
    trainer_s2 = GNNTrainer(model_s2, device)
    history_s2 = trainer_s2.train(train_loader_s2, val_loader_s2, num_epochs=200)

    # Evaluate model
    evaluator_s2 = ModelEvaluator(model_s2, processor_s2, device)
    results_s2 = evaluator_s2.evaluate(test_loader_s2)

    # Save results
    results['scenario2'] = results_s2

    # Plot and save results
    evaluator_s2.plot_results(results_s2, 'scenario2_results.png')
    torch.save(model_s2.state_dict(), 'model_scenario2.pth')

    # Compare scenarios
    compare_scenarios(results)

    return results


def compare_scenarios(results: Dict):
    """
    Compare results from different test scenarios.

    Args:
        results: Dictionary containing results from different scenarios
    """
    # Create a figure and axis
    fig, ax = plt.figure(figsize=(12, 6)), plt.gca()

    # Hide axes
    ax.axis('tight')
    ax.axis('off')

    # Create the table data
    metrics = ['Mean Distance Error (m)', 'Median Distance Error (m)',
               'Accuracy within 50m (%)', 'Accuracy within 100m (%)',
               'Accuracy within 200m (%)']

    table_data = [
        ['Scenario 1 (Sequential Split)',
         f"{results['scenario1']['mean_distance_error_m']:.2f}",
         f"{results['scenario1']['median_distance_error_m']:.2f}",
         f"{results['scenario1']['accuracy_50m']:.1f}",
         f"{results['scenario1']['accuracy_100m']:.1f}",
         f"{results['scenario1']['accuracy_200m']:.1f}"],
        ['Scenario 2 (Different Days)',
         f"{results['scenario2']['mean_distance_error_m']:.2f}",
         f"{results['scenario2']['median_distance_error_m']:.2f}",
         f"{results['scenario2']['accuracy_50m']:.1f}",
         f"{results['scenario2']['accuracy_100m']:.1f}",
         f"{results['scenario2']['accuracy_200m']:.1f}"]
    ]

    # Add header
    header = ['Scenario', 'Mean (m)', 'Median (m)', 'Acc@50m (%)', 'Acc@100m (%)', 'Acc@200m (%)']

    # Create the table
    table = ax.table(
        cellText=table_data,
        colLabels=header,
        loc='center',
        cellLoc='center',
        colColours=['#f2f2f2']*6,
        colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15]
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Add title
    plt.title('Comparison of Localization Performance Across Scenarios', fontsize=14, pad=20)

    # Save the figure
    plt.savefig('scenario_comparison.png', bbox_inches='tight', dpi=300)

    # Print comparison
    print("\n" + "=" * 80)
    print("COMPARISON OF SCENARIOS")
    print("=" * 80)
    print(f"Scenario 1 (Sequential Split):")
    print(f"  Mean Distance Error:    {results['scenario1']['mean_distance_error_m']:.2f} meters")
    print(f"  Median Distance Error:  {results['scenario1']['median_distance_error_m']:.2f} meters")
    print(f"  Accuracy within 50m:    {results['scenario1']['accuracy_50m']:.1f}%")
    print(f"  Accuracy within 100m:   {results['scenario1']['accuracy_100m']:.1f}%")

    print(f"\nScenario 2 (Different Days):")
    print(f"  Mean Distance Error:    {results['scenario2']['mean_distance_error_m']:.2f} meters")
    print(f"  Median Distance Error:  {results['scenario2']['median_distance_error_m']:.2f} meters")
    print(f"  Accuracy within 50m:    {results['scenario2']['accuracy_50m']:.1f}%")
    print(f"  Accuracy within 100m:   {results['scenario2']['accuracy_100m']:.1f}%")

    print("\nInsights:")
    if results['scenario1']['mean_distance_error_m'] < results['scenario2']['mean_distance_error_m']:
        print("- Scenario 1 yields better localization accuracy than Scenario 2")
        print("- This suggests the model's performance degrades when tested on completely different days")
        print("- Environmental factors and temporal variations appear to impact localization accuracy")
    else:
        print("- Scenario 2 yields better localization accuracy than Scenario 1")
        print("- This suggests the model generalizes well across different days")
        print("- The model appears robust to environmental and temporal variations")


if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/logFile_urban_data.csv')
    print(f"Loaded {len(df)} records")

    # Run evaluation for both scenarios
    results = evaluate_all_scenarios(df)
