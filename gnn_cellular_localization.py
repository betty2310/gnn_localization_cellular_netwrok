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
        Convert a cellular measurement record into a PyTorch Geometric graph.

        Graph Construction Strategy:
        - Nodes: Each detected cell becomes a node with features [lat, lon, rssi]
        - Edges: Connect cells within distance_threshold (spatial proximity)
        - Node features are normalized for better training stability
        - Self-loops are added to ensure all nodes receive their own information

        Args:
            record: Parsed cellular measurement record

        Returns:
            PyTorch Geometric Data object representing the measurement as a graph
        """
        cells = record['cells']
        num_nodes = len(cells)

        if num_nodes == 0:
            return None

        # Create node features: [normalized_lat, normalized_lon, normalized_rssi]
        node_features = []
        positions = []

        for cell in cells:
            node_features.append([cell['lat'], cell['lon'], cell['rssi']])
            positions.append([cell['lat'], cell['lon']])

        node_features = np.array(node_features)
        positions = np.array(positions)

        # Create edges based on spatial proximity
        edge_indices = []

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # No self-loops initially
                    dist = self.calculate_distance(
                        positions[i][0], positions[i][1],
                        positions[j][0], positions[j][1]
                    )
                    if dist <= self.distance_threshold:
                        edge_indices.append([i, j])

        # Add self-loops for all nodes (important for GNN message passing)
        for i in range(num_nodes):
            edge_indices.append([i, i])

        # Convert to tensor format
        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        else:
            # If no edges, create self-loops only
            edge_index = torch.tensor([[i, i] for i in range(num_nodes)],
                                      dtype=torch.long).t().contiguous()

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

        return Data(x=x, edge_index=edge_index, y=y,
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
        Normalize node features across all graphs for better training stability.
        Stores original coordinate ranges for later denormalization.

        Args:
            graphs: List of graph objects to normalize
        """
        # Collect all node features
        all_coords = []
        all_rssi = []
        all_targets = []  # Store target coordinates for proper scaling

        for graph in graphs:
            if graph is None or not hasattr(graph, 'x') or graph.x is None:
                continue

            coords = graph.x[:, :2].numpy()  # lat, lon
            rssi = graph.x[:, 2:3].numpy()   # rssi
            all_coords.append(coords)
            all_rssi.append(rssi)

            # Handle target coordinates properly
            target = graph.y.view(-1, 2).numpy() if hasattr(graph.y, 'view') else graph.y.reshape(1, 2)
            all_targets.append(target)

        if not all_coords or not all_rssi or not all_targets:
            print("Warning: Empty feature lists. Check graph creation.")
            return

        all_coords = np.vstack(all_coords)
        all_rssi = np.vstack(all_rssi)
        all_targets = np.vstack(all_targets)

        # Store min/max values for denormalization during evaluation
        self.lat_min = float(np.min(all_coords[:, 0]))
        self.lat_max = float(np.max(all_coords[:, 0]))
        self.lon_min = float(np.min(all_coords[:, 1]))
        self.lon_max = float(np.max(all_coords[:, 1]))

        # Store target coordinate ranges
        self.target_lat_min = float(np.min(all_targets[:, 0]))
        self.target_lat_max = float(np.max(all_targets[:, 0]))
        self.target_lon_min = float(np.min(all_targets[:, 1]))
        self.target_lon_max = float(np.max(all_targets[:, 1]))

        print(f"Coordinate ranges - Lat: [{self.lat_min:.6f}, {self.lat_max:.6f}], Lon: [{self.lon_min:.6f}, {self.lon_max:.6f}]")
        print(f"Target ranges - Lat: [{self.target_lat_min:.6f}, {self.target_lat_max:.6f}], Lon: [{self.target_lon_min:.6f}, {self.target_lon_max:.6f}]")

        # Fit scalers
        self.scaler_coords.fit(all_coords)
        self.scaler_rssi.fit(all_rssi)

        # Apply normalization
        for graph in graphs:
            if graph is None or not hasattr(graph, 'x') or graph.x is None:
                continue

            coords = graph.x[:, :2].numpy()
            rssi = graph.x[:, 2:3].numpy()

            coords_norm = self.scaler_coords.transform(coords)
            rssi_norm = self.scaler_rssi.transform(rssi)

            graph.x = torch.tensor(
                np.hstack([coords_norm, rssi_norm]),
                dtype=torch.float
            )

            # Also normalize target coordinates using the same scaler
            if hasattr(graph.y, 'view'):
                target = graph.y.view(-1, 2).numpy()
            else:
                target = np.array([graph.y[0], graph.y[1]]).reshape(1, 2)

            target_norm = self.scaler_coords.transform(target)
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

    def __init__(self, model: GNNLocalizationModel, device=None):
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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

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
              num_epochs: int = 200) -> Dict:
        """
        Full training loop with early stopping.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs

        Returns:
            Training history dictionary
        """
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(train_loader)

            # Validation
            val_loss = self.validate(val_loader)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Record losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_gnn_model.pth')
            else:
                self.patience_counter += 1

            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}, LR: {self.optimizer.param_groups[0]['lr']:.2e}")

            # Early stopping
            if self.patience_counter >= self.max_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Load best model
        self.model.load_state_dict(torch.load('best_gnn_model.pth'))

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }


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

    def __init__(self, model: GNNLocalizationModel, processor: CellularDataProcessor, device=None):
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
            'CNN-CellImage', 'GNN (Our Method)'
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


def main():
    """
    Main execution function for GNN-based cellular localization.
    """
    print("GNN-Based Cellular Network Localization System")
    print("=" * 50)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and process data
    print("\n1. Loading and processing data...")
    df = pd.read_csv('data/logFile_urban_data.csv')
    print(f"Loaded {len(df)} records")

    # Initialize data processor with smaller distance threshold
    processor = CellularDataProcessor(distance_threshold=0.005)  # Reduced from 0.01

    # Process dataset into graphs
    graphs = processor.process_dataset(df)
    print(f"Created {len(graphs)} valid graphs")

    # Create sequential train/test split
    print("\n2. Creating train/test split...")
    train_graphs, test_graphs = create_sequential_split(graphs, group_size=5)

    # Create data loaders with smaller batch size
    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)  # Reduced from 32
    test_loader = DataLoader(test_graphs, batch_size=16, shuffle=False)   # Reduced from 32

    # Initialize model
    print("\n3. Initializing GNN model...")
    model = GNNLocalizationModel(
        input_dim=3,      # lat, lon, rssi
        hidden_dim=128,   # Increased from 64
        num_layers=4,     # Increased from 3
        num_heads=4,      # Reduced from 8 for better stability
        dropout=0.2       # Reduced from 0.3
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Initialize trainer
    trainer = GNNTrainer(model, device)

    # Train model
    print("\n4. Training model...")
    # Create validation split from training data
    train_graphs_split, val_graphs = train_test_split(
        train_graphs, test_size=0.2, random_state=42
    )

    train_loader_split = DataLoader(train_graphs_split, batch_size=16, shuffle=True)  # Reduced from 32
    val_loader = DataLoader(val_graphs, batch_size=16, shuffle=False)                 # Reduced from 32

    history = trainer.train(train_loader_split, val_loader, num_epochs=300)  # Increased from 200

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


if __name__ == "__main__":
    model, results, history = main()
