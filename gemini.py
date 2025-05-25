
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_add_pool
import torch.nn as nn

# --- Configuration ---
DATA_FILE = 'cellular_data.csv'  # Replace with your actual data file
N_CELL_FEATURES = 5  # lac, cid, cell_lat, cell_lon, rssi (raw input)
NODE_FEATURE_DIM = 3  # After processing: rssi_norm, dlat_cell_centroid, dlon_cell_centroid
OUTPUT_DIM = 2  # dlat_user_centroid, dlon_user_centroid
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 32  # Smaller batch size can be good for GNNs
EPOCHS = 50  # Adjust based on convergence
GNN_HIDDEN_DIM = 128
MLP_HIDDEN_DIM = 64
NUM_GNN_LAYERS = 2  # Number of GATv2Conv layers
NUM_HEADS = 4  # Number of attention heads for GATv2Conv
DROPOUT_RATE = 0.3

# --- 1. Data Loading and Preprocessing ---


def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} records.")

    processed_data = []
    # Max number of cells observed in any record, used to parse columns
    # The paper implies a fixed 'k' for some methods, but GNNs can handle variable
    # For parsing, find max number of cells from the 'cells' column

    # Assuming the file has a header like:
    # stt,lat_ref,lon_ref,time,cells,lac1,cid1,cell_lat1,cell_lon1,rssi1,lac2,cid2,...
    # We need to determine the maximum number of cell groups based on columns
    max_cell_groups = 0
    for col in df.columns:
        if col.startswith('lac'):
            group_num = int(col[3:])
            if group_num > max_cell_groups:
                max_cell_groups = group_num

    print(f"Maximum cell groups found in columns: {max_cell_groups}")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing records"):
        user_lat_ref = row['lat_ref']
        user_lon_ref = row['lon_ref']
        num_observed_cells = int(row['cells'])

        cells_info = []
        actual_cells_parsed = 0
        for i in range(1, max_cell_groups + 1):  # Iterate up to max_cell_groups
            # Check if essential cell data exists for this group
            # Essential data: lat, lon, rssi. LAC/CID might be missing if cell is not in DB
            # The sample data has empty strings for missing cell info.
            # We need at least cell_lat, cell_lon, rssi to be valid.
            lat_col = f'cell_lat{i}'
            lon_col = f'cell_lon{i}'
            rssi_col = f'rssi{i}'

            # Check if columns exist and are not NaN/empty
            if lat_col in row and lon_col in row and rssi_col in row and \
               pd.notna(row[lat_col]) and pd.notna(row[lon_col]) and pd.notna(row[rssi_col]) and \
               row[lat_col] != '' and row[lon_col] != '' and row[rssi_col] != '':
                try:
                    cell_data = {
                        'lac': int(row[f'lac{i}']) if pd.notna(row[f'lac{i}']) and row[f'lac{i}'] != '' else 0,  # Use 0 or specific placeholder
                        'cid': int(row[f'cid{i}']) if pd.notna(row[f'cid{i}']) and row[f'cid{i}'] != '' else 0,
                        'lat': float(row[lat_col]),
                        'lon': float(row[lon_col]),
                        'rssi': float(row[rssi_col])
                    }
                    cells_info.append(cell_data)
                    actual_cells_parsed += 1
                except ValueError:
                    # Skip this cell group if conversion fails
                    # print(f"Skipping cell group {i} due to ValueError for row {row['stt']}")
                    continue

            if actual_cells_parsed >= num_observed_cells:  # Optimization: stop if we parsed enough cells
                break

        if not cells_info:  # Skip if no valid cells found for this record
            continue

        processed_data.append({
            'user_lat': user_lat_ref,
            'user_lon': user_lon_ref,
            'cells': cells_info
        })

    print(f"Successfully processed {len(processed_data)} records with cell info.")
    return processed_data


def create_graph_dataset(processed_data_list):
    graph_list = []
    # Normalize RSSI: typically -113 dBm (weakest) to -51 dBm (strongest for UMTS/LTE)
    # We'll use a simple min-max normalization approach for RSSI based on typical GSM range.
    # The paper mentions RSSI range 0 to -113 dBm. Let's assume -113 is min, 0 is max.
    # Scaling: (RSSI - (-113)) / (0 - (-113)) = (RSSI + 113) / 113
    # Or, use StandardScaler across the dataset for RSSI, dLat_cell, dLon_cell

    # For simplicity, let's make RSSI positive and then scale. (RSSI + 120)
    # And lat/lon offsets are small, can be used directly or scaled too.

    rssi_values_for_scaling = []
    for record in processed_data_list:
        for cell in record['cells']:
            rssi_values_for_scaling.append(cell['rssi'])

    rssi_scaler = StandardScaler()
    if rssi_values_for_scaling:
        # Ensure it's a 2D array for scaler
        rssi_scaler.fit(np.array(rssi_values_for_scaling).reshape(-1, 1))
    else:  # Handle case with no RSSI values
        print("Warning: No RSSI values found for scaling.")

    for record_idx, record in enumerate(tqdm(processed_data_list, desc="Creating graphs")):
        cells = record['cells']
        if not cells:
            continue

        # 1. Calculate centroid of observed cells
        cell_lats = np.array([cell['lat'] for cell in cells])
        cell_lons = np.array([cell['lon'] for cell in cells])
        centroid_lat = np.mean(cell_lats)
        centroid_lon = np.mean(cell_lons)

        # 2. Node features
        node_features = []
        for cell in cells:
            # RSSI: Normalize (e.g. (val + 113) / 113 to get approx 0-1, or use StandardScaler)
            # For now, using StandardScaler fitted on all RSSIs
            rssi_norm = rssi_scaler.transform(np.array([[cell['rssi']]]))[0, 0] if rssi_values_for_scaling else 0.0

            # Delta lat/lon from centroid
            dlat_cell_centroid = cell['lat'] - centroid_lat
            dlon_cell_centroid = cell['lon'] - centroid_lon

            # Node feature: [normalized_rssi, dlat_to_centroid, dlon_to_centroid]
            # Could also include LAC/CID if embedded, but let's start simple.
            node_features.append([rssi_norm, dlat_cell_centroid, dlon_cell_centroid])

        x = torch.tensor(node_features, dtype=torch.float)

        # 3. Edge index (fully connected graph for co-observed cells)
        num_nodes = len(cells)
        if num_nodes <= 1:  # Need at least 2 nodes for edges. If 1, it's an isolated node.
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            adj = torch.ones(num_nodes, num_nodes)  # Adjacency matrix
            np.fill_diagonal(adj.numpy(), 0)  # No self-loops for simplicity here
            edge_index = adj.nonzero(as_tuple=False).t().contiguous()

        # 4. Target: User's lat/lon offset from the cell centroid
        user_dlat_centroid = record['user_lat'] - centroid_lat
        user_dlon_centroid = record['user_lon'] - centroid_lon
        y = torch.tensor([user_dlat_centroid, user_dlon_centroid], dtype=torch.float)

        # 5. Store centroid for reconstruction
        graph_data = Data(x=x, edge_index=edge_index, y=y)
        graph_data.centroid_lat = torch.tensor([centroid_lat], dtype=torch.float)
        graph_data.centroid_lon = torch.tensor([centroid_lon], dtype=torch.float)
        graph_data.user_lat_abs = torch.tensor([record['user_lat']], dtype=torch.float)  # For eval
        graph_data.user_lon_abs = torch.tensor([record['user_lon']], dtype=torch.float)  # For eval

        graph_list.append(graph_data)

    return graph_list, rssi_scaler  # Return scaler for use during inference

# --- 2. GNN Model Architecture ---


class GNNLocalizationModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, heads=4, dropout=0.3):
        super(GNNLocalizationModel, self).__init__()
        self.convs = torch.nn.ModuleList()

        # Input layer
        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout, concat=True))
        # Hidden layers
        for _ in range(num_layers - 1):
            # Input to subsequent GAT layers is hidden_channels * heads
            self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout, concat=True))

        # MLP for graph regression
        # Input to MLP is the output of the last GAT layer (hidden_channels * heads) after pooling
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * heads, MLP_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(MLP_HIDDEN_DIM, MLP_HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(MLP_HIDDEN_DIM // 2, out_channels)  # Predicts (dLat, dLon)
        )

        """
        Explanation of Layer Choices:
        - GATv2Conv: Graph Attention Network v2. Chosen over GCN or GATv1 as it often provides
          better performance by learning dynamic attention weights, making it more expressive.
          'concat=True' means outputs of multi-head attention are concatenated.
        - ReLU: Standard activation function for non-linearity.
        - Dropout: Regularization technique to prevent overfitting.
        - global_mean_pool (or global_add_pool): Aggregates node embeddings into a single graph-level embedding.
          `global_mean_pool` is often a good default. `global_add_pool` can also work well.
        - Linear Layers (MLP): Standard fully connected layers to map the graph embedding to the
          final 2D coordinate offset prediction.
        """

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)  # Apply ReLU after each GAT layer
            # No dropout directly after GAT in this setup, but it's in GAT's definition

        # Graph pooling
        x_graph = global_mean_pool(x, batch)  # Pool node features to get graph representation
        # x_graph = global_add_pool(x, batch)

        # Pass through MLP
        out = self.mlp(x_graph)
        return out

# --- 3. Training and Evaluation ---


def haversine_distance(lat1, lon1, lat2, lon2, R=6371000):  # R in meters
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance


def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data_batch in loader:
        data_batch = data_batch.to(device)
        optimizer.zero_grad()
        out = model(data_batch)
        loss = criterion(out, data_batch.y)  # y is (dlat_user_centroid, dlon_user_centroid)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data_batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_distances = []

    for data_batch in loader:
        data_batch = data_batch.to(device)
        out_offsets = model(data_batch)  # Predicted (dLat_offset, dLon_offset)
        loss = criterion(out_offsets, data_batch.y)
        total_loss += loss.item() * data_batch.num_graphs

        # Reconstruct absolute predicted coordinates
        pred_lat_abs = data_batch.centroid_lat.squeeze() + out_offsets[:, 0]
        pred_lon_abs = data_batch.centroid_lon.squeeze() + out_offsets[:, 1]

        true_lat_abs = data_batch.user_lat_abs.squeeze()
        true_lon_abs = data_batch.user_lon_abs.squeeze()

        for i in range(len(pred_lat_abs)):
            dist = haversine_distance(pred_lat_abs[i].item(), pred_lon_abs[i].item(),
                                      true_lat_abs[i].item(), true_lon_abs[i].item())
            all_distances.append(dist)

    mean_loss = total_loss / len(loader.dataset)
    mean_dist_err = np.mean(all_distances) if all_distances else float('nan')
    median_dist_err = np.median(all_distances) if all_distances else float('nan')

    return mean_loss, mean_dist_err, median_dist_err, all_distances


# --- Main Execution ---
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load and process data
    raw_data = load_and_preprocess_data(DATA_FILE)

    # Create graph dataset (this can take time)
    graph_dataset, rssi_scaler_obj = create_graph_dataset(raw_data)
    print(f"Created {len(graph_dataset)} graph samples.")

    if not graph_dataset:
        print("No graph data created. Exiting.")
        exit()

    # 2. Data Splitting (as per prompt: 16924 train, 4231 test)
    # Total records in paper: 21155
    # (16924/21155) ~ 0.8, (4231/21155) ~ 0.2
    # Sequential 5-record groups (4 train, 1 test)

    train_graphs = []
    test_graphs = []
    for i in range(0, len(graph_dataset) - 4, 5):  # Ensure we have full groups of 5
        train_graphs.extend(graph_dataset[i: i+4])
        test_graphs.append(graph_dataset[i+4])

    # Handle remaining records if not perfectly divisible by 5
    remaining_start_index = (len(graph_dataset) // 5) * 5
    if remaining_start_index < len(graph_dataset):
        # Add remaining to training, or distribute as per a more complex rule
        # For simplicity, add to training if any are left
        train_graphs.extend(graph_dataset[remaining_start_index:])

    print(f"Train graphs: {len(train_graphs)}, Test graphs: {len(test_graphs)}")

    # Check if split sizes match roughly (adjust if your dataset size is different)
    # target_train_size = 16924
    # target_test_size = 4231
    # if len(train_graphs) < target_train_size * 0.9 or len(test_graphs) < target_test_size * 0.9:
    # print(f"Warning: Split sizes ({len(train_graphs)}, {len(test_graphs)}) "
    #       f"differ significantly from target ({target_train_size}, {target_test_size}). "
    #       "This might be due to initial dataset size or records skipped during preprocessing.")

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Initialize Model, Optimizer, Criterion
    model = GNNLocalizationModel(
        in_channels=NODE_FEATURE_DIM,
        hidden_channels=GNN_HIDDEN_DIM,
        out_channels=OUTPUT_DIM,
        num_layers=NUM_GNN_LAYERS,
        heads=NUM_HEADS,
        dropout=DROPOUT_RATE
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)  # L2 regularization
    criterion = nn.MSELoss()  # Mean Squared Error for regression on offsets

    """
    Loss Function: MSELoss is chosen because we are predicting continuous lat/lon offsets.
                   It penalizes larger errors more heavily.
    Optimizer: AdamW is an extension of Adam that decouples weight decay from the gradient updates,
               which can lead to better generalization.
    """

    # 4. Training Loop
    print("\n--- Starting Training ---")
    best_test_median_err = float('inf')
    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        test_loss, test_mean_err, test_median_err, _ = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch:02d}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.6f}, "
              f"Test Mean Err: {test_mean_err:.2f}m, Test Median Err: {test_median_err:.2f}m")

        if test_median_err < best_test_median_err:
            best_test_median_err = test_median_err
            # torch.save(model.state_dict(), 'best_gnn_model.pth')
            print(f"  New best model saved with median error: {best_test_median_err:.2f}m")

    # 5. Final Evaluation on Test Set (using the last epoch's model or best saved model)
    print("\n--- Final Evaluation on Test Set ---")
    # model.load_state_dict(torch.load('best_gnn_model.pth')) # If saved
    final_test_loss, final_mean_err, final_median_err, all_distances = evaluate(model, test_loader, criterion, device)

    rmse = np.sqrt(np.mean(np.array(all_distances)**2)) if all_distances else float('nan')
    percentile_90 = np.percentile(all_distances, 90) if all_distances else float('nan')

    print(f"Final Test MSE Loss: {final_test_loss:.6f}")
    print(f"Final Test Mean Distance Error: {final_mean_err:.2f} m")
    print(f"Final Test Median Distance Error: {final_median_err:.2f} m")
    print(f"Final Test RMSE Distance Error: {rmse:.2f} m")
    print(f"Final Test 90th Percentile Error: {percentile_90:.2f} m")

    # To plot histogram of errors:
    # import matplotlib.pyplot as plt
    # plt.hist(all_distances, bins=50)
    # plt.xlabel("Distance Error (m)")
    # plt.ylabel("Frequency")
    # plt.title("GNN Model - Distribution of Localization Errors")
    # plt.show()
