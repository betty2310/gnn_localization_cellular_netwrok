"""
Comparative Analysis Framework for Cellular Network Localization Methods

This module provides a comprehensive comparison between:
1. GNN-based localization (our implementation)
2. Fingerprint-based localization (traditional method)
3. CNN-CellImage localization (image-based approach)

The framework evaluates performance, computational efficiency, and practical considerations
for each method using the same dataset and evaluation metrics.
"""

from gnn_cellular_localization import (
    CellularDataProcessor, GNNLocalizationModel, GNNTrainer, ModelEvaluator,
    create_sequential_split
)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import time
import warnings
warnings.filterwarnings('ignore')

# Import our GNN implementation


class FingerprintLocalization:
    """
    Traditional fingerprint-based localization method.

    Method Overview:
    ================
    1. Offline Phase: Build fingerprint database mapping RSSI patterns to locations
    2. Online Phase: Match current RSSI measurements to database entries
    3. Location Estimation: Use k-nearest neighbors or weighted average

    Advantages:
    - Simple and interpretable
    - No training required
    - Works well with sufficient fingerprint density

    Disadvantages:
    - Requires extensive site survey for fingerprint collection
    - Database maintenance overhead
    - Poor generalization to new environments
    - Sensitive to environmental changes
    """

    def __init__(self, k_neighbors: int = 5, distance_metric: str = 'euclidean'):
        """
        Initialize fingerprint localization system.

        Args:
            k_neighbors: Number of nearest neighbors for location estimation
            distance_metric: Distance metric for RSSI pattern matching
        """
        self.k_neighbors = k_neighbors
        self.distance_metric = distance_metric
        self.fingerprint_db = None
        self.knn_model = None
        self.scaler = StandardScaler()

    def build_fingerprint_database(self, train_data: List[Dict]):
        """
        Build fingerprint database from training data.

        Args:
            train_data: List of parsed cellular measurement records
        """
        fingerprints = []
        locations = []

        print(f"Building fingerprint database from {len(train_data)} records...")

        for record in train_data:
            # Create RSSI fingerprint vector
            # Use fixed-size vector with known cell IDs
            fingerprint = self._create_fingerprint_vector(record)
            if fingerprint is not None:
                fingerprints.append(fingerprint)
                locations.append([record['lat_ref'], record['lon_ref']])

        fingerprints = np.array(fingerprints)
        locations = np.array(locations)

        # Normalize RSSI values
        fingerprints_norm = self.scaler.fit_transform(fingerprints)

        # Train KNN model
        self.knn_model = KNeighborsRegressor(
            n_neighbors=self.k_neighbors,
            metric=self.distance_metric,
            weights='distance'  # Weight by inverse distance
        )

        self.knn_model.fit(fingerprints_norm, locations)

        print(f"Fingerprint database built with {len(fingerprints)} entries")

    def _create_fingerprint_vector(self, record: Dict, max_cells: int = 20) -> Optional[np.ndarray]:
        """
        Create fixed-size RSSI fingerprint vector from measurement record.

        Args:
            record: Parsed cellular measurement record
            max_cells: Maximum number of cells to include in fingerprint

        Returns:
            Fixed-size RSSI vector or None if insufficient data
        """
        if len(record['cells']) == 0:
            return None

        # Create fingerprint vector with cell ID mapping
        fingerprint = np.full(max_cells, -120.0)  # Default to very weak signal

        for i, cell in enumerate(record['cells'][:max_cells]):
            fingerprint[i] = cell['rssi']

        return fingerprint

    def predict(self, test_data: List[Dict]) -> np.ndarray:
        """
        Predict locations for test data using fingerprint matching.

        Args:
            test_data: List of test measurement records

        Returns:
            Predicted coordinates array
        """
        if self.knn_model is None:
            raise ValueError("Fingerprint database not built. Call build_fingerprint_database first.")

        test_fingerprints = []

        for record in test_data:
            fingerprint = self._create_fingerprint_vector(record)
            if fingerprint is not None:
                test_fingerprints.append(fingerprint)
            else:
                # Handle missing data with zeros
                test_fingerprints.append(np.full(20, -120.0))

        test_fingerprints = np.array(test_fingerprints)
        test_fingerprints_norm = self.scaler.transform(test_fingerprints)

        predictions = self.knn_model.predict(test_fingerprints_norm)
        return predictions


class CNNCellImageLocalization:
    """
    CNN-based localization using cell information converted to images.

    Method Overview:
    ================
    1. Convert cell measurements to 2D images with RSSI as pixel intensities
    2. Use CNN to learn spatial patterns from cell images
    3. Regression head predicts coordinates from learned features

    Advantages:
    - Leverages powerful CNN architectures
    - Can learn complex spatial patterns
    - End-to-end trainable

    Disadvantages:
    - Artificial conversion of non-image data to images
    - Loss of actual spatial relationships
    - Fixed grid structure doesn't match network topology
    - Requires careful image construction strategy
    """

    def __init__(self, image_size: int = 32, device: str = 'cuda'):
        """
        Initialize CNN-CellImage localization system.

        Args:
            image_size: Size of generated cell images
            device: Training device
        """
        self.image_size = image_size
        self.device = device
        self.model = None
        self.scaler_coords = StandardScaler()
        self.scaler_rssi = StandardScaler()

    def _create_cell_image(self, record: Dict) -> np.ndarray:
        """
        Convert cellular measurement to 2D image representation.

        Strategy:
        - Map cell coordinates to image grid positions
        - Use RSSI values as pixel intensities
        - Apply Gaussian smoothing for spatial continuity

        Args:
            record: Parsed cellular measurement record

        Returns:
            2D image array representing cell measurements
        """
        image = np.zeros((self.image_size, self.image_size))

        if len(record['cells']) == 0:
            return image

        # Get coordinate bounds for normalization
        lats = [cell['lat'] for cell in record['cells']]
        lons = [cell['lon'] for cell in record['cells']]

        if len(lats) == 1:
            # Single cell - place at center
            center = self.image_size // 2
            rssi_norm = (record['cells'][0]['rssi'] + 120) / 60  # Normalize RSSI to [0,1]
            image[center, center] = max(0, min(1, rssi_norm))
        else:
            # Multiple cells - map to image coordinates
            lat_min, lat_max = min(lats), max(lats)
            lon_min, lon_max = min(lons), max(lons)

            # Avoid division by zero
            lat_range = max(lat_max - lat_min, 1e-6)
            lon_range = max(lon_max - lon_min, 1e-6)

            for cell in record['cells']:
                # Map to image coordinates
                x = int((cell['lat'] - lat_min) / lat_range * (self.image_size - 1))
                y = int((cell['lon'] - lon_min) / lon_range * (self.image_size - 1))

                # Normalize RSSI to [0,1]
                rssi_norm = (cell['rssi'] + 120) / 60
                rssi_norm = max(0, min(1, rssi_norm))

                # Place in image with Gaussian smoothing
                self._add_gaussian_blob(image, x, y, rssi_norm, sigma=2.0)

        return image

    def _add_gaussian_blob(self, image: np.ndarray, x: int, y: int,
                           intensity: float, sigma: float = 2.0):
        """
        Add Gaussian blob to image at specified position.

        Args:
            image: Target image array
            x, y: Center coordinates
            intensity: Peak intensity value
            sigma: Gaussian standard deviation
        """
        size = self.image_size
        for i in range(max(0, x-3*int(sigma)), min(size, x+3*int(sigma)+1)):
            for j in range(max(0, y-3*int(sigma)), min(size, y+3*int(sigma)+1)):
                dist_sq = (i - x)**2 + (j - y)**2
                value = intensity * np.exp(-dist_sq / (2 * sigma**2))
                image[i, j] = max(image[i, j], value)

    def build_model(self) -> nn.Module:
        """
        Build CNN model for cell image localization.

        Returns:
            CNN model for coordinate regression
        """
        class CNNCellImageModel(nn.Module):
            def __init__(self, image_size: int = 32):
                super(CNNCellImageModel, self).__init__()

                # CNN feature extractor
                self.features = nn.Sequential(
                    # First conv block
                    nn.Conv2d(1, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.MaxPool2d(2),

                    # Second conv block
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2),

                    # Third conv block
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(2),

                    # Fourth conv block
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((2, 2))
                )

                # Regression head
                self.regressor = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(256 * 4, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 2)  # lat, lon
                )

            def forward(self, x):
                features = self.features(x)
                output = self.regressor(features)
                return output

        model = CNNCellImageModel(self.image_size)
        return model.to(self.device)

    def train(self, train_data: List[Dict], val_data: List[Dict],
              num_epochs: int = 100, batch_size: int = 32) -> Dict:
        """
        Train CNN model on cell image data.

        Args:
            train_data: Training measurement records
            val_data: Validation measurement records
            num_epochs: Number of training epochs
            batch_size: Training batch size

        Returns:
            Training history dictionary
        """
        # Convert data to images
        print("Converting measurements to cell images...")
        train_images, train_targets = self._prepare_data(train_data)
        val_images, val_targets = self._prepare_data(val_data)

        # Create datasets and loaders
        train_dataset = CellImageDataset(train_images, train_targets)
        val_dataset = CellImageDataset(val_images, val_targets)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        self.model = self.build_model()

        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        criterion = nn.MSELoss()

        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 20

        print(f"Training CNN model for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_images, batch_targets in train_loader:
                batch_images = batch_images.to(self.device)
                batch_targets = batch_targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_images)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_images, batch_targets in val_loader:
                    batch_images = batch_images.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    outputs = self.model(batch_images)
                    loss = criterion(outputs, batch_targets)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Record losses
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_cnn_model.pth')
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}")

            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Load best model
        self.model.load_state_dict(torch.load('best_cnn_model.pth'))

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss
        }

    def _prepare_data(self, data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert measurement records to images and targets.

        Args:
            data: List of measurement records

        Returns:
            Tuple of (images, targets)
        """
        images = []
        targets = []

        for record in data:
            image = self._create_cell_image(record)
            images.append(image)
            targets.append([record['lat_ref'], record['lon_ref']])

        images = np.array(images)
        targets = np.array(targets)

        # Add channel dimension for CNN
        images = images[:, np.newaxis, :, :]

        return images, targets

    def predict(self, test_data: List[Dict]) -> np.ndarray:
        """
        Predict locations using trained CNN model.

        Args:
            test_data: Test measurement records

        Returns:
            Predicted coordinates
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        test_images, _ = self._prepare_data(test_data)
        test_dataset = CellImageDataset(test_images, np.zeros((len(test_images), 2)))
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch_images, _ in test_loader:
                batch_images = batch_images.to(self.device)
                outputs = self.model(batch_images)
                predictions.append(outputs.cpu().numpy())

        return np.vstack(predictions)


class CellImageDataset(Dataset):
    """Dataset class for CNN cell image training."""

    def __init__(self, images: np.ndarray, targets: np.ndarray):
        self.images = torch.FloatTensor(images)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]


class ComparativeAnalyzer:
    """
    Comprehensive comparison framework for localization methods.

    Compares GNN, Fingerprint, and CNN-CellImage approaches across:
    1. Accuracy metrics
    2. Computational efficiency
    3. Training time and memory usage
    4. Robustness to data variations
    """

    def __init__(self, device: str = 'cuda'):
        """
        Initialize comparative analyzer.

        Args:
            device: Computing device for neural network methods
        """
        self.device = device
        self.results = {}

    def haversine_distance(self, lat1: np.ndarray, lon1: np.ndarray,
                           lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
        """Calculate haversine distance between coordinate pairs."""
        R = 6371000  # Earth radius in meters

        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)

        a = (np.sin(dlat/2)**2 +
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

        return R * c

    def evaluate_method(self, method_name: str, predictions: np.ndarray,
                        targets: np.ndarray, training_time: float,
                        inference_time: float) -> Dict:
        """
        Evaluate a localization method with comprehensive metrics.

        Args:
            method_name: Name of the method
            predictions: Predicted coordinates
            targets: Ground truth coordinates
            training_time: Training time in seconds
            inference_time: Inference time in seconds

        Returns:
            Dictionary of evaluation metrics
        """
        # Basic regression metrics
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mse)

        # Geographic distance errors
        distances = self.haversine_distance(
            targets[:, 0], targets[:, 1],
            predictions[:, 0], predictions[:, 1]
        )

        mean_distance_error = np.mean(distances)
        median_distance_error = np.median(distances)
        std_distance_error = np.std(distances)

        # Accuracy at thresholds
        thresholds = [10, 25, 50, 100, 200, 500]
        accuracies = {}
        for threshold in thresholds:
            accuracy = np.mean(distances <= threshold) * 100
            accuracies[f'accuracy_{threshold}m'] = accuracy

        # Coordinate-wise errors
        lat_mae = mean_absolute_error(targets[:, 0], predictions[:, 0])
        lon_mae = mean_absolute_error(targets[:, 1], predictions[:, 1])

        results = {
            'method': method_name,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mean_distance_error_m': mean_distance_error,
            'median_distance_error_m': median_distance_error,
            'std_distance_error_m': std_distance_error,
            'lat_mae': lat_mae,
            'lon_mae': lon_mae,
            'training_time_s': training_time,
            'inference_time_s': inference_time,
            **accuracies
        }

        return results

    def run_comparative_analysis(self, df: pd.DataFrame) -> Dict:
        """
        Run comprehensive comparison of all three methods.

        Args:
            df: Dataset DataFrame

        Returns:
            Dictionary containing results for all methods
        """
        print("="*60)
        print("COMPREHENSIVE LOCALIZATION METHOD COMPARISON")
        print("="*60)

        # Process data
        processor = CellularDataProcessor(distance_threshold=0.01)
        graphs = processor.process_dataset(df)

        # Create train/test split
        train_graphs, test_graphs = create_sequential_split(graphs, group_size=5)

        # Convert graphs back to records for non-GNN methods
        train_records = self._graphs_to_records(train_graphs, processor)
        test_records = self._graphs_to_records(test_graphs, processor)

        # Split training data for validation
        train_records_split, val_records = train_test_split(
            train_records, test_size=0.2, random_state=42
        )

        results = {}

        # 1. GNN Method
        print("\n1. Evaluating GNN Method...")
        gnn_results = self._evaluate_gnn_method(
            train_graphs, test_graphs, val_records
        )
        results['GNN'] = gnn_results

        # 2. Fingerprint Method
        print("\n2. Evaluating Fingerprint Method...")
        fingerprint_results = self._evaluate_fingerprint_method(
            train_records, test_records
        )
        results['Fingerprint'] = fingerprint_results

        # 3. CNN-CellImage Method
        print("\n3. Evaluating CNN-CellImage Method...")
        cnn_results = self._evaluate_cnn_method(
            train_records_split, val_records, test_records
        )
        results['CNN-CellImage'] = cnn_results

        # Generate comparison report
        self._generate_comparison_report(results)

        # Create visualization
        self._create_comparison_plots(results)

        return results

    def _graphs_to_records(self, graphs, processor) -> List[Dict]:
        """Convert graph objects back to record format for non-GNN methods."""
        records = []

        for graph in graphs:
            # Extract information from graph
            cells = []
            for i in range(graph.num_nodes):
                # Denormalize coordinates and RSSI
                lat_norm, lon_norm, rssi_norm = graph.x[i].numpy()

                # Reverse normalization (approximate)
                lat = lat_norm * 0.01 + 20.95  # Approximate based on data range
                lon = lon_norm * 0.01 + 105.77
                rssi = rssi_norm * 20 - 80  # Approximate RSSI range

                cells.append({
                    'lat': lat,
                    'lon': lon,
                    'rssi': rssi,
                    'lac': 12125,  # Default LAC
                    'cid': 30555 + i  # Approximate CID
                })

            record = {
                'lat_ref': graph.y[0].item(),
                'lon_ref': graph.y[1].item(),
                'cells': cells,
                'stt': len(records) + 1,
                'time': 0,
                'num_cells': len(cells)
            }
            records.append(record)

        return records

    def _evaluate_gnn_method(self, train_graphs, test_graphs, val_records) -> Dict:
        """Evaluate GNN method performance."""
        start_time = time.time()

        # Initialize and train GNN model
        model = GNNLocalizationModel(input_dim=3, hidden_dim=64, num_layers=3)
        trainer = GNNTrainer(model, self.device)

        # Create data loaders
        from torch_geometric.data import DataLoader
        train_loader = DataLoader(train_graphs[:int(0.8*len(train_graphs))],
                                  batch_size=32, shuffle=True)
        val_loader = DataLoader(train_graphs[int(0.8*len(train_graphs)):],
                                batch_size=32, shuffle=False)

        # Train model
        history = trainer.train(train_loader, val_loader, num_epochs=50)
        training_time = time.time() - start_time

        # Evaluate on test set
        test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
        evaluator = ModelEvaluator(model, self.device)

        start_inference = time.time()
        eval_results = evaluator.evaluate(test_loader)
        inference_time = time.time() - start_inference

        # Extract predictions and targets
        predictions = eval_results['predictions']
        targets = eval_results['targets']

        return self.evaluate_method('GNN', predictions, targets,
                                    training_time, inference_time)

    def _evaluate_fingerprint_method(self, train_records, test_records) -> Dict:
        """Evaluate fingerprint method performance."""
        start_time = time.time()

        # Initialize and train fingerprint model
        fingerprint_model = FingerprintLocalization(k_neighbors=5)
        fingerprint_model.build_fingerprint_database(train_records)

        training_time = time.time() - start_time

        # Predict on test set
        start_inference = time.time()
        predictions = fingerprint_model.predict(test_records)
        inference_time = time.time() - start_inference

        # Extract targets
        targets = np.array([[r['lat_ref'], r['lon_ref']] for r in test_records])

        return self.evaluate_method('Fingerprint', predictions, targets,
                                    training_time, inference_time)

    def _evaluate_cnn_method(self, train_records, val_records, test_records) -> Dict:
        """Evaluate CNN-CellImage method performance."""
        start_time = time.time()

        # Initialize and train CNN model
        cnn_model = CNNCellImageLocalization(image_size=32, device=self.device)
        history = cnn_model.train(train_records, val_records, num_epochs=50)

        training_time = time.time() - start_time

        # Predict on test set
        start_inference = time.time()
        predictions = cnn_model.predict(test_records)
        inference_time = time.time() - start_inference

        # Extract targets
        targets = np.array([[r['lat_ref'], r['lon_ref']] for r in test_records])

        return self.evaluate_method('CNN-CellImage', predictions, targets,
                                    training_time, inference_time)

    def _generate_comparison_report(self, results: Dict):
        """Generate detailed comparison report."""
        print("\n" + "="*80)
        print("DETAILED COMPARISON REPORT")
        print("="*80)

        methods = ['GNN', 'Fingerprint', 'CNN-CellImage']

        # Performance comparison table
        print("\nPERFORMANCE METRICS:")
        print("-" * 80)
        print(f"{'Metric':<25} {'GNN':<15} {'Fingerprint':<15} {'CNN-CellImage':<15}")
        print("-" * 80)

        metrics = [
            ('Mean Distance Error (m)', 'mean_distance_error_m'),
            ('Median Distance Error (m)', 'median_distance_error_m'),
            ('RMSE', 'rmse'),
            ('MAE', 'mae'),
            ('Accuracy @ 50m (%)', 'accuracy_50m'),
            ('Accuracy @ 100m (%)', 'accuracy_100m'),
            ('Training Time (s)', 'training_time_s'),
            ('Inference Time (s)', 'inference_time_s')
        ]

        for metric_name, metric_key in metrics:
            values = []
            for method in methods:
                if method in results:
                    value = results[method][metric_key]
                    if 'time' in metric_key:
                        values.append(f"{value:.2f}")
                    elif 'accuracy' in metric_key:
                        values.append(f"{value:.1f}")
                    else:
                        values.append(f"{value:.3f}")
                else:
                    values.append("N/A")

            print(f"{metric_name:<25} {values[0]:<15} {values[1]:<15} {values[2]:<15}")

        # Best method analysis
        print("\n" + "="*80)
        print("BEST METHOD ANALYSIS:")
        print("="*80)

        best_accuracy = min(results.keys(),
                            key=lambda x: results[x]['mean_distance_error_m'])
        best_speed = min(results.keys(),
                         key=lambda x: results[x]['training_time_s'])

        print(f"Best Accuracy: {best_accuracy} "
              f"({results[best_accuracy]['mean_distance_error_m']:.2f}m mean error)")
        print(f"Fastest Training: {best_speed} "
              f"({results[best_speed]['training_time_s']:.2f}s)")

        # Method characteristics
        print("\nMETHOD CHARACTERISTICS:")
        print("-" * 80)

        characteristics = {
            'GNN': [
                "✓ Leverages natural graph structure of cellular networks",
                "✓ Adaptive to varying numbers of detected cells",
                "✓ Learns complex spatial relationships",
                "✗ Requires GPU for efficient training",
                "✗ More complex implementation"
            ],
            'Fingerprint': [
                "✓ Simple and interpretable",
                "✓ No training required",
                "✓ Fast inference",
                "✗ Requires extensive site survey",
                "✗ Poor generalization to new environments"
            ],
            'CNN-CellImage': [
                "✓ Leverages proven CNN architectures",
                "✓ End-to-end trainable",
                "✗ Artificial image conversion loses spatial meaning",
                "✗ Fixed grid structure doesn't match network topology",
                "✗ Requires careful image construction"
            ]
        }

        for method, chars in characteristics.items():
            print(f"\n{method}:")
            for char in chars:
                print(f"  {char}")

    def _create_comparison_plots(self, results: Dict):
        """Create comprehensive comparison visualizations."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        methods = list(results.keys())
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        # 1. Distance Error Comparison
        distance_errors = [results[method]['mean_distance_error_m'] for method in methods]
        bars1 = axes[0, 0].bar(methods, distance_errors, color=colors)
        axes[0, 0].set_ylabel('Mean Distance Error (meters)')
        axes[0, 0].set_title('Mean Distance Error Comparison')
        axes[0, 0].grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars1, distance_errors):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                            f'{value:.1f}m', ha='center', va='bottom')

        # 2. Accuracy at 100m Comparison
        accuracy_100m = [results[method]['accuracy_100m'] for method in methods]
        bars2 = axes[0, 1].bar(methods, accuracy_100m, color=colors)
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Accuracy within 100m')
        axes[0, 1].grid(True, alpha=0.3)

        for bar, value in zip(bars2, accuracy_100m):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                            f'{value:.1f}%', ha='center', va='bottom')

        # 3. Training Time Comparison
        training_times = [results[method]['training_time_s'] for method in methods]
        bars3 = axes[0, 2].bar(methods, training_times, color=colors)
        axes[0, 2].set_ylabel('Training Time (seconds)')
        axes[0, 2].set_title('Training Time Comparison')
        axes[0, 2].grid(True, alpha=0.3)

        for bar, value in zip(bars3, training_times):
            axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                            f'{value:.1f}s', ha='center', va='bottom')

        # 4. Multi-threshold Accuracy Comparison
        thresholds = [10, 25, 50, 100, 200, 500]
        for i, method in enumerate(methods):
            accuracies = [results[method][f'accuracy_{t}m'] for t in thresholds]
            axes[1, 0].plot(thresholds, accuracies, marker='o',
                            label=method, color=colors[i], linewidth=2)

        axes[1, 0].set_xlabel('Distance Threshold (meters)')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].set_title('Accuracy vs Distance Threshold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xscale('log')

        # 5. RMSE vs Training Time Trade-off
        rmse_values = [results[method]['rmse'] for method in methods]
        axes[1, 1].scatter(training_times, rmse_values,
                           c=colors, s=100, alpha=0.7)

        for i, method in enumerate(methods):
            axes[1, 1].annotate(method,
                                (training_times[i], rmse_values[i]),
                                xytext=(5, 5), textcoords='offset points')

        axes[1, 1].set_xlabel('Training Time (seconds)')
        axes[1, 1].set_ylabel('RMSE')
        axes[1, 1].set_title('Accuracy vs Training Time Trade-off')
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Overall Performance Radar Chart
        # Normalize metrics for radar chart
        metrics_for_radar = ['mean_distance_error_m', 'training_time_s',
                             'inference_time_s', 'accuracy_100m']

        # Create performance scores (higher is better)
        performance_scores = {}
        for method in methods:
            scores = []
            # Distance error (lower is better, so invert)
            max_dist = max(results[m]['mean_distance_error_m'] for m in methods)
            scores.append(100 * (1 - results[method]['mean_distance_error_m'] / max_dist))

            # Training time (lower is better, so invert)
            max_train = max(results[m]['training_time_s'] for m in methods)
            scores.append(100 * (1 - results[method]['training_time_s'] / max_train))

            # Inference time (lower is better, so invert)
            max_infer = max(results[m]['inference_time_s'] for m in methods)
            scores.append(100 * (1 - results[method]['inference_time_s'] / max_infer))

            # Accuracy (higher is better)
            scores.append(results[method]['accuracy_100m'])

            performance_scores[method] = scores

        # Simple bar chart instead of radar for clarity
        x_pos = np.arange(len(metrics_for_radar))
        width = 0.25

        for i, method in enumerate(methods):
            axes[1, 2].bar(x_pos + i*width, performance_scores[method],
                           width, label=method, color=colors[i], alpha=0.7)

        axes[1, 2].set_xlabel('Performance Metrics')
        axes[1, 2].set_ylabel('Normalized Score')
        axes[1, 2].set_title('Overall Performance Comparison')
        axes[1, 2].set_xticks(x_pos + width)
        axes[1, 2].set_xticklabels(['Accuracy', 'Train Speed', 'Infer Speed', 'Precision'])
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('method_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """
    Main function to run comprehensive comparative analysis.
    """
    print("Comprehensive Cellular Localization Method Comparison")
    print("=" * 60)

    # Load dataset
    df = pd.read_csv('data/logFile_urban_data.csv')
    print(f"Loaded dataset with {len(df)} records")

    # Initialize comparative analyzer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    analyzer = ComparativeAnalyzer(device=device)

    # Run comprehensive comparison
    results = analyzer.run_comparative_analysis(df)

    print("\nComparative analysis completed!")
    print("Results saved to 'method_comparison.png'")

    return results


if __name__ == "__main__":
    results = main()
