"""
Deep learning supervised learning models including LSTM and Transformer.
"""
from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import from base module
try:
    from .base import PyTorchSLModel, set_all_seeds
except ImportError:
    # Fallback for development environment
    from src.sl.models.base import PyTorchSLModel, set_all_seeds


class CNNLSTMModel(PyTorchSLModel):
    """CNN-LSTM model for financial time-series forecasting."""

    def __init__(self, config: dict):
        """
        Initialize the CNN-LSTM model.

        Args:
            config (dict): Configuration dictionary for the model
        """
        super().__init__(config)

        # Model architecture parameters
        self.input_size = config.get('input_size', 10)
        self.cnn_channels = config.get('cnn_channels', [32, 64])
        self.cnn_kernel_sizes = config.get('cnn_kernel_sizes', [3, 3])
        self.lstm_hidden_size = config.get('lstm_hidden_size', 64)
        self.lstm_num_layers = config.get('lstm_num_layers', 2)
        self.output_size = config.get('output_size', 1)
        self.dropout = config.get('dropout', 0.2)
        self.sequence_length = config.get('sequence_length', 10)

        # Create the CNN-LSTM model
        self._create_model()

        # Set loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=config.get('weight_decay', 1e-5)
        )

    def _create_model(self):
        """Create the CNN-LSTM model architecture."""
        self.model = CNNLSTMNetwork(
            input_size=self.input_size,
            cnn_channels=self.cnn_channels,
            cnn_kernel_sizes=self.cnn_kernel_sizes,
            lstm_hidden_size=self.lstm_hidden_size,
            lstm_num_layers=self.lstm_num_layers,
            output_size=self.output_size,
            dropout=self.dropout,
            sequence_length=self.sequence_length
        ).to(self.device)

    def _prepare_sequences(self, X: np.ndarray) -> torch.Tensor:
        """
        Prepare input data as sequences for CNN-LSTM.

        Args:
            X: Input feature matrix

        Returns:
            torch.Tensor: Sequenced input data
        """
        # Convert to tensor
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Ensure we have enough data for sequences
        if len(X) < self.sequence_length:
            raise ValueError(f"Input data length ({len(X)}) is less than sequence length ({self.sequence_length})")

        # Create sequences
        sequences = []
        for i in range(len(X) - self.sequence_length + 1):
            sequences.append(X[i:i + self.sequence_length])

        return torch.FloatTensor(np.array(sequences)).to(self.device)

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        """
        Fit the CNN-LSTM model on training data.

        Args:
            X: Feature matrix
            y: Target vector
        """
        # Set seeds for deterministic processing
        set_all_seeds(self.random_state)

        # Convert to numpy if pandas
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Prepare sequences
        X_seq = self._prepare_sequences(X)
        y_seq = torch.FloatTensor(y[self.sequence_length-1:]).to(self.device)

        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X_seq, y_seq)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += loss.item()

        self.is_fitted = True

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions with the CNN-LSTM model.

        Args:
            X: Feature matrix

        Returns:
            np.ndarray: Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Convert to numpy if pandas
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Prepare sequences
        X_seq = self._prepare_sequences(X)

        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_seq)
            return predictions.cpu().numpy().squeeze()


class CNNLSTMNetwork(nn.Module):
    """CNN-LSTM network architecture."""

    def __init__(self, input_size: int, cnn_channels: list, cnn_kernel_sizes: list,
                 lstm_hidden_size: int, lstm_num_layers: int, output_size: int,
                 dropout: float, sequence_length: int):
        """
        Initialize the CNN-LSTM network.

        Args:
            input_size: Number of input features
            cnn_channels: List of CNN channel sizes
            cnn_kernel_sizes: List of CNN kernel sizes
            lstm_hidden_size: Number of LSTM hidden units
            lstm_num_layers: Number of LSTM layers
            output_size: Number of output units
            dropout: Dropout rate
            sequence_length: Length of input sequences
        """
        super().__init__()

        self.input_size = input_size
        self.sequence_length = sequence_length
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers

        # CNN layers
        cnn_layers = []
        in_channels = input_size
        for i, (out_channels, kernel_size) in enumerate(zip(cnn_channels, cnn_kernel_sizes)):
            cnn_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.Dropout(dropout))
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1] if cnn_channels else input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Output layer
        self.fc = nn.Linear(lstm_hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            torch.Tensor: Output tensor
        """
        batch_size = x.size(0)

        # Reshape for CNN: (batch_size, input_size, sequence_length)
        x = x.transpose(1, 2)

        # CNN forward pass
        x = self.cnn(x)

        # Reshape back for LSTM: (batch_size, sequence_length, cnn_output_channels)
        x = x.transpose(1, 2)

        # Initialize hidden state
        h0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(x.device)

        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))

        # Take the last output
        out = self.dropout(out[:, -1, :])

        # Linear layer
        out = self.fc(out)

        return out


class TransformerModel(PyTorchSLModel):
    """Transformer model for financial time-series forecasting."""

    def __init__(self, config: dict):
        """
        Initialize the Transformer model.

        Args:
            config (dict): Configuration dictionary for the model
        """
        super().__init__(config)

        # Model architecture parameters
        self.input_size = config.get('input_size', 10)
        self.d_model = config.get('d_model', 64)
        self.nhead = config.get('nhead', 8)
        self.num_layers = config.get('num_layers', 2)
        self.dim_feedforward = config.get('dim_feedforward', 256)
        self.output_size = config.get('output_size', 1)
        self.dropout = config.get('dropout', 0.1)
        self.sequence_length = config.get('sequence_length', 10)

        # Create the Transformer model
        self._create_model()

        # Set loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=config.get('weight_decay', 1e-5)
        )

    def _create_model(self):
        """Create the Transformer model architecture."""
        self.model = TransformerNetwork(
            input_size=self.input_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            output_size=self.output_size,
            dropout=self.dropout,
            sequence_length=self.sequence_length
        ).to(self.device)

    def _prepare_sequences(self, X: np.ndarray) -> torch.Tensor:
        """
        Prepare input data as sequences for Transformer.

        Args:
            X: Input feature matrix

        Returns:
            torch.Tensor: Sequenced input data
        """
        # Convert to tensor
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Ensure we have enough data for sequences
        if len(X) < self.sequence_length:
            raise ValueError(f"Input data length ({len(X)}) is less than sequence length ({self.sequence_length})")

        # Create sequences
        sequences = []
        for i in range(len(X) - self.sequence_length + 1):
            sequences.append(X[i:i + self.sequence_length])

        return torch.FloatTensor(np.array(sequences)).to(self.device)

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        """
        Fit the Transformer model on training data.

        Args:
            X: Feature matrix
            y: Target vector
        """
        # Set seeds for deterministic processing
        set_all_seeds(self.random_state)

        # Convert to numpy if pandas
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Prepare sequences
        X_seq = self._prepare_sequences(X)
        y_seq = torch.FloatTensor(y[self.sequence_length-1:]).to(self.device)

        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X_seq, y_seq)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += loss.item()

        self.is_fitted = True

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions with the Transformer model.

        Args:
            X: Feature matrix

        Returns:
            np.ndarray: Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Convert to numpy if pandas
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Prepare sequences
        X_seq = self._prepare_sequences(X)

        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_seq)
            return predictions.cpu().numpy().squeeze()


class TransformerNetwork(nn.Module):
    """Transformer network architecture."""

    def __init__(self, input_size: int, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int, output_size: int, dropout: float, sequence_length: int):
        """
        Initialize the Transformer network.

        Args:
            input_size: Number of input features
            d_model: Dimension of the model
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of the feedforward network
            output_size: Number of output units
            dropout: Dropout rate
            sequence_length: Length of input sequences
        """
        super().__init__()

        self.d_model = d_model
        self.sequence_length = sequence_length

        # Input projection layer
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(d_model, sequence_length)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output layer
        self.fc = nn.Linear(d_model, output_size)

    def _create_positional_encoding(self, d_model: int, max_len: int) -> torch.Tensor:
        """
        Create positional encoding.

        Args:
            d_model: Dimension of the model
            max_len: Maximum sequence length

        Returns:
            torch.Tensor: Positional encoding tensor
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        # Project input to d_model dimensions
        x = self.input_projection(x)

        # Add positional encoding
        pos_encoding = self.pos_encoding[:, :x.size(1), :].to(x.device)
        x = x + pos_encoding

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Take the last output
        x = x[:, -1, :]

        # Linear layer
        x = self.fc(x)

        return x


class MLPModel(PyTorchSLModel):
    """Multi-Layer Perceptron model for financial forecasting."""

    def __init__(self, config: dict):
        """
        Initialize the MLP model.

        Args:
            config (dict): Configuration dictionary for the model
        """
        super().__init__(config)

        # Model architecture parameters
        self.input_size = config.get('input_size', 10)
        self.hidden_sizes = config.get('hidden_sizes', [64, 32])
        self.output_size = config.get('output_size', 1)
        self.dropout = config.get('dropout', 0.2)

        # Create the MLP model
        self._create_model()

        # Set loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=config.get('weight_decay', 1e-5)
        )

    def _create_model(self):
        """Create the MLP model architecture."""
        layers = []
        prev_size = self.input_size

        # Hidden layers
        for hidden_size in self.hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, self.output_size))

        self.model = nn.Sequential(*layers).to(self.device)

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        """
        Fit the MLP model on training data.

        Args:
            X: Feature matrix
            y: Target vector
        """
        # Set seeds for deterministic processing
        set_all_seeds(self.random_state)

        # Convert to numpy if pandas
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += loss.item()

        self.is_fitted = True

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions with the MLP model.

        Args:
            X: Feature matrix

        Returns:
            np.ndarray: Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Convert to numpy if pandas
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy().squeeze()
