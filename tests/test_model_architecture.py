"""Tests for model functionality."""

import torch

from ml_monitoring_service.model import (
    HybridAutoencoderTransformerModel,
    PositionalEncoding,
)


def test_positional_encoding_shape():
    """Test that positional encoding produces correct output shape."""
    d_model = 64
    max_len = 100
    batch_size = 8
    seq_len = 20

    pe = PositionalEncoding(d_model, max_len)

    # Create input tensor
    x = torch.randn(batch_size, seq_len, d_model)

    # Apply positional encoding
    output = pe(x)

    # Check output shape matches input shape
    assert output.shape == x.shape


def test_positional_encoding_buffer():
    """Test that positional encoding is registered as buffer."""
    d_model = 64
    pe = PositionalEncoding(d_model)

    # Check that pe buffer exists
    assert hasattr(pe, "pe")
    assert isinstance(pe.pe, torch.Tensor)


def test_hybrid_model_forward_shape():
    """Test that model forward pass produces correct output shape."""
    num_services = 5
    num_features = 10
    batch_size = 8
    seq_len = 20

    model = HybridAutoencoderTransformerModel(num_services, num_features)

    # Create dummy input
    x = torch.randn(batch_size, seq_len, num_services, num_features)
    time_features = torch.randn(batch_size, seq_len, 4)

    # Forward pass
    output = model(x, time_features)

    # Check output shape matches input shape
    assert output.shape == x.shape


def test_hybrid_model_parameters():
    """Test that model has trainable parameters."""
    num_services = 5
    num_features = 10

    model = HybridAutoencoderTransformerModel(num_services, num_features)

    # Check that model has parameters
    params = list(model.parameters())
    assert len(params) > 0

    # Check that service embeddings are trainable
    assert model.service_embeddings.requires_grad


def test_hybrid_model_service_embeddings():
    """Test that service embeddings have correct shape."""
    num_services = 5
    num_features = 10

    model = HybridAutoencoderTransformerModel(num_services, num_features)

    # Check service embeddings shape
    assert model.service_embeddings.shape == (num_services, model.hidden_dim)


def test_hybrid_model_deterministic_with_same_input():
    """Test that model produces same output for same input."""
    num_services = 3
    num_features = 8
    batch_size = 4
    seq_len = 10

    model = HybridAutoencoderTransformerModel(num_services, num_features)
    model.eval()  # Set to evaluation mode

    # Create dummy input
    x = torch.randn(batch_size, seq_len, num_services, num_features)
    time_features = torch.randn(batch_size, seq_len, 4)

    # Forward pass twice
    with torch.no_grad():
        output1 = model(x, time_features)
        output2 = model(x, time_features)

    # Outputs should be identical
    assert torch.allclose(output1, output2)


def test_hybrid_model_different_batch_sizes():
    """Test that model works with different batch sizes."""
    num_services = 5
    num_features = 10
    seq_len = 20

    model = HybridAutoencoderTransformerModel(num_services, num_features)

    for batch_size in [1, 4, 8, 16]:
        x = torch.randn(batch_size, seq_len, num_services, num_features)
        time_features = torch.randn(batch_size, seq_len, 4)

        output = model(x, time_features)

        assert output.shape == (batch_size, seq_len, num_services, num_features)


def test_hybrid_model_gradient_flow():
    """Test that gradients flow through the model."""
    num_services = 3
    num_features = 5
    batch_size = 4
    seq_len = 10

    model = HybridAutoencoderTransformerModel(num_services, num_features)
    model.train()

    # Create dummy input and target
    x = torch.randn(batch_size, seq_len, num_services, num_features)
    time_features = torch.randn(batch_size, seq_len, 4)
    target = torch.randn(batch_size, seq_len, num_services, num_features)

    # Forward pass
    output = model(x, time_features)

    # Compute loss
    loss = torch.nn.functional.mse_loss(output, target)

    # Backward pass
    loss.backward()

    # Check that gradients exist
    has_gradients = False
    for param in model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_gradients = True
            break

    assert has_gradients, "No gradients found in model parameters"
