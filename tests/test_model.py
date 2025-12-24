import torch
import torch.nn as nn

from ml_monitoring_service.model import (
    HybridAutoencoderTransformerModel,
    PositionalEncoding,
)


class TestPositionalEncoding:
    """Tests for PositionalEncoding module"""

    def test_positional_encoding_initialization(self):
        """Test that positional encoding initializes correctly"""
        d_model = 64
        max_len = 100
        pe = PositionalEncoding(d_model, max_len)

        assert hasattr(pe, "pe")
        assert pe.pe.shape == (1, max_len, d_model)

    def test_positional_encoding_forward(self):
        """Test forward pass of positional encoding"""
        d_model = 64
        batch_size = 2
        seq_len = 10

        pe = PositionalEncoding(d_model)
        x = torch.randn(batch_size, seq_len, d_model)
        output = pe(x)

        assert output.shape == x.shape
        assert not torch.equal(output, x)  # Should be different after adding PE

    def test_positional_encoding_deterministic(self):
        """Test that positional encoding is deterministic"""
        d_model = 64
        pe = PositionalEncoding(d_model)

        x1 = torch.randn(2, 10, d_model)
        x2 = x1.clone()

        output1 = pe(x1)
        output2 = pe(x2)

        assert torch.equal(output1, output2)


class TestHybridAutoencoderTransformerModel:
    """Tests for HybridAutoencoderTransformerModel"""

    def test_model_initialization(self):
        """Test that model initializes with correct parameters"""
        num_services = 3
        num_features = 5

        model = HybridAutoencoderTransformerModel(num_services, num_features)

        assert model.num_services == num_services
        assert model.hidden_dim == 64
        assert hasattr(model, "service_embeddings")
        assert hasattr(model, "feature_encoder")
        assert hasattr(model, "cross_service_attention")
        assert hasattr(model, "transformer_encoder")
        assert hasattr(model, "decoder")

    def test_service_embeddings_shape(self):
        """Test that service embeddings have correct shape"""
        num_services = 5
        num_features = 4

        model = HybridAutoencoderTransformerModel(num_services, num_features)

        assert model.service_embeddings.shape == (num_services, model.hidden_dim)

    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape"""
        num_services = 3
        num_features = 5
        batch_size = 2
        window_size = 30

        model = HybridAutoencoderTransformerModel(num_services, num_features)

        # Create input tensors
        x = torch.randn(batch_size, window_size, num_services, num_features)
        time_features = torch.randn(batch_size, window_size, 4)

        # Forward pass
        output = model(x, time_features)

        # Check output shape
        assert output.shape == (batch_size, window_size, num_services, num_features)

    def test_forward_pass_no_nan(self):
        """Test that forward pass doesn't produce NaN values"""
        num_services = 2
        num_features = 3
        batch_size = 1
        window_size = 10

        model = HybridAutoencoderTransformerModel(num_services, num_features)
        model.eval()

        x = torch.randn(batch_size, window_size, num_services, num_features)
        time_features = torch.randn(batch_size, window_size, 4)

        with torch.no_grad():
            output = model(x, time_features)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_model_device_compatibility(self):
        """Test that model can be moved to different devices"""
        num_services = 2
        num_features = 3

        model = HybridAutoencoderTransformerModel(num_services, num_features)

        # Test CPU
        model_cpu = model.to("cpu")
        assert next(model_cpu.parameters()).device.type == "cpu"

        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.to("cuda")
            assert next(model_cuda.parameters()).device.type == "cuda"

    def test_model_parameters_require_grad(self):
        """Test that model parameters require gradients"""
        num_services = 2
        num_features = 3

        model = HybridAutoencoderTransformerModel(num_services, num_features)

        for param in model.parameters():
            assert param.requires_grad

    def test_reconstruction_autoencoder(self):
        """Test that model can reconstruct input (autoencoder property)"""
        num_services = 2
        num_features = 3
        batch_size = 1
        window_size = 10

        model = HybridAutoencoderTransformerModel(num_services, num_features)
        model.train()

        x = torch.randn(batch_size, window_size, num_services, num_features)
        time_features = torch.randn(batch_size, window_size, 4)

        output = model(x, time_features)

        # Output should have same shape as input
        assert output.shape == x.shape

    def test_model_with_different_batch_sizes(self):
        """Test that model works with different batch sizes"""
        num_services = 2
        num_features = 3
        window_size = 10

        model = HybridAutoencoderTransformerModel(num_services, num_features)
        model.eval()

        with torch.no_grad():
            for batch_size in [1, 4, 8]:
                x = torch.randn(batch_size, window_size, num_services, num_features)
                time_features = torch.randn(batch_size, window_size, 4)

                output = model(x, time_features)
                assert output.shape == (
                    batch_size,
                    window_size,
                    num_services,
                    num_features,
                )

    def test_cross_service_attention_applied(self):
        """Test that cross-service attention modifies the output"""
        num_services = 3
        num_features = 4
        batch_size = 1
        window_size = 5

        model = HybridAutoencoderTransformerModel(num_services, num_features)

        # Create input with distinct patterns per service
        x = torch.zeros(batch_size, window_size, num_services, num_features)
        x[:, :, 0, :] = 1.0  # Service 0 has all 1s
        x[:, :, 1, :] = 2.0  # Service 1 has all 2s
        x[:, :, 2, :] = 3.0  # Service 2 has all 3s

        time_features = torch.randn(batch_size, window_size, 4)

        output = model(x, time_features)

        # Output should be different from input due to attention
        assert not torch.allclose(output, x, atol=0.1)

    def test_gradient_flow(self):
        """Test that gradients flow through the model"""
        num_services = 2
        num_features = 3
        batch_size = 2
        window_size = 10

        model = HybridAutoencoderTransformerModel(num_services, num_features)
        criterion = nn.MSELoss()

        x = torch.randn(batch_size, window_size, num_services, num_features)
        time_features = torch.randn(batch_size, window_size, 4)

        output = model(x, time_features)
        loss = criterion(output, x)
        loss.backward()

        # Check that gradients exist for parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


class TestModelIntegration:
    """Integration tests for the model"""

    def test_training_step(self):
        """Test a single training step"""
        num_services = 2
        num_features = 3
        batch_size = 4
        window_size = 10

        model = HybridAutoencoderTransformerModel(num_services, num_features)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        model.train()

        x = torch.randn(batch_size, window_size, num_services, num_features)
        time_features = torch.randn(batch_size, window_size, 4)

        optimizer.zero_grad()
        output = model(x, time_features)
        loss = criterion(output, x)
        loss.backward()
        optimizer.step()

        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_inference_mode(self):
        """Test model in inference mode"""
        num_services = 2
        num_features = 3
        batch_size = 1
        window_size = 30

        model = HybridAutoencoderTransformerModel(num_services, num_features)
        model.eval()

        x = torch.randn(batch_size, window_size, num_services, num_features)
        time_features = torch.randn(batch_size, window_size, 4)

        with torch.no_grad():
            output1 = model(x, time_features)
            output2 = model(x, time_features)

        # In eval mode with same input, output should be identical
        assert torch.equal(output1, output2)
