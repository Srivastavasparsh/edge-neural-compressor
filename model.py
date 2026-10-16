import torch
import torch.nn as nn

class CompressionAutoencoder(nn.Module):
    def __init__(self):
        super(CompressionAutoencoder, self).__init__()
        
        # ENCODER: Compresses the image
        # Input: (Channels=3, H, W)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # Halves spatial dimensions
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # Halves again
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # Halves again
            nn.ReLU(True)
        )
        # The output of the encoder is the "Bottleneck" or "Latent Space"
        
        # DECODER: Reconstructs the image from the Bottleneck
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() # Squashes output pixel values between 0 and 1
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

if __name__ == "__main__":
    # Simulate a small RGB image: Batch Size 1, 3 Channels (RGB), 64x64 Pixels
    dummy_image = torch.randn(1, 3, 64, 64)
    print(f"Original Image Tensor Shape: {dummy_image.shape}")
    print(f"Original Data Volume (Elements): {dummy_image.numel()}")
    
    # Initialize the model
    model = CompressionAutoencoder()
    
    # Pass the image through the model
    reconstructed_img, compressed_latent = model(dummy_image)
    
    print("\n--- Compression Metrics ---")
    print(f"Bottleneck Tensor Shape: {compressed_latent.shape}")
    print(f"Compressed Data Volume (Elements): {compressed_latent.numel()}")
    
    compression_ratio = dummy_image.numel() / compressed_latent.numel()
    print(f"\nArchitectural Compression Ratio: {compression_ratio:.2f}X")