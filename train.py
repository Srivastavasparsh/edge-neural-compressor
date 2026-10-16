import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import CompressionAutoencoder
import os

# 1. Hardware Acceleration (Apple Silicon MPS, CUDA, or CPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 Using Apple Metal (MPS) for hardware acceleration!")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("⚠️ No GPU detected, using CPU.")

def train_model():
    # 2. Download and Load Dataset (Resizing to 64x64 to match our architecture)
    print("\nPreparing CIFAR-10 Dataset...")
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    # 3. Initialize Model, Loss (Mean Squared Error), and Optimizer
    model = CompressionAutoencoder().to(device)
    criterion = nn.MSELoss() # We want the output pixels to exactly match the input pixels
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 3 # Just 3 epochs for this portfolio prototype
    print(f"\n🧠 Starting Training Loop for {epochs} epochs...")

    # 4. Training Loop
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Autoencoders are unique: the input image is ALSO the target label
            inputs, _ = data 
            inputs = inputs.to(device)

            optimizer.zero_grad()
            outputs, _ = model(inputs)
            
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:    # Print every 200 mini-batches
                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{i + 1:4d}], MSE Loss: {running_loss / 200:.4f}")
                running_loss = 0.0

    # 5. Save the trained weights for Edge Inference
    os.makedirs("weights", exist_ok=True)
    model_path = "weights/autoencoder_32bit.pth"
    torch.save(model.state_dict(), model_path)
    
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"\n✅ Training Complete!")
    print(f"💾 Saved standard 32-bit model to {model_path} ({file_size_mb:.2f} MB)")

if __name__ == "__main__":
    train_model()