import os
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from model import CompressionAutoencoder
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

def create_sample_image(filename="data/sample_input.png"):
    """Generates a simple test image if one doesn't exist."""
    os.makedirs("data", exist_ok=True)
    img = Image.new('RGB', (64, 64), color=(73, 109, 137))
    d = ImageDraw.Draw(img)
    d.text((10,10), "AI", fill=(255,255,0))
    img.save(filename)
    return filename

def load_quantized_model(model_path="weights/autoencoder_8bit.pth"):
    """Rebuilds the FX Graph and loads the 8-bit weights."""
    torch.backends.quantized.engine = 'qnnpack'
    model_32bit = CompressionAutoencoder()
    model_32bit.eval()
    
    qconfig = get_default_qconfig_mapping("qnnpack")
    dummy_input = (torch.randn(1, 3, 64, 64),)
    prepared = prepare_fx(model_32bit, qconfig, dummy_input)
    quantized_model = convert_fx(prepared)
    
    quantized_model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    quantized_model.eval()
    return quantized_model

def run_pipeline():
    print("🚀 Initiating Edge-Optimized Neural Compression Pipeline...")
    
    # 1. Load the Image
    img_path = create_sample_image()
    img = Image.open(img_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    input_tensor = transform(img).unsqueeze(0) # Add batch dimension
    
    # 2. Load the 8-bit Edge Model
    print("🧠 Loading 8-bit INT8 Autoencoder...")
    model = load_quantized_model()
    
    # 3. COMPRESS: Run the encoder to get the latent bottleneck
    print("🗜️ Compressing image to latent binary representation...")
    with torch.no_grad():
        # The FX traced model returns the exact same tuple as the original forward() pass
        reconstructed_tensor, latent_tensor = model(input_tensor)
        
    # Save the latent tensor as a raw binary file (The "Compressed" file)
    compressed_path = "data/compressed_image.bin"
    # Convert from INT8 tensor back to standard bytes for storage
    with open(compressed_path, "wb") as f:
        f.write(latent_tensor.numpy().tobytes())
        
    # 4. DECOMPRESS: Save the reconstructed output
    print("🖼️ Reconstructing image from latent space...")
    recon_img_path = "data/reconstructed_output.png"
    # Squeeze out batch dim and convert back to PIL Image
    recon_transform = transforms.ToPILImage()
    recon_image = recon_transform(reconstructed_tensor.squeeze(0))
    recon_image.save(recon_img_path)

    # 5. Output Final System Metrics
    original_size = os.path.getsize(img_path)
    compressed_size = os.path.getsize(compressed_path)
    
    print("\n================ PIPELINE METRICS ================")
    print(f"Original Image Size (PNG): {original_size} bytes")
    print(f"Neural Compressed Size (BIN): {compressed_size} bytes")
    print("==================================================")
    print(f"✅ Pipeline complete. Check the /data folder to see the input and reconstructed images.")

if __name__ == "__main__":
    run_pipeline()