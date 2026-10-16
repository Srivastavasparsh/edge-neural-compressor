import os
import torch
from model import CompressionAutoencoder
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

def run_quantization():
    # TELL PYTORCH WHICH MATH ENGINE TO USE FOR 8-BIT OPERATIONS
    torch.backends.quantized.engine = 'qnnpack'
    
    print("🧠 Loading 32-bit Floating Point Model...")
    model_32bit = CompressionAutoencoder()
    model_32bit.load_state_dict(torch.load("weights/autoencoder_32bit.pth", map_location="cpu", weights_only=True))
    model_32bit.eval()

    print("⚙️ Initiating FX Graph Post-Training Quantization (PTQ)...")
    qconfig_mapping = get_default_qconfig_mapping("qnnpack")
    
    # 1. PREPARE
    example_inputs = (torch.randn(1, 3, 64, 64),)
    prepared_model = prepare_fx(model_32bit, qconfig_mapping, example_inputs)
    
    # 2. CALIBRATE
    print("📊 Calibrating activation ranges...")
    for _ in range(10): 
        prepared_model(torch.randn(1, 3, 64, 64))
        
    # 3. CONVERT
    print("🗜️ Converting to INT8...")
    quantized_8bit_model = convert_fx(prepared_model)
    
    # Save the optimized Edge model
    save_path = "weights/autoencoder_8bit.pth"
    torch.save(quantized_8bit_model.state_dict(), save_path)
    
    # Calculate Hard Metrics for the Resume
    size_32bit = os.path.getsize("weights/autoencoder_32bit.pth") / 1024
    size_8bit = os.path.getsize(save_path) / 1024
    reduction = (1 - (size_8bit / size_32bit)) * 100
    
    print("\n================ QUANTIZATION METRICS ================")
    print(f"Original 32-bit Model Size: {size_32bit:.2f} KB")
    print(f"Quantized 8-bit Model Size: {size_8bit:.2f} KB")
    print(f"Memory Footprint Reduction: {reduction:.2f}%")
    print("======================================================")

if __name__ == "__main__":
    run_quantization()