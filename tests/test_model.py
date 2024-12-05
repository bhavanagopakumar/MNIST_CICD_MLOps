import os
import pytest
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from model.network import SimpleCNN
from train import train

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_accuracy(model, device='cpu'):
    model.eval()
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def test_model_creation():
    model = SimpleCNN()
    assert isinstance(model, SimpleCNN)
    
def test_model_forward():
    model = SimpleCNN()
    x = torch.randn(1, 1, 28, 28)  # MNIST image size
    output = model(x)
    assert output.shape == (1, 10)  # 10 classes for MNIST

def test_model_parameters():
    model = SimpleCNN()
    num_params = count_parameters(model)

    print("\n")
    print("=" * 50)
    print(f"Model has {num_params} parameters")
    print("=" * 50)
    print("\n")
    assert num_params < 25000, f"Model has {num_params} parameters, which exceeds the limit of 25000"


def test_model_accuracy():
    # Train the model
    model_path = train()
    
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Calculate accuracy
    accuracy = calculate_accuracy(model, device)
    
    print("\n")
    print("=" * 50)
    print(f"Model accuracy: {accuracy:.2f}%")
    print(f"Using model: {checkpoint['model_state_dict']}")
    print("=" * 50)
    print("\n")
    assert accuracy > 95.0, f"Model accuracy {accuracy:.2f}% is below the required 95%"

def test_model_training():
    # Run training and get model path
    model_path = train()
    
    # Check if model file was created
    assert os.path.exists(model_path)
    
    # Load and verify the model
    checkpoint = torch.load(model_path)
    assert 'model_state_dict' in checkpoint
    assert 'timestamp' in checkpoint
    assert 'device' in checkpoint

def get_latest_model():
    if not os.path.exists('models'):
        return None
        
    model_files = [f for f in os.listdir('models') if f.endswith('.pkl')]
    if not model_files:
        return None
        
    return max((os.path.join('models', f) for f in model_files), 
              key=os.path.getctime)

def test_latest_model():
    latest_model = get_latest_model()
    if latest_model is None:
        pytest.skip("No model files found - run training first")
    
    # Load and verify the model
    checkpoint = torch.load(latest_model)
    assert 'model_state_dict' in checkpoint
