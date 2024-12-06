import os
import pytest
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from model.network import SimpleCNN
from train import train, get_transforms
import numpy as np

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

def calculate_loss(model, device='cpu'):
    model.eval()
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            total_loss += loss.item()
            num_batches += 1
    
    average_loss = total_loss / num_batches
    return average_loss

def test_model_creation():
    model = SimpleCNN()
    assert isinstance(model, SimpleCNN)
    
def test_model_forward():
    model = SimpleCNN()
    x = torch.randn(1, 1, 28, 28)  # MNIST image size
    output = model(x)
    assert output.shape == (1, 16)  # 10 classes for MNIST

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
    print("=" * 51)
    print(f"Model accuracy: {accuracy:.2f}%")
    #print(f"Using model: {checkpoint['model_state_dict']}")
    print("=" * 51) # On Windows: venv\Scripts\activate)
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

def test_model_loss():
    # Train or load the model
    model_path = train()
    
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Calculate loss
    loss = calculate_loss(model, device)
    assert loss < 0.5, f"Model loss {loss:.4f} is above the threshold of 0.5"

def test_transform_consistency():
    """Test if transforms are consistent with expected parameters"""
    train_transform = get_transforms(is_train=True)
    test_transform = get_transforms(is_train=False)
    
    # Check if training transform has augmentation
    assert len(train_transform.transforms) > len(test_transform.transforms)
    
    # Verify test transform only has ToTensor and Normalize
    assert len(test_transform.transforms) == 2

def test_augmentation_variability():
    """Test if augmentation produces different results"""
    train_transform = get_transforms(is_train=True)
    dataset = datasets.MNIST('data', train=True, download=True)
    image, _ = dataset[0]
    
    # Generate multiple augmented versions
    augmented1 = train_transform(image)
    augmented2 = train_transform(image)
    
    # Check if augmentations are different
    assert not torch.allclose(augmented1, augmented2)

def test_augmentation_range():
    """Test if augmented values stay in expected range"""
    train_transform = get_transforms(is_train=True)
    dataset = datasets.MNIST('data', train=True, download=True)
    image, _ = dataset[0]
    
    augmented = train_transform(image)
    
    # Check value ranges after normalization
    assert augmented.min() >= -1.0
    assert augmented.max() <= 3.0

def test_augmentation_shape():
    """Test if augmentation preserves image shape"""
    train_transform = get_transforms(is_train=True)
    dataset = datasets.MNIST('data', train=True, download=True)
    image, _ = dataset[0]
    
    augmented = train_transform(image)
    
    # Check if shape is preserved (1x28x28 for MNIST)
    assert augmented.shape == (1, 28, 28)

def test_random_erasing():
    """Test if random erasing is applied"""
    train_transform = get_transforms(is_train=True)
    dataset = datasets.MNIST('data', train=True, download=True)
    image, _ = dataset[0]
    
    # Apply transform multiple times
    erased = False
    for _ in range(10):
        augmented = train_transform(image)
        if not torch.allclose(augmented, transforms.ToTensor()(image)):
            erased = True
            break
    
    assert erased, "Random erasing was never applied in 10 attempts"

def test_rotation_limits():
    """Test if rotation stays within specified limits"""
    train_transform = get_transforms(is_train=True)
    dataset = datasets.MNIST('data', train=True, download=True)
    image, _ = dataset[0]
    
    original = transforms.ToTensor()(image)
    augmented = train_transform(image)
    
    # Calculate maximum pixel shift (rough estimate for 10-degree rotation)
    max_shift = int(28 * np.sin(np.radians(10)))
    assert torch.any(torch.abs(augmented - original) > 0)

def test_model_augmentation_robustness():
    """Test if model performs well on both original and augmented data"""
    model_path = train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test on original and augmented versions
    test_transform = get_transforms(is_train=False)
    train_transform = get_transforms(is_train=True)
    
    dataset = datasets.MNIST('data', train=False, download=True)
    image, label = dataset[0]
    
    # Test prediction on original
    original = test_transform(image).unsqueeze(0).to(device)
    orig_pred = model(original).argmax(dim=1)
    
    # Test prediction on augmented
    augmented = train_transform(image).unsqueeze(0).to(device)
    aug_pred = model(augmented).argmax(dim=1)
    
    # Both should predict correctly
    assert orig_pred.item() == label
    assert aug_pred.item() == label
