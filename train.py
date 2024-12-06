import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model.network import SimpleCNN
import os
from datetime import datetime

def get_transforms(is_train=True):
    """
    Define transforms for training and testing
    """
    if is_train:
        return transforms.Compose([
            transforms.RandomRotation(10),  # Rotate +/- 10 degrees 
            transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Random shift
            transforms.RandomAffine(0, scale=(0.9, 1.1)),  # Random scaling
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.RandomErasing(p=0.2)  # Random erasing for robustness
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

def train(epochs=1, batch_size=64, learning_rate=0.01):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load MNIST Dataset with augmentation
    train_transform = get_transforms(is_train=True)
    test_transform = get_transforms(is_train=False)
    
    train_dataset = datasets.MNIST('data', train=True, download=True, 
                                 transform=train_transform)
    test_dataset = datasets.MNIST('data', train=False, download=True, 
                                transform=test_transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                             batch_size=batch_size, 
                                             shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=False)
    
    # Initialize model
    model = SimpleCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 
                                                   patience=2, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    
    best_accuracy = 0
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        # Validation phase
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(test_loader)
        accuracy = 100. * correct / len(test_loader.dataset)
        
        print(f'Epoch {epoch}: Test loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        # Update learning rate
        scheduler.step(test_loss)
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # Save model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            device_type = "cuda" if torch.cuda.is_available() else "cpu"
            
            if not os.path.exists('models'):
                os.makedirs('models')
            
            model_path = f'models/model_{timestamp}_{device_type}_acc_{accuracy:.2f}.pkl'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'accuracy': accuracy,
                'timestamp': timestamp,
                'device': device_type
            }, model_path)
    
    return model_path

if __name__ == "__main__":
    train() 