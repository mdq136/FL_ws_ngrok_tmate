import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

# Define your neural network architecture
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)

# Define transform for the evaluation dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST evaluation dataset
eval_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=32, shuffle=False)

# Initialize the model
model = SimpleNet().to(torch.device('cpu'))

# Load the saved model weights
model.load_state_dict(torch.load('global_model.pth'))

# Evaluate the model
def evaluate_model(model, dataloader):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Accuracy on evaluation set: {accuracy:.2f}")
    return accuracy

# Evaluate the loaded model on the evaluation dataset
evaluate_model(model, eval_loader)
