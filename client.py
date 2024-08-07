import socketio
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import argparse
import time
import json
import zlib

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--server', type=str)
parser.add_argument('-u', '--username', type=str)
parser.add_argument('-p', '--password', type=str)
parser.add_argument('-e', '--epochs', type=int)

args = parser.parse_args()

server = args.server
username = args.username
password = args.password
epochs = args.epochs

# Define a simple neural network model
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

# Connect to the server
sio = socketio.Client()

# Global variables
local_model = SimpleNet()
optimizer = optim.Adam(local_model.parameters(), lr=0.001)
criterion = nn.NLLLoss()

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

@sio.event
def connect():
    print("Connected to the server")
    # Register with the server
    sio.emit('register', {'username': username, 'password': password})

@sio.event
def disconnect():
    print("Disconnected from the server")

@sio.on('registration_success')
def on_registration_success(data):
    print(data['message'])

@sio.on('registration_failure')
def on_registration_failure(data):
    print(data['message'])

@sio.on('global_model')
def on_global_model(data):
    print("Received global model")
    model_state_dict = {key: torch.tensor(value) for key, value in data.items()}
    print(type(model_state_dict))
    local_model.load_state_dict(model_state_dict)
    # Start local training
    train()

def convert_tensors_to_lists(state_dict):
    if isinstance(state_dict, dict):
        return {key: convert_tensors_to_lists(value) for key, value in state_dict.items()}
    elif isinstance(state_dict, torch.Tensor):
        return state_dict.tolist()
    return state_dict

def serialize_state_dict(state_dict):
    serialized_dict = {k: v.cpu().tolist() for k, v in state_dict.items()}
    json_str = json.dumps(serialized_dict)
    compressed_data = zlib.compress(json_str.encode())
    return compressed_data

def train():
    local_model.train()
    for epoch in tqdm(range(epochs)):  # Train for one epoch
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = local_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Training epoch loss: {avg_loss}")

        # After training, send model updates to the server
        # model_state_dict = {key: value.tolist() for key, value in local_model.state_dict().items()}
    # model_state_dict = {key: value.cpu().tolist() for key, value in local_model.state_dict().items()}
    # optimizer_state_dict = {key: list(value) for key, value in optimizer.state_dict().items()}

    try:
        # print("Hello")
        # sio.emit('hello',{'hello':"json.dumps(model_state_dict)"})
        sio.emit('client_update',{'hello':serialize_state_dict(local_model.state_dict())})
        # time.sleep(1)
        # sio.emit('hello',{'hello':"json.dumps(model_state_dict)"})
    except Exception as err:
        print(f"Error sending model state: {err}")
    # sio.emit('hello', {
    #     'model_state_dict': model_state_dict,
    #     'optimizer_state_dict': optimizer_state_dict,
    #     'loss': avg_loss
    # })
    print("Sending update")
    # os.wait()
    sio.disconnect()



if __name__ == '__main__':
    sio.connect(server)
    sio.wait()
