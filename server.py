from flask import Flask, request
from flask_socketio import SocketIO, emit
import torch
import torch.nn as nn
import torch.optim as optim
import os

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

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

# Instantiate the global model
global_model = SimpleNet()

# Loss function and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(global_model.parameters(), lr=0.001)

# Example users with passwords
users = {
    "client1": "1",
    "client2": "2",
    "client3": "3",
    "client4": "4"
}

clients = {}
num_clients = 2  # Number of expected clients

MODEL_DIR = 'saved_models'
os.makedirs(MODEL_DIR, exist_ok=True)

def save_model_state(model_state_dict, filename):
    filepath = os.path.join(MODEL_DIR, filename)
    torch.save(model_state_dict, filepath)
    print(f"Model state saved to {filename}")

def broadcast_model():
    print("Sending global model")
    global global_model
    model_state_dict = global_model.state_dict()
    
    # Convert model state dictionary tensors to lists
    model_state_list = {key: value.tolist() for key, value in model_state_dict.items()}
    for client_sid in clients:
        socketio.emit('global_model', model_state_list, to=client_sid)
    # Save model state after broadcasting
    save_model_state(model_state_dict, 'global_model.pth')

def convert_nested_list_to_tensor(data):
    if isinstance(data, list):
        return torch.tensor(data)
    elif isinstance(data, dict):
        return {key: convert_nested_list_to_tensor(value) for key, value in data.items()}
    return data

def average_model_weights():
    global global_model

    # Initialize a dictionary to store accumulated weights
    accumulated_weights = {}
    num_clients_contributed = len(clients)

    # Sum up the model weights from all clients
    for client_sid in clients:
        client_model_state_dict = clients[client_sid]['model_state_dict']
        for key in client_model_state_dict:
            if key in accumulated_weights:
                accumulated_weights[key] += client_model_state_dict[key]
            else:
                accumulated_weights[key] = client_model_state_dict[key]

    # Average the accumulated weights
    averaged_weights = {key: value / num_clients_contributed for key, value in accumulated_weights.items()}

    # Update the global model with averaged weights
    global_model.load_state_dict(averaged_weights)

    # Broadcast the averaged global model to all clients
    broadcast_model()

@app.route('/')
def index():
    return "Federated Learning Server"

@socketio.on('connect')
def handle_connect():
    print("Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    username = clients.get(sid, {}).get('username', 'Unknown')
    print(f"Client {username} disconnected")
    if sid in clients:
        del clients[sid]

@socketio.on('register')
def handle_register(data):
    sid = request.sid
    username = data.get('username')
    password = data.get('password')

    if username in users and users[username] == password:
        clients[sid] = {'username': username}
        print(f"Client {username} registered")
        emit('registration_success', {'message': 'Registration successful'})
        print(len(clients) == num_clients)
        if len(clients) == num_clients:
            broadcast_model()
    else:
        print(f"Client {username} failed to register")
        emit('registration_failure', {'message': 'Invalid username or password'})

@socketio.on('client_update')
def handle_client_update(data):
    global global_model, optimizer

    sid = request.sid
    username = clients.get(sid, {}).get('username', 'Unknown')
    print(f"Received update from {username}")

    # Convert lists back to tensors
    model_state_dict = {key: torch.tensor(value) for key, value in data['model_state_dict'].items()}
    optimizer_state_dict = {key: convert_nested_list_to_tensor(value) for key, value in data['optimizer_state_dict'].items()}
    client_loss = data['loss']

    # Store the model state dict received from the client
    clients[sid]['model_state_dict'] = model_state_dict

    # Update global model
    global_model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)

    # Averaging step
    if all('model_state_dict' in clients[client_sid] for client_sid in clients):
        average_model_weights()

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5001)
