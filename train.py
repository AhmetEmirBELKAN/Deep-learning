import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 195, 1280)
        self.fc2 = nn.Linear(1280, 640)
        self.fc3 = nn.Linear(640, 1)

    def forward(self, x):
        print(f"x.shape : {x.shape}")
        x = self.pool1(F.relu(self.conv1(x.unsqueeze(1))))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * x.size(2))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class ModelTrainer:
    def __init__(self, model, X_train, y_train, X_val, y_val, momentum, batch_size, learning_rate):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        
        self.y_val = y_val
        self.momentum = momentum
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        self.train_losses = []
        self.val_losses = []

    def train(self, epochs=40):
        self.model.train()
        for epoch in range(epochs):
            for i in range(0, len(self.X_train), self.batch_size):
                batch_X = self.X_train[i:i+self.batch_size]
                batch_y = self.y_train[i:i+self.batch_size]

                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y.view(-1, 1))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.train_losses.append(loss.item())

            val_outputs = self.model(self.X_val)
            val_loss = self.criterion(val_outputs, self.y_val.view(-1, 1))
            self.val_losses.append(val_loss.item())

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}')
def train_and_evaluate(momentum_values, batch_sizes, learning_rates):
    df = pd.read_csv('dataset.csv')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X = df.iloc[:, 1:-1].values
    y = df['prime'].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_train_tensor = torch.Tensor(X_train)
    y_train_tensor = torch.Tensor(y_train)
    X_val_tensor = torch.Tensor(X_val)
    y_val_tensor = torch.Tensor(y_val)

    for momentum in momentum_values:
        for batch_size in batch_sizes:
            for learning_rate in learning_rates:
                model = CustomModel().to(device)
                trainer = ModelTrainer(model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, momentum, batch_size, learning_rate)
                trainer.train()
                trainer.plot_losses()

                with torch.no_grad():
                    train_predicted_labels = (model(X_train_tensor) >= 0.5).float()
                    val_predicted_labels = (model(X_val_tensor) >= 0.5).float()

                    train_accuracy = torch.sum(train_predicted_labels.view(-1) == y_train_tensor).item() / len(y_train)
                    val_accuracy = torch.sum(val_predicted_labels.view(-1) == y_val_tensor).item() / len(y_val)

                    print(f'Momentum: {momentum}, Batch Size: {batch_size}, Learning Rate: {learning_rate}')
                    print(f'Training Accuracy: {train_accuracy * 100:.2f}%, Validation Accuracy: {val_accuracy * 100:.2f}%')


momentum_values = [0.01, 0.5, 0.99]
batch_sizes = [256]
learning_rates = [0.1]
train_and_evaluate(momentum_values, batch_sizes, learning_rates)