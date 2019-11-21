import torch.nn as nn
import torch.nn.functional as F
import torch


class CriticModel(nn.Module):
    def __init__(self, size_in, hidden_units, size_out):
        super(CriticModel, self).__init__()
        self.fc1 = nn.Linear(size_in, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, size_out)
        return

    def forward(self, x0):
        x = F.relu(self.fc1(x0))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ActorModel(nn.Module):
    def __init__(self, size_in, hidden_units, size_out):
        super(ActorModel, self).__init__()
        self.fc1 = nn.Linear(size_in, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, size_out)
        return

    def forward(self, x0):
        x = F.relu(self.fc1(x0))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x
