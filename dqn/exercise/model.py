import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        print("HI")
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"

        # self.conv1 = nn.Conv2d(4, 32, (8,8), 4)
        # self.conv2 = nn.Conv2d(32, 64, (4,4), 2)
        # self.conv3 = nn.Conv2d(64, 64, (3,3), 1)
        # self.fc1 = nn.Linear(2304, 512)
        # self.fc2 = nn.Linear(512, action_size)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        # print(state)
        # x = F.relu(self.conv1(state))
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = x.view(-1, 6*6*64)
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
