import torch
from torch.nn import functional as F
import torch.nn as nn
import torch.optim as optim
import os

class LinearQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_hidden = nn.Linear(input_size, hidden_size)
        self.hidden_output = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        to_hidden = F.relu(self.input_hidden(input))
        to_output = self.hidden_output(to_hidden)

        return to_output

    def save(self, file_name='model.pth'):
        model_folder_path = './model'

        if not os.path.exists(model_folder_path):
            os.mkdir(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class Trainer:
    def __init__(self, model, learning_rate, gamma):
        self.model = model
        self.lr = learning_rate
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, old_state, action, reward, new_state, done):
        # check whether each parameter is multiple values or a single value
        old_state = torch.tensor(old_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        new_state = torch.tensor(new_state, dtype=torch.float)

        # if multiple values, it is in the form (n, param), n is number of batches

        if len(old_state.shape) == 1:
            # it is a single value, in the form (1, state)
            old_state = torch.unsqueeze(old_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            new_state = torch.unsqueeze(new_state, 0)
            done = (done, )

        # Q values for each batch from the old state
        prediction = self.model(old_state)

        target = prediction.clone()
        # refine Q values using bellman equation if not done

        for index in range(len(done)): # for each batch
            if not done[index]:
                """
                # if this batch(game) isn't done
                # use Bellman equation to find new Q-value
                
                new Q val = current reward from this batch + discount rate * max possible reward from next state
                """
                Q_new = reward[index] + self.gamma * torch.max(self.model(new_state[index]))

            else:
                # if done, new Q val is reward from this game(batch)
                Q_new = reward[index]
            # set up target value, where it is trying to maximise Q val

            try:
                target[index][torch.argmax(action).item()] = Q_new
            except IndexError:
                pass
            # apply loss function, then back-propagate

        self.optimizer.zero_grad()
        loss = self.criterion(target, prediction)
        loss.backward()

        self.optimizer.step()



