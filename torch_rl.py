import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
from env import HangmanEnv
from torch.autograd import Variable

import yaml


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import torchvision.transforms as T

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
obscured_string_len = 27


env = HangmanEnv()

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

config = None

with open("config.yaml", 'r') as stream:
	try:
		config = yaml.safe_load(stream)
	except yaml.YAMLError as exc:
		print(exc)


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    
class DQN(nn.Module):

    def __init__(self, config):
        super(DQN, self).__init__()
        self.rnn_name = config['rnn']
        self.input_dim = config['vocab_size'] + 1
        self.hidden_dim = config['hidden_dim'] 
        self.num_layers = config['num_layers']
        self.embed_dim = config['embedding_dim']
        self.output_dim = config['vocab_size']

        #whether to use character embeddings
        if config['use_embedding']:
            self.use_embedding = True
            self.embedding = nn.Embedding(self.input_dim, self.embed_dim)
        else:
            self.use_embedding = False
            
        #linear layer after RNN output
        in_features = config['miss_linear_dim'] + self.hidden_dim*2
        mid_features = config['output_mid_features']
        self.linear1_out = nn.Linear(in_features, mid_features)
        self.relu = nn.ReLU()
        self.linear2_out = nn.Linear(mid_features, self.output_dim)

        #linear layer after missed characters
        self.miss_linear = nn.Linear(config['vocab_size'], config['miss_linear_dim'])        

        #declare RNN
        if self.rnn_name == 'LSTM':
            self.rnn = nn.LSTM(input_size=self.embed_dim if self.use_embedding else self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                               dropout=config['dropout'],
                               bidirectional=True, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=self.embed_dim if self.use_embedding else self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                              dropout=config['dropout'],
                              bidirectional=True, batch_first=True)

        #optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=config['lr'])
        
        num_classes = 26
        num_layers = 1
        input_size = 27
        hidden_size = 32
        seq_length = 27
        
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, num_classes) #fully connected 1
        self.softmax = nn.Softmax(num_classes)
        # self.fc = nn.Linear(num_classes, num_classes) #fully connected last layer

        self.relu = nn.ReLU()
    
    # def forward(self,x):
    #     print("X = ", x[0].shape)
    #     # print())
    #     # obscured_string = np.reshape(x[0], (None,x[0].shape[0],32))
    #     # actions_used = np.reshape(x[1], (None, x[1].shape[0],1))
    #     in1 = torch.tensor(x[0])
    #     in2 = torch.tensor(x[1])
    #     # in1 = in1.type(torch.LongTensor)
    #     # in2 = in2.type(torch.LongTensor)
    #     print("this = ", in2.size(0))
    #     h_0 = Variable(torch.zeros(self.num_layers, in1.size(0), self.hidden_size)).type(torch.LongTensor) #hidden state
    #     c_0 = Variable(torch.zeros(self.num_layers, in1.size(0), self.hidden_size)).type(torch.LongTensor) #internal state
        
    #     print("H0 =", h_0.type())
    #     print("C0 =", c_0.type())
    #     print("in1 =", in1.type())
    #     print("in1=", in2.type())
    #     print(h_0)
    #     # Propagate input through LSTM
    #     output, (hn, cn) = self.lstm(in1, (h_0, c_0)) #lstm with input, hidden, and internal state
    #     hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
    #     combined = torch.cat((hn, in2), 1)
    #     out = self.relu(combined)
    #     print("Out = ", out)
    #     out = self.fc_1(out) #first Dense
    #     # out = self.relu(out) #relu
    #     # out = self.fc(out) #Final Output
    #     out = self.softmax(out)
    #     return out
    
    def forward(self, x, x_lens, miss_chars):
        """
        Forward pass through RNN
        :param x: input tensor of shape (batch size, max sequence length, input_dim)
        :param x_lens: actual lengths of each sequence < max sequence length (since padded with zeros)
        :param miss_chars: tensor of length batch_size x vocab size. 1 at index i indicates that ith character is NOT present
        :return: tensor of shape (batch size, max sequence length, output dim)
        """        
        if self.use_embedding:
            x = self.embedding(x)
            
        batch_size, seq_len, _ = x.size()
        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)
        # now run through RNN
        output, hidden = self.rnn(x)
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)
        hidden = hidden[-1]
        hidden = hidden.permute(1, 0, 2)

        hidden = hidden.contiguous().view(hidden.shape[0], -1)
        #project miss_chars onto a higher dimension
        miss_chars = self.miss_linear(miss_chars)
        #concatenate RNN output and miss chars
        concatenated = torch.cat((hidden, miss_chars), dim=1)
        #predict
        return self.linear2_out(self.relu(self.linear1_out(concatenated)))
    
    
    def initHidden(self, device):
        return (torch.zeros(1, 1, self.hidden_size).to(device), torch.zeros(1, 1, self.hidden_size).to(device))
    
    def calculate_loss(self, model_out, labels, input_lens, miss_chars, use_cuda):
        """
        :param model_out: tensor of shape (batch size, max sequence length, output dim) from forward pass
        :param labels: tensor of shape (batch size, vocab_size). 1 at index i indicates that ith character should be predicted
        :param: miss_chars: tensor of length batch_size x vocab size. 1 at index i indicates that ith character is NOT present
							passed here to check if model's output probability of missed_chars is decreasing
        """
        outputs = nn.functional.log_softmax(model_out, dim=1)
        #calculate model output loss for miss characters
        miss_penalty = torch.sum(outputs*miss_chars, dim=(0,1))/outputs.shape[0]
        
        input_lens = input_lens.float()
        #weights per example is inversely proportional to length of word
        #this is because shorter words are harder to predict due to higher chances of missing a character
        weights_orig = (1/input_lens)/torch.sum(1/input_lens).unsqueeze(-1)
        weights = torch.zeros((weights_orig.shape[0], 1))    
        #resize so that torch can process it correctly
        weights[:, 0] = weights_orig

        if use_cuda:
        	weights = weights.cuda()
        
        #actual loss
        loss_func = nn.BCEWithLogitsLoss(weight=weights, reduction='sum')
        actual_penalty = loss_func(model_out, labels)
        return actual_penalty, miss_penalty
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    # def forward(self, x):
    #     print("X = ", x[0].shape)
    #     x = torch.tensor(x).to(device)
    #     x = F.relu(self.bn1(self.conv1(x)))
    #     x = F.relu(self.bn2(self.conv2(x)))
    #     x = F.relu(self.bn3(self.conv3(x)))
    #     return self.head(x.view(x.size(0), -1))
    
    

n_actions = env.action_space.n

policy_net = DQN(config).to(device)
target_net = DQN(config).to(device)
print(policy_net)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            # print("This value is ", policy_net(state).max(1))
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())
        
        
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    print("State batch = ", state_batch.shape)
    print("action batch = ", action_batch.shape)
    print("reward batch = ", reward_batch.shape)
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device, dtype=torch.long)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
    
num_episodes = 50
for i_episode in range(num_episodes):
    # Initialize the environment and state
    last_state = None
    state = env.reset()
    print(state)
    # last_screen = get_screen()
    # current_screen = get_screen()
    # state = current_screen - last_screen
    for t in count():
        state = (torch.tensor(state[0].reshape(-1, 1, 27), dtype=torch.long), torch.tensor(state[1], dtype=torch.long))
        # state = torch.tensor(state)
        print(t)
        # Select and perform an action
        print("Selecting Action")
        action = select_action(state)
        print("Action selected = ", action)
        next_state, reward, done, info = env.step(action.item())
        print(next_state, reward)
        reward = torch.tensor([reward], device=device)
        next_state = (torch.tensor(next_state[0].reshape(-1, 1, 27), dtype=torch.long), torch.tensor(next_state[1], dtype=torch.long))
        # next_state = torch.tensor(next_state)
        # Observe new state
        # last_state = state
        # state=next_state
        # current_screen = get_screen()
        # if not done:
        #     next_state = current_screen - last_screen
        # else:
        #     next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            # plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()