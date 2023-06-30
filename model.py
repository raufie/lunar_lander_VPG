from torch import nn
class MLP(nn.Module):
    def __init__(self, input_features=24, n_actions=4, hidden_units=100):
        super(MLP, self).__init__()
        
        self.linear1 = nn.Linear(input_features, hidden_units)
        self.linear2 = nn.Linear(hidden_units, hidden_units)
        self.linear3 = nn.Linear(hidden_units, n_actions)
#         self.mean = nn.Linear(100, n_actions)
#         self.logstd = nn.Linear(100, n_actions)
        
    def forward(self, x):
        
        x = self.linear1(x)
        x = nn.ReLU()(x)
        x = self.linear2(x)
        x = nn.ReLU()(x)
        x = self.linear3(x)

        
        return x
        