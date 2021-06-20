import torch



class Feedforward(torch.nn.Module):
    
    def __init__(self, input_size=2, hidden_size=500):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = torch.nn.Linear(self.hidden_size, input_size-1)
       
        
    def forward(self, x):
        hidden = (self.fc1(x))
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.relu(output)
        output = self.fc3(output)
        return output
    
    def predict(self, x, debug=False):
        return self.forward(x)   


def train_nn(model, x_train, y_train):
    
#     optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.005)
    
    model.train()
    epoch = 250
    for epoch in range(epoch):
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(x_train)
        # Compute Loss
        diff = y_train - y_pred
#         loss = torch.matmul(torch.transpose(diff, 0, 1), diff).sum() / diff.shape[0]
        loss = (diff**2).sum()
#         import pdb;pdb.set_trace()
#         import pdb; pdb.set_trace()
#         loss = loss_criteria(y_pred.squeeze(), y_train)

#         print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
        # Backward pass
        loss.backward(retain_graph=True)
        optimizer.step()
    return model