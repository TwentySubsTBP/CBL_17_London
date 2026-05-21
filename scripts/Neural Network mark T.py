import torch as pt 
from torch.utils.data import DataLoader

"""
We're going to create a Neural Network that trains on the crime data
"""

input_dim = 1
hidden_dim1 = 1
hidden_dim2 = 1
output_dim = 1

" This is the actual Neural Network architecture that is going to be used"

class CrimeRiskNetwork(pt.nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = pt.nn.Linear(input_dim, hidden_dim1)
        self.layer_2 = pt.nn.linear(hidden_dim1, hidden_dim2)
        self.output_layer = pt.nn.linear(hidden_dim2, output_dim)

        pt.nn.init.kaiming_uniform(self.layer_1.weight, nonlinearity="relu")
        pt.nn.init.kaiming_uniform(self_layer_2.weight, nonlinearity = "relu")


    def forward(self, x):
        x = pt.nn.functional.relu(self.layer_1(x))
        x = pt.nn.functional.sigmoid(self.layer_2(x))

        logits = self.output_layer(x)

        return logits
       
model = CrimeRiskNetwork(input_dim, hidden_dim1, hidden_dim2, output_dim)
print(model)


"This part is used for training"

learning_rate = 0.01
loss_fn = pt.nn.BCEWithLogitsLoss
optimizer = pt.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay=0.001)

num_epochs = 10000 # Epochs are basically just the number of times a dataset is 'studied' by a Neural Net 
loss_values = []

training_data = pd.read_parquet()
test_data = pd.read_parquet()

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

for epoch in range(num_epochs):
    for X, y in pt.train_dataloader:
        # zero the parameter gradients
        optimizer.zero_grad()
       
        # forward + backward + optimize
        pred = model(X)
        loss = loss_fn(pred, y.unsqueeze(-1))
        loss_values.append(loss.item())
        loss.backward()
        optimizer.step()

print("Training Complete")

"""
Training Complete
"""

