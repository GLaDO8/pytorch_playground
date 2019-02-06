import torch

#torch initialisation
dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") #to run on GPU

#data initialisation
batch_size, inpdim, hid_size, outdim = 64, 1000, 100, 10
x = torch.randn(batch_size, inpdim, device = device, dtype = dtype)
y = torch.randn(batch_size, outdim, device = device, dtype = dtype)

model = torch.nn.Sequential(
    torch.nn.Linear(inpdim, hid_size),
    torch.nn.ReLU(),
    torch.nn.Linear(hid_size, outdim)
)

loss_func = torch.nn.MSELoss(reduction = 'sum')

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters, learning_rate)

for epoch in range(500):
    y_hat = model(x)
    loss = loss_func(y_hat, y)

    #zero all gradients as by default gradients are not overwritten but fed into buffers
    optimizer.zero_grad()

    #compute all gradients 
    loss.backward()

    #update all weights and biases (parameters)
    optimizer.step()

    
