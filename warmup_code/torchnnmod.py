import torch

#torch initialisation
dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") #to run on GPU

#data initialisation
batch_size, inpdim, hid_size, outdim = 64, 1000, 100, 10
x = torch.randn(batch_size, inpdim, device = device, dtype = dtype)
y = torch.randn(batch_size, outdim, device = device, dtype = dtype)


#use the nn module to define our sequence of layers. nn.sequential hold all the other layers in a sequence to produce the output. Each module inside nn.sequential will hold the output for the next module and also the weights and bias. 

model = torch.nn.Sequential(
    torch.nn.Linear(inpdim, hid_size),
    torch.nn.ReLU(),
    torch.nn.Linear(hid_size, outdim)
)

#torch.nn also has a loss function defined inside it which we can use. 

loss_func = torch.nn.MSELoss(reduction = 'sum')


learning_rate = 1e-6
for epoch in range(500):

    #forward prop. This will propgate the input through all the layers of nn.sequential
    y_hat = model(x)

    #loss function
    loss = loss_func(y_hat, y)

    #zero the gradients before going for backward prop
    model.zero_grad()

    #will compute backward gradients
    loss.backward()

    #params are biases and weights. 
    with torch.no_grad():
        for params in model.parameters():
            params -= learning_rate*params.grad
