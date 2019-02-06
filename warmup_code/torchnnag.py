import torch

#torch initialisation
dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") #to run on GPU

#data initialisation
batch_size, inpdim, hid_size, outdim = 64, 1000, 100, 10
x = torch.randn(batch_size, inpdim, device = device, dtype = dtype)
y = torch.randn(batch_size, outdim, device = device, dtype = dtype)

w1 = torch.randn(inpdim, hid_size, device = device, dtype = dtype, requires_grad=True)
w2 = torch.randn(hid_size, outdim, device = device, dtype = dtype, requires_grad=True)

learning_rate = 1e-6
for epoch in range(500):
    #forward pass
    #no need to store intermediate values as we are not calculating gradients manually
    y_hat = x.mm(w1).clamp(min = 0).mm(w2)

    #loss
    loss = (y_hat - y).pow(2).sum().item()
    
    #backward pass
    #this will compute gradients of loss with respect to all tensors with requires_grad = true

    #after this call, w1.grad() and w2.grad() will hold the gradient of loss wrt w1 and w2
    loss.backward()

    #update weights 
    #torch.no_grad tells pytorch to stop tracking the gradients as w1 and w2 still have require_grad = true
    with torch.no_grad():
        w1 -= learning_rate*w1.grad
        w2 -= learning_rate*w2.grad




