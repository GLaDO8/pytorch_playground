import torch

#torch initialisation
dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") #to run on GPU

#data initialisation
batch_size, inpdim, hid_size, outdim = 64, 1000, 100, 10
x = torch.randn(batch_size, inpdim, device = device, dtype = dtype)
y = torch.randn(batch_size, outdim, device = device, dtype = dtype)

w1 = torch.randn(inpdim, hid_size, device = device, dtype = dtype)
w2 = torch.randn(hid_size, outdim, device = device, dtype = dtype)

learning_rate = 1e-6
for epoch in range(500):
    #forward pass
    z1 = x.mm(w1) # matrix multiply
    h_ReLu = z1.clamp(min = 0)
    y_hat = h_ReLu.mm(w2)

    #loss calc
    loss = (y_hat - y).pow(2).sum().item()
    print(epoch, loss)

    #backward pass
    #x(64, 1000)
    #y(64, 10)
    #w1(1000, 100)
    #w2(100, 10)
    grad_y_hat = 2*(y_hat - y)
    grad_w2 = h_ReLu.t().mm(grad_y_hat)
    grad_h_ReLu = grad_y_hat.mm(w2.t())
    grad_h = grad_h_ReLu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    #weight update
    w1 -= learning_rate*grad_w1
    w2 -= learning_rate*grad_w2