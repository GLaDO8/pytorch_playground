import numpy as np

batch_size, inpdim, hid_size, outdim = 64, 1000, 100, 10
x = np.random.randn(batch_size, inpdim)
y = np.random.randn(batch_size, outdim)

w1 = np.random.randn(inpdim, hid_size)
w2 = np.random.randn(hid_size, outdim)

for epoch in range(500):
    #forward pass
    z1 = x.dot(w1)
    h_ReLu = np.maximum(z1, 0) #np.maximum does elementwise maximum and returns the maximum of same dim
    y_hat = h_ReLu.dot(w2)

    #loss calc
    loss = np.sum(np.square(y_hat - y))
    print(epoch, loss)

    #backward pass
    grad_l_y_hat = 2*(y_hat - y)
    grad_w2 = h_ReLu.T.dot(grad_l_y_hat)
    grad_h_ReLu = grad_l_y_hat.dot(w2.T)









