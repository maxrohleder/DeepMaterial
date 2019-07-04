import torch
from tqdm import trange
import time

dtype = torch.float
device = torch.device("cpu")
#device = torch.device("cuda:0")

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64000*3, 1000, 100, 10

print("total trainable weights: ", H*D_in+H*D_out)

# Create random input and output data
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Randomly initialize weights
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)
start = time.time()
lr = 1e-6
for t in trange(500):
    # forward pass
    h = x.mm(w1)
    h_relu = h.clamp_min(0)
    y_pred = h_relu.mm(w2)

    # loss
    loss = (y_pred - y).pow(2).sum().item()
    #print(t, loss)

    # backprob
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # update
    w1 -= lr * grad_w1
    w2 -= lr * grad_w2

print("duration: ", time.time() - start)