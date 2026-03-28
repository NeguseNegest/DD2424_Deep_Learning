import torch
import numpy as np

def ComputeGradsWithTorch(X, y, network_params, lam=0):
    Xt = torch.from_numpy(X)

    W = torch.tensor(network_params['W'], requires_grad=True)
    b = torch.tensor(network_params['b'], requires_grad=True)

    N = X.shape[1]

    scores = torch.matmul(W, Xt) + b
    apply_softmax = torch.nn.Softmax(dim=0)
    P = apply_softmax(scores)

    loss = torch.mean(-torch.log(P[y, np.arange(N)]))
    cost = loss + lam * torch.sum(torch.multiply(W, W))

    cost.backward()

    grads = {}
    grads['W'] = W.grad.numpy()
    grads['b'] = b.grad.numpy()
    return grads