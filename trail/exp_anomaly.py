import torch
from torch import autograd as autograd

autograd.set_detect_anomaly(True)


if __name__ == "__main__":
    with autograd.detect_anomaly():
        a = torch.tensor([1e40, 3.0], requires_grad=True)
        b = torch.tensor([-12e40, 4.0], requires_grad=True)
        Q = torch.exp(3 * a**3 - b**2)
        external_grad = torch.tensor([1.0, 1.0])
        Q.backward(gradient=external_grad)
        print(a.grad)
        print(b.grad)
