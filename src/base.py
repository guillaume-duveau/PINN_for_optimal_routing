import torch, torch.nn as nn

c = 1.0                    # toy example airspeed (normalized units)
goal = torch.tensor([[1.0, 0.5]])

def wind(x):                              # toy jet stream along +x, must satisfy |w| < c
    jet = 0.5 * torch.exp(-((x[:, 1:2] - 0.5) / 0.1) ** 2)
    return torch.cat([jet, torch.zeros_like(jet)], dim=1)

class VNet(nn.Module):
    """V(x) = |x - goal| * softplus(MLP(x)): enforces V(goal)=0 and V>=0 by construction."""
    def __init__(self, width=32, depth=3):
        super().__init__()
        layers, d = [], 2
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        layers += [nn.Linear(d, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        r = torch.sqrt(((x - goal) ** 2).sum(1, keepdim=True) + 1e-12)
        return r * nn.functional.softplus(self.mlp(x))

def grad_V(model, x):
    """
    model : PINN model developed above
    x : 
    """
    x = x.clone().requires_grad_(True)
    V = model(x)
    g, = torch.autograd.grad(V.sum(), x, create_graph=True)
    return x, V, g

def residual(model, x,wind_field=c):
    x, V, g = grad_V(model, x)
    gnorm = torch.sqrt((g ** 2).sum(1, keepdim=True) + 1e-12)
    return wind_field * gnorm - (wind(x) * g).sum(1, keepdim=True) - 1.0   # c|∇V| - w·∇V - 1


def outputs_simple_trained_model(): 
    
    """
    Returns a toy example model trained in a single case with the constant univariate wind
    Outputs: model, loss_curve
    
    """

    model_simple = VNet()
    opt = torch.optim.Adam(model_simple.parameters(), lr=1e-3)

    loss_curve = []

    for _ in range(5000):
        x = torch.rand(2048, 2)               # collocation points in [0,1]^2
        loss = (residual(model_simple, x) ** 2).mean()
        opt.zero_grad()
        loss.backward() 
        opt.step()
        loss_curve.append(loss.item())

        return model_simple , loss_curve