import torch, torch.nn as nn
import numpy as np

airspeed = 1.0                    # toy example airspeed (normalized units)
goal = torch.tensor([[1.0, 0.5]])

def wind_jet_along_x(x,rotation_angle=0): 

    """
    Generated a jet stream rotated by the angle rotation_angle, with a velocity 
    following a normal distribution 
    """  
    
    jet = 0.5 * torch.exp(-((x[:, 1:2] - 0.5) / 0.1) ** 2)
    rotation_angle = rotation_angle/180*np.pi
    mat_rot = np.array([[np.cos(rotation_angle),-np.sin(rotation_angle)],[np.sin(rotation_angle),np.cos(rotation_angle)]])   
    rotated_jet = np.dot(torch.cat([jet, torch.zeros_like(jet)], dim=1),mat_rot)
    
    return rotated_jet

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

def residual(model, x):
    x, V, g = grad_V(model, x)
    gnorm = torch.sqrt((g ** 2).sum(1, keepdim=True) + 1e-12)
    return airspeed * gnorm - (wind(x) * g).sum(1, keepdim=True) - 1.0   # c|∇V| - w·∇V - 1


def route(model, x0, dt=1e-2, steps=2000, tol=2e-2):
    x = torch.tensor([x0], dtype=torch.float32)
    path = [x.clone()]
    for _ in range(steps):
        xg, _, g = grad_V(model, x)
        u = -g / (g.norm(dim=1, keepdim=True) + 1e-12)
        x = (xg + dt * (airspeed * u + wind(xg))).detach()
        path.append(x.clone())
        if (x - goal).norm() < tol:
            break
    return torch.cat(path).numpy()


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


