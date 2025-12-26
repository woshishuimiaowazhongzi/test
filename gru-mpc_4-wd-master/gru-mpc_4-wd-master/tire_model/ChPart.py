import torch
class ChPart:
    def __init__(self, name, device='cpu'):
        self.name = name
        self.device = device
        self.mass = torch.tensor(0.0, device=device)
        self.inertia = torch.zeros(3, device=device)

    def Create(self, d):
        if "Name" in d: self.name = d["Name"]
        if "Mass" in d: self.mass = torch.tensor(d["Mass"], device=self.device)
        if "Inertia" in d: self.inertia = torch.tensor(d["Inertia"], device=self.device)