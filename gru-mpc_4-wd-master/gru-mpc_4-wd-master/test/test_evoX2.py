import torch
from evox.core import Algorithm, Mutable, Parameter, Problem
from evox.utils import clamp
from evox.workflows import EvalMonitor, StdWorkflow
from evox.algorithms import PSO    

class Sphere(Problem):
    def __init__(self):
        super().__init__()

    # pop_size,网络参数维度
    def evaluate(self, pop: torch.Tensor):
        print(pop.shape)
        # pop后两个维度求和
        return (pop**2).sum(dim=-1)
    
algorithm = PSO(
    pop_size=100,
    lb=torch.tensor([-10.0]*2),
    ub=torch.tensor([10.0]*2),
    w=0.6,
    phi_p=2.5,
    phi_g=0.8,
)
problem = Sphere()
monitor = EvalMonitor()
workflow = StdWorkflow(algorithm=algorithm, problem=problem, monitor=monitor)

for _ in range(100):
    workflow.step()
workflow.monitor.plot()
fig = monitor.plot()
fig.show()