import numpy as np
import torch
from init_vehicle_sim import P_Vehicle
from init_steer_actuator_sim import init_steer_actuator_sim
def load_transfer(acc_x, acc_y, p_vehicle):    
    delta_x = p_vehicle.m * acc_x * p_vehicle.h / p_vehicle.l / 2  # Axle load transfer
    delta_y = p_vehicle.m * acc_y * p_vehicle.h / p_vehicle.b / 2  # Wheel load transfer    
    # Vertical loads batch_size,4,1 
    Fz=p_vehicle.Fz_static-(torch.sign(p_vehicle.tire_x) * delta_x + torch.sign(p_vehicle.tire_y) * delta_y)
    Fz = torch.max(Fz,torch.tensor([10.0],device = p_vehicle.device))
    return Fz

if __name__ == '__main__':
    p_vehicle = P_Vehicle()
    init_steer_actuator_sim(p_vehicle)
    batch_size  = 2
    acc_x = torch.ones(batch_size,1,device=p_vehicle.device)
    acc_y = torch.ones(batch_size,1,device=p_vehicle.device)
    Fz = load_transfer(acc_x, acc_y, p_vehicle)
    print((Fz))
    print((Fz[0,2]+Fz[0,3])/p_vehicle.m/10.0)