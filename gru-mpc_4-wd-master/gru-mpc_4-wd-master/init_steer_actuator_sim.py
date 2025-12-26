import torch


def init_steer_actuator_sim(p_vehicle):

    # switch case 选择不同的执行器类型  
    p_vehicle.tire_act_delta = torch.tensor([[1.0], 
                                        [1.0], 
                                        [0.0],
                                        [0.0]], device=p_vehicle.device)
    p_vehicle.is_all_wheel_steer = False
    p_vehicle.is_all_axle_steer = False
    
    # 获取轮胎数量和执行器数量
    p_vehicle.tire_num, p_vehicle.delta_num = p_vehicle.tire_act_delta.shape
    
    # 统计tire_act_delta每一列非零元素的个数
    delta_nnz = torch.sum(p_vehicle.tire_act_delta != 0, dim=0, keepdim=True)  # sum(tire_act_delta~=0,1)

    p_vehicle.act_tire_delta = torch.zeros((p_vehicle.delta_num, p_vehicle.tire_num), device=p_vehicle.device)

    # 转向反映射 执行器转角<-轮胎转角
    for tire_index in range(p_vehicle.tire_num):
        for delta_index in range(p_vehicle.delta_num):
            temp = p_vehicle.tire_act_delta[tire_index, delta_index]
            if temp != 0:
                p_vehicle.act_tire_delta[delta_index, tire_index] = 1.0 / temp / delta_nnz[0, delta_index]  
    p_vehicle.drive_num = 4
    p_vehicle.Nu = p_vehicle.drive_num
    p_vehicle.Nx = 3 + p_vehicle.tire_num

    
if __name__ == '__main__':
# 使用示例:
    from init_vehicle_sim import P_Vehicle 
    vehicle = P_Vehicle(device='cuda' if torch.cuda.is_available() else 'cpu')
    init_steer_actuator_sim(vehicle)
    #打印vehicle属性
    # 打印 vehicle 的所有属性
    print("Vehicle 对象的所有属性:")
    print("=" * 50)

    for attr_name in sorted(dir(vehicle)):
        # 跳过私有属性和方法
        if not attr_name.startswith('__') and not callable(getattr(vehicle, attr_name)):
            attr_value = getattr(vehicle, attr_name)
            print(f"{attr_name}:")
            # 打印属性值
            print(attr_value)
    print(vehicle.to_numpy().tire_x)