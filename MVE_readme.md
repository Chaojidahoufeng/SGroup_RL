# MVE 笔记

## 3p1t2f

### make_world()
1. self.add_direction_encoder 将direction编码
2. 加入N个agent（直接写到了环境里）,设置硬件参数
3. 加入landmark
4. 加入obstacle
5. world.data_slot['max_step_number']?
6. reset_world()

### reset_world()
1. 设置agent的theta, vel_b, phi, ctrl_vel_b, ctrl_phi, movable, crashed
2. 设置landmark, obstacle和agent的coordinate, 去除conflict
3. 随机选择一个landmark作为real_landmark, real_landmark.color[1] = 1.0
4. direction是real_landmark.state.coordinate[0,1]-agent.state.coordinate[0,1]; ++=>0 -+=>1 --=>2 +-=>3, 然后转入one-hot编码，存入agent.data_slot['direction_obs']
 <这里是否应该改成相对参考系下的判断？>

### reward()
1. utils.naive_inference要好好看看，貌似有点奇怪。 ^是异或操作，相同为0。
in_min_r,  xt<0 -> vel = 1
in_min_r,  xt>0 -> vel = -1
~in_min_r, xt>0 -> vel = 1
~in_min_r, xt<0 -> vel = -1

in_min_r,  yt<0 -> phi = 1
in_min_r,  yt>0 -> phi = -1
~in_min_r, yt>0 -> phi = 1
~in_min_r, yt<0 -> phi = -1
目前in_min_r肯定是false，因此xt和yt直接决定vel和phi的正负，很合理
vel,phi存入prefer_action

2. same_direction 判断agent.state.ctrl_vel_b/ctrl_phi与prefer_action一致，rew会增加 rew += 1.0 * direction_alpha
3. 判断是否reach，rew += 1.0
4. 判断是否crashed，rew -= 1.0

### observation()
1. agent.data_slot['direction_obs']], 
   agent_pos是当前agent的位置， 
   landmark_pos是landmark的位置，注意是所有landmark的位置
   obstacle_pos是obstacle的位置，注意也是所有obstacle的位置
   in_view: 1.0 landmark在view_threshold之内


action和实际动作的映射, 返回的二元组是ctrl_vel_b,ctrl_phi
self.discrete_table:Dict[int,Tuple[float,float]] = {0:( 0.0, 0.0),
                      1:( 1.0, 0.0), 2:( 1.0, 1.0), 3:( 1.0, -1.0),
                      4:(-1.0, 0.0), 5:(-1.0, 1.0), 6:(-1.0, -1.0)}

目前使用的是1s控制一次？



## _update_one_sim_step_warp()
单步控制的部分：src/MultiVehicleEnv/basic/_update_one_sim_step_warp
dv_dt: 加速度，写定在scenario中,0.2
K_vel: 0.707，这个是干啥的
dphi_dt: 车的前轮偏转度的角速度

ctrl_vel_b: state中的量

## _update_one_sim_step_njit()
state.phi: 车的前轮的偏转度
state.theta: 车头朝向角
state.omega: 角速度