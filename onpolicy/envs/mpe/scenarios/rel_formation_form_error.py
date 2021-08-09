# place the agent at the center of map
# only have formation reward and avoidance reward, no other reward
# --map-max-size 2400 --save-dir "../policy/model_maddpg.ckpt"
import numpy as np
import math as math
import random as random
from numpy.linalg import norm

from onpolicy.envs.mpe.scenario import BaseScenario

import onpolicy.utils.MDS as MDS

class Scenario(BaseScenario):
    def make_world(self, arglist):
        self.args = arglist
        if self.args.use_11_discrete_action:
            from onpolicy.envs.mpe.core_11_discrete import World, Agent, Landmark, Static_obs
        else:
            from onpolicy.envs.mpe.core_alpha import World, Agent, Landmark, Static_obs
        #rendering unit is in centimeter
        world = World()
        # set any world properties first
        world.dim_c = 0 # communication dimension
        #world.damping = 1

        world.width = self.args.map_max_size

        num_follower_agents = 4
        num_leader_agent = 0
        num_agents = num_leader_agent + num_follower_agents
        num_static_obs = self.args.num_static_obs
        num_landmarks = 1# tracking center and side barrier indicators
        num_agent_ray = 60
        # add agents
        world.agents = [Agent() for _ in range(num_agents)]

        # ideal formation topo side length
        world.ideal_side_len = self.args.ideal_side_len
        # calculate the ideal formation topo
        world.ideal_topo_point = [[],[]]
        for i in range(num_follower_agents):
            world.ideal_topo_point[0].append(world.ideal_side_len / np.sqrt(2) * np.cos(i/num_follower_agents*2*np.pi))
            world.ideal_topo_point[1].append(world.ideal_side_len / np.sqrt(2) * np.sin(i/num_follower_agents*2*np.pi))


        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            # check if agent already collided
            agent.collide = True
            agent.leader = False
            agent.silent = True #if i > 0 else False
            #agent.adversary = True if i < num_adversaries else False
            agent.boundary = False
            agent.size = 10*np.sqrt(2)
            agent.accel = 3
            agent.max_speed = 100
            agent.u_noise = 0.2
            agent.color = np.array([0., 1, 0.45]) if not agent.leader else np.array([1.0, 1.0, 0.0])
            agent.crash = 0  # calculate how many time the agent crashed
            agent.ray = np.zeros((num_agent_ray, 2))  # num_agent_ray*[dis, ang, type]
            agent.ang_range = [180/num_agent_ray*(i + 0.5) * np.pi / 180 for i in range(num_agent_ray//2)] + [-180/num_agent_ray*(i + 0.5) * np.pi / 180 for i in range(num_agent_ray//2)]
            # for stream function based avoidance
            agent.avoiding = np.array([False, False])
            agent.U = np.array([0., 0.])
            agent.start_ray = np.array([-1, -1])
            agent.end_ray = np.array([-1, -1])
            agent.min_ray = np.array([0, len(agent.ray)//2])
            agent.obs_r = np.array([0., 0.])
            agent.obs_dis = np.array([2.0, 2.0])
            agent.obs_ang = np.array([0., 0.])
            agent.C = np.array([0., 0.])
            agent.C_bound = np.array([0., 0.])
            agent.stream_err = np.array([0., 0.])

        world.static_obs = [Static_obs() for _ in range(num_static_obs)]
        for i, static_obs in enumerate(world.static_obs):
            static_obs.name = 'static_obs %d' % i
            static_obs.collide = True
            static_obs.movable = False
            static_obs.boundary = False
            static_obs.color = np.array([0.8, 0.8, 0.8])
            static_obs.state.p_vel = np.zeros(world.dim_p)
            static_obs.state.p_ang = 0.0
        world.landmarks = [Landmark() for _ in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.boundary = False

            if i==0:
                landmark.center = True
                landmark.size = 5
                landmark.color = np.array([1, 1, 0.4])
                landmark.distance = 0
            else:
                landmark.center = False
                landmark.size = 20
                landmark.color = np.array([0.5, 0.5, 0.5])

        # make initial conditions
        self.reset_world(world)
        return world

    def set_path(self, path_type, start):
        if path_type == 'square':
            xs = np.array([-start, start, start, -start, -start])
            ys = np.array([-start, -start, start, start, -start])
            xys = list(zip(xs, ys))
        elif path_type == 'line':
            xs = np.array([start[0], 4])
            ys = np.array([start[1], start[1]])
            xys = list(zip(xs, ys))
        return xys

    def reset_world(self, world):
        #np.random.seed(5)
        world.time = 0
        world.distance = 0
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.crash = 0  # reset collision counter
        # random properties for landmarks
        num_dynamic_obs = 0#np.random.randint(1,3)
        world.dynamic_obs = [Dynamic_obs() for _ in range(num_dynamic_obs)]

            #s.state.p_pos = np.array([500.0, 50 * i])

        # set initial states of all agents
        agents_ctr = np.array((0.0, 0.0))
        # add Path
        path_type = 'line'
        world.start = agents_ctr
        world.goal = agents_ctr + 1000 * np.array([1, 1]) #set the boundary range from 0,0 to 1000,1000
        #world.path = self.set_path(path_type, world.start)

        #world.station_num = len(world.path)
        world.station = 1
        #goal_vect = world.goal-agents_ctr
        leader_vel = np.random.uniform(0.1, 0.3)

        for e, agent in enumerate(world.agents):
            agent.agents_ctr = agent.agents_ctr_prev = agents_ctr
            agent.state.p_ang = math.atan2(world.goal[1] - world.start[1], world.goal[0] - world.start[0])
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.p_omg = 0.0
            agent.state.c = np.zeros(world.dim_c)

            if agent.leader:
                agent.state.p_pos = agents_ctr
            else:
                agent.state.p_pos = np.array([self.args.map_max_size/2, self.args.map_max_size/2]) + ( np.random.rand(2) - np.array([1,1]) ) * self.args.agent_init_bound # randomly initialize position range from (0,0) to (100, 100)
            
            #TODO: 这里到底需要计算什么东西
            if not agent.leader:
                agent.dis2leader = None
                agent.ang2leader = None
                agent.dis2leader_prev = None
                agent.ang2leader_prev = None
                # initialize the desire relative position of agents in the formation
                agent.d_des = None # desired distance
                agent.w_des = None # desired w (angle)
                agent.p_des = None # desired position
                agent.p_rel = None # desired 
                agent.his = [] # initialize leader history
                agent.P = np.eye(4) # initialize follower predict variance
                agent.dis2goal = norm(agent.state.p_pos - world.goal) / 100 # meters (m)
                agent.ang2goal = math.atan2(world.goal[1]-agent.state.p_pos[1],
                                              world.goal[0]-agent.state.p_pos[0]) - agent.state.p_ang
                agent.dis2goal_prev = agent.dis2goal
                agent.ang2goal_prev = agent.ang2goal
            else:
                agent.dis2goal = norm(agent.state.p_pos - world.goal) / 100
                agent.ang2goal = math.atan2(world.goal[1]-agent.state.p_pos[1],
                                              world.goal[0]-agent.state.p_pos[0]) - agent.state.p_ang
                agent.dis2goal_prev = None
                agent.ang2goal_prev = None

            agent.err = np.zeros(2)
            agent.err_prev = 0.0
            #agent.dis2goal = norm(agent.state.p_pos - world.goal - agent.p_des) / 100
            #agent.ang2goal = math.atan2(agent.state.p_pos[1] - world.goal[1] - agent.p_des[1], agent.state.p_pos[0] - world.goal[0] - agent.p_des[0])
            #agent.dis2goal_prev = None
            #agent.ang2goal_prev = None
            # if e > 0:
            #     agent.state.p_pos += np.array([2*np.random.uniform(-10, 10), 2*np.random.uniform(-10, 10)])
            #     agent.state.p_ang += np.random.uniform(-np.pi/6,np.pi/6)
            self.set_agent_ray(agent)

        for i, dynamic_obs in enumerate(world.dynamic_obs):
            dynamic_obs.name = 'dynamic_obs %d' % i
            dynamic_obs.collide = True
            dynamic_obs.movable = True
            dynamic_obs.boundary = False
            dynamic_obs.size = 10*np.sqrt(2)
            dynamic_obs.accel = 50
            dynamic_obs.max_speed = 100
            dynamic_obs.color = np.array([0., 0., 1])

        for s in world.static_obs:
            s.size = np.random.uniform(10.0, 50.0)
            s.state.p_pos = np.array([np.random.uniform(100, 1000), np.random.uniform(100, 1000)])
            min_agt_dis = np.min([norm(s.state.p_pos-a.state.p_pos)/100 for a in world.agents])
            while min_agt_dis < 1 or norm(s.state.p_pos-world.goal)/100 < 2:
                s.state.p_pos = np.array([np.random.uniform(100, 1000), np.random.uniform(100, 1000)])
                min_agt_dis = np.min([norm(s.state.p_pos - a.state.p_pos) / 100 for a in world.agents])

        
        for i,landmark in enumerate(world.landmarks):
            if landmark.center:
                landmark.state.p_pos = world.goal

        #np.random.seed()
    def benchmark_data(self, agent, world):
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def get_relAngle(self, center, target):
        ang = math.atan2(target.state.p_pos[1] - center.state.p_pos[1], target.state.p_pos[0] - center.state.p_pos[0]) - center.state.p_ang
        if abs(ang) > np.pi:
            ang -= np.sign(ang) * 2 * np.pi
        return ang

    def wrap2pi(self, ang):
        if abs(ang) > np.pi:
            ang -= np.sign(ang) * 2 * np.pi
        return ang

    def set_agent_ray(self, agent, dis2obs=None, ang2obs=None, obs_size=None):
        ang_range = agent.ang_range
        for i in range(len(agent.ray)):
            #ray_ang = self.wrap2pi(ray_ang + ang_range[i])
            agent.ray[i][0] = 2
            agent.ray[i][1] = self.wrap2pi(ang_range[i])
            #agent.ray[i][2] = 0
            if ang2obs:
                for j in range(len(ang2obs)):
                    delt_ang2obs = abs(self.wrap2pi(ang2obs[j] - agent.ray[i][1]))
                    max_delt_ang2obs = abs(np.tan(obs_size[j]/(obs_size[j] + dis2obs[j])))
                    delt_dis2obs = delt_ang2obs/max_delt_ang2obs*obs_size[j]
                    #max_dis2obs = norm([obs_size[j], obs_size[j] + dis2obs[j]])
                    sense_dis = abs(dis2obs[j] + delt_dis2obs) + np.random.normal(0.0, 0.2)
                    # todo: modify the detect condition
                    if delt_ang2obs < max_delt_ang2obs and sense_dis < agent.ray[i][0]:
                        agent.ray[i][0] = np.clip(sense_dis, agent.size/100, 2.0)

    # collision detect
    def is_collision(self, center, target):
        delta_pos = center.state.p_pos - target.state.p_pos
        if (center.state.p_pos[0] - target.state.p_pos[0]) != 0:
            ang = math.atan((center.state.p_pos[1] - target.state.p_pos[1]) / (
                        center.state.p_pos[0] - target.state.p_pos[0]))
        else:
            ang = np.pi / 2
        dist = norm(delta_pos)/100
        dist_min = (center.size + target.size)*math.cos(abs(ang)-np.pi/4)/100

        return dist <= dist_min

    # return all agents that are not adversaries
    def follower_agents(self, world):
        return [agent for agent in world.agents if not agent.leader]

    # return all adversarial agents
    def leader_agents(self, world):
        return [agent for agent in world.agents if agent.leader]

    # return all static obstacles
    def get_static_obstacles(self, world):
        return [static_obs for static_obs in world.static_obs]

    # return all dynamic obstacles
    def get_dynamic_obstacles(self, world):
        return [dynamic_obs for dynamic_obs in world.dynamic_obs]
    
    # return all landmarks
    def get_landmarks(self, world):
        return [landmark for landmark in world.landmarks]

    # return current state of formation center
    def get_formation_center(self, world):
        return world.agents[0].p_pos

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        #boundary_reward = -10 if self.outside_boundary(agent) else 0
        main_reward = self.dis2goal_reward(agent, world)
        #main_reward = self.leader_reward(agent, world) if agent.leader else self.agent_reward(agent, world)
        return main_reward

    def outside_boundary(self, entity):
        # 20190711 restrict the agents in the frame
        if entity.state.p_pos[0] + entity.size > self.args.map_max_size:
            entity.state.p_pos[0] = self.args.map_max_size - entity.size
            #return True
        if entity.state.p_pos[0] - entity.size < 0:
            entity.state.p_pos[0] = 0 + entity.size
            #return True
        if entity.state.p_pos[1] + entity.size > self.args.map_max_size:
            entity.state.p_pos[1] = self.args.map_max_size - entity.size
            #return True
        if entity.state.p_pos[1] - entity.size < 0:
            entity.state.p_pos[1] = 0 + entity.size
            #return True
        #return False

    def get_virtual_obstacle_pos(self, ray, obs_r):
        left_start, left_end, left_min, right_start, right_end, right_min = -1, -1, 0, -1, -1, len(ray)//2

        x = [0., 0.]
        y = [0., 0.]
        left_avoid, right_avoid = False, False
        detect_range = 2
        # check left side
        for n in range(0, len(ray) // 2):
            if ray[n][0] < detect_range:
                if left_start < 0:
                    left_start = n
                else:
                    left_end = n
            else:
                if left_end > 0:
                    break

        # check right side
        for n in range(len(ray)//2, len(ray)):
            if ray[n][0] < detect_range:
                if right_start < 0:
                    right_start = n
                else:
                    right_end = n
            else:
                if right_end > 0:
                    break

        if left_start >= 0 and left_end >= 0:
            left_avoid = True
        if right_start >= 0 and right_end >= 0:
            right_avoid = True
        avoid = [left_avoid, right_avoid]

        for i in range(len(avoid)):
            if avoid[i]:
                min_dis = 2
                if i == 0:
                    for n in range(left_start+1, left_end):
                        if ray[n][0] < min_dis:
                            left_min = n
                            min_dis = ray[n][0]
                else:
                    for n in range(right_start+1, right_end):
                        if ray[n][0] < min_dis:
                            right_min = n
                            min_dis = ray[n][0]

                # relative position of the three chosen points with respect of the agent's coordinate system
                d_s, x_s, y_s = np.array([ray[left_start][0], ray[right_start][0]]), np.array([ray[left_start][0]*np.cos(ray[left_start][1]), ray[right_start][0]*np.cos(ray[right_start][1])]), np.array([ray[left_start][0]*np.sin(ray[left_start][1]), ray[right_start][0]*np.sin(ray[right_start][1])])
                d_e, x_e, y_e = np.array([ray[left_end][0], ray[right_end][0]]), np.array([ray[left_end][0]*np.cos(ray[left_end][1]), ray[right_end][0]*np.cos(ray[right_end][1])]), np.array([ray[left_end][0]*np.sin(ray[left_end][1]), ray[right_end][0]*np.sin(ray[right_end][1])])
                d_m, x_m, y_m = np.array([ray[left_min][0], ray[right_min][0]]), np.array([ray[left_min][0]*np.cos(ray[left_min][1]), ray[right_min][0]*np.cos(ray[right_min][1])]), np.array([ray[left_min][0]*np.sin(ray[left_min][1]), ray[right_min][0]*np.sin(ray[right_min][1])])

                # requirements for solving obstacle radius (using concept of circumcentre)
                left_theta1 = np.array([ray[left_min][1] - ray[left_start][1], ray[left_end][1] - ray[left_min][1]])
                right_theta1 = np.array([ray[right_start][1] - ray[right_min][1], ray[right_min][1] - ray[right_end][1]])
                left_theta2 = [0.0, 0.0]
                right_theta2 = [0.0, 0.0]
                if left_theta1[0] != 0:
                    left_theta2[0] = np.arcsin(d_m[0] * np.sin(left_theta1[0]) / norm([x_s[0] - x_m[0], y_s[0] - y_m[0]]))
                if left_theta1[1] != 0:
                    left_theta2[1] = np.arcsin(d_m[0] * np.sin(left_theta1[1]) / norm([x_e[0] - x_m[0], y_e[0] - y_m[0]]))
                if right_theta1[0] != 0:
                    right_theta2[0] = np.arcsin(d_m[1] * np.sin(right_theta1[0]) / norm([x_s[1] - x_m[1], y_s[1] - y_m[1]]))
                if right_theta1[1] != 0:
                    right_theta2[1] = np.arcsin(d_m[1] * np.sin(right_theta1[1]) / norm([x_e[1] - x_m[1], y_e[1] - y_m[1]]))
                theta_a = np.array([np.sum(left_theta1)+np.sum(left_theta2), np.sum(right_theta1)+np.sum(right_theta2)])
                # solving obstacle radius, circumcentre coordinate

                # if obstacle is not convex, return results according to start and end points
                if x_m[i] == x_s[i] or x_m[i] == x_e[i] or x_e[i] == x_s[i] or y_m[i] == y_s[i] or y_m[i] == y_e[i] or y_e[i] == y_s[i] or d_m[i] == d_e[i] or d_m[i] == d_s[i] or d_s[i] == d_e[i]:
                    if obs_r[i] == 0.:
                        obs_r[i] = np.clip(norm([x_s[i]-x_e[i], y_s[i]-y_e[i]]), 0.1, 5.0)
                    
                    x[i] = - x_m[i]
                    y[i] = - y_m[i]
                    
                else:
                    if obs_r[i] == 0.:
                        obs_r[i] = np.clip(norm([x_s[i]-x_e[i], y_s[i]-y_e[i]])/abs(2*np.sin(theta_a[i])), 0.1, 5.0)
                    '''
                    if i == 0:
                        x[i] = - np.clip(
                            ((y_s[i] - y_m[i]) * (np.square(d_m[i]) - np.square(d_e[i])) - (y_m[i] - y_e[i]) * (
                                    np.square(d_s[i]) - np.square(d_m[i]))) / (2 * (
                                    (x_m[i] - x_e[i]) * (y_s[i] - y_m[i]) - (x_s[i] - x_m[i]) * (y_m[i] - y_e[i]))), x_m[0] + obs_r[0], detect_range + obs_r[0])
                        y[i] = - np.clip(((x_s[i] - x_m[i]) * (np.square(d_m[i]) - np.square(d_e[i])) - (x_m[i] - x_e[i]) * (
                                    np.square(d_s[i]) - np.square(d_m[i]))) / (2 * (
                                    (x_s[i] - x_m[i]) * (y_m[i] - y_e[i]) - (x_m[i] - x_e[i]) * (y_s[i] - y_m[i]))), y_m[0] + obs_r[0], detect_range + obs_r[0])
                    else:
                        x[i] = - np.clip(
                            ((y_s[i] - y_m[i]) * (np.square(d_m[i]) - np.square(d_e[i])) - (y_m[i] - y_e[i]) * (
                                    np.square(d_s[i]) - np.square(d_m[i]))) / (2 * (
                                    (x_m[i] - x_e[i]) * (y_s[i] - y_m[i]) - (x_s[i] - x_m[i]) * (y_m[i] - y_e[i]))), x_m[1] + obs_r[1], detect_range + obs_r[1])
                        y[i] = - np.clip(((x_s[i] - x_m[i]) * (np.square(d_m[i]) - np.square(d_e[i])) - (
                                    x_m[i] - x_e[i]) * (np.square(d_s[i]) - np.square(d_m[i]))) / (2 * (
                                    (x_s[i] - x_m[i]) * (y_m[i] - y_e[i]) - (x_m[i] - x_e[i]) * (y_s[i] - y_m[i]))), - detect_range - obs_r[1],  y_m[1] - obs_r[1])
                    '''
                    if i == 0:
                        x[i] = -(x_m[i] + obs_r[i] * np.cos(ray[left_min][1]))
                        y[i] = -(y_m[i] + obs_r[i] * np.sin(ray[left_min][1]))
                    else:
                        x[i] = -(x_m[i] + obs_r[i] * np.cos(ray[right_min][1]))
                        y[i] = -(y_m[i] + obs_r[i] * np.sin(ray[right_min][1]))
        return x, y, obs_r, avoid, [left_start, right_start], [left_end, right_end], [left_min, right_min]


    def get_stream_value(self, agent):
        agent.start_ray = [-1, -1]
        agent.end_ray = [-1, -1]
        agent.min_ray = [0, len(agent.ray)//2]
        err = 0.
        obs_r = np.copy(agent.obs_r)
        x, y, obs_r, avoid, start_ray, end_ray, min_ray = self.get_virtual_obstacle_pos(agent.ray, obs_r)

        # check avoiding state
        if not agent.avoiding[0]:
            #print("before", obs_r[0], obs_r)
            agent.obs_r[0] = 0.
            # check if obstacle detected at left side
            if avoid[0]:
                agent.U[0] = 1 # norm(agent.state.p_vel)/100
                agent.C_bound[0] = - agent.U[0] * (3 * agent.size / 100 + obs_r[0]) * (
                            1 - np.square(obs_r[0]) / ((3 * agent.size / 100 + obs_r[0]) ** 2))

                agent.C[0] = min(agent.U[0] * y[0] * (1 - np.square(obs_r[0]) / max((x[0] ** 2 + y[0] ** 2), np.square(obs_r[0]))), agent.C_bound[0])
                agent.obs_r[0] = obs_r[0]

        if not agent.avoiding[1]:
            #print("before",obs_r[1], obs_r)
            agent.obs_r[1] = 0.
            # check if obstacle detected at right side
            if avoid[1]:
                agent.U[1] = 1 # norm(agent.state.p_vel)/100
                agent.C_bound[1] = agent.U[1] * (3 * agent.size / 100 + obs_r[1]) * (
                            1 - np.square(obs_r[1]) / ((3 * agent.size / 100 + obs_r[1]) ** 2))

                agent.C[1] = max(agent.U[1] * y[1] * (1 - np.square(obs_r[1]) / max((x[1] ** 2 + y[1] ** 2), np.square(obs_r[1]))), agent.C_bound[1])
                agent.obs_r[1] = obs_r[1]

        agent.avoiding = np.copy(avoid)
        agent.start_ray = np.copy(start_ray)
        agent.end_ray = np.copy(end_ray)
        agent.min_ray = np.copy(min_ray)
        # check if left and right sensors sensed same object
        obs_dis = np.clip(norm([x, y], axis=0) - obs_r, agent.size/100, None)
        obs_ang = np.arctan2(np.array(y), np.array(x))
        for ang in obs_ang:
            ang = self.wrap2pi(ang)
        #print(x, y, agent.ray[min_ray, 0], obs_dis)
        # choose the shorter side to follow the streamline
        if agent.avoiding[0] and agent.avoiding[1] and agent.start_ray[0] < 3 and agent.start_ray[1] - len(agent.ray)//2 < 3:
            if obs_r[0] == 0.:
                agent.avoiding[0] = False
            if obs_r[1] == 0.:
                agent.avoiding[1] = False
            if obs_dis[1] < obs_dis[0]:
                agent.avoiding[0] = False
            elif obs_dis[0] < obs_dis[1]:
                agent.avoiding[1] = False
        
        # calculate streamline following error
        if agent.avoiding[0]:
            c = agent.U[0] * y[0] * (1 - np.square(agent.obs_r[0]) / max(np.square(obs_dis[0] + agent.obs_r[0]), np.square(agent.obs_r[0])))
            #print(c, agent.C[0], agent.C_bound[0])
            #err += np.sqrt(np.square(c - agent.C[0]))
            err += np.sqrt(np.square(c - agent.C[0])) * np.clip(1/max(obs_dis[0], agent.size/100) - 1/(5*agent.size/100), 0, None)
            if start_ray[0] < agent.start_ray[0] or obs_r[0] > agent.obs_r[0]:
                # new obstacle detected
                agent.U[0] = 1 # norm(agent.state.p_vel)/100
                agent.C_bound[0] = - agent.U[0] * (3 * agent.size / 100 + obs_r[0]) * (
                            1 - np.square(obs_r[0]) / ((3 * agent.size / 100 + obs_r[0]) ** 2))
                agent.C[0] = min(c, agent.C_bound[0])
                agent.obs_r[0] = obs_r[0]
            agent.obs_dis[0] = obs_dis[0]
            agent.obs_ang[0] = obs_ang[0]

        if agent.avoiding[1]:
            c = agent.U[1] * y[1] * (1 - np.square(agent.obs_r[1]) / max(np.square(obs_dis[1] + agent.obs_r[1]), np.square(agent.obs_r[1])))
            #print(c, agent.C[1], agent.C_bound[1])
            #err += np.sqrt(np.square(c - agent.C[1]))
            err += np.sqrt(np.square(c - agent.C[1])) * np.clip(1/max(obs_dis[1], agent.size/100) - 1/(5*agent.size/100), 0, None)
            if start_ray[1] < agent.start_ray[1] or obs_r[1] > agent.obs_r[1]:
                # new obstacle detected
                agent.U[1] = 1 # norm(agent.state.p_vel)/100
                agent.C_bound[1] = agent.U[1] * (3 * agent.size / 100 + obs_r[1]) * (
                            1 - np.square(obs_r[1]) / ((3 * agent.size / 100 + obs_r[1]) ** 2))
                agent.C[1] = max(c, agent.C_bound[1])
                agent.obs_r[1] = obs_r[1]
            agent.obs_dis[1] = obs_dis[1]
            agent.obs_ang[1] = obs_ang[1]
        #print(avoid, agent.avoiding, agent.obs_dis, agent.obs_ang)
        #print("error: ", err, "obs dis: ", obs_dis)
        return err


    def agent_reward(self, agent, world):
        # 20190711
        # Agents are rewarded based on
        # 1.whether any collision happened
        # 2.the distance between formation center and desire path
        # 3.how the shape of formation is maintained

        # step reward
        #rew = -0.3
        rew = 0

        # parameters for collision reward
        alpha = 1
        beta = 3
        # get the angle of target position
        target_ang = self.wrap2pi(np.arctan2(agent.err[1], agent.err[0]))
        leader_ang = self.get_relAngle(agent, world.agents[0])
        agents = self.follower_agents(world) + self.leader_agents(world)
        static_obs = self.get_static_obstacles(world)
        dynamic_obs = self.get_dynamic_obstacles(world)
        obs = agents + static_obs + dynamic_obs
        # penalty of collision with agents
        '''
        for i, o in enumerate(obs):
            if o != agent:
                if self.is_collision(agent, o):
                    if not agent.collide:
                        agent.crash += 1
                        agent.collide = True
                    break
            if i == len(obs) - 1:
                agent.collide = False
        '''
        # min of all ray's detect result
        min_obs_dis = np.clip(np.min(np.array(agent.obs_dis)), 0, None)
        min_agt_dis = 50 * agent.size / 100
        for entity in agents:
            if entity != agent:
                dis2agt = np.array([min(norm(entity.state.p_pos - agent.state.p_pos) / 100, 50 * agent.size / 100)])
                if dis2agt < min_agt_dis:
                    min_agt_dis = dis2agt
        #rew -= alpha * math.exp(-beta * min(agt_dis))
        min_dis = min(min_obs_dis, np.min(min_agt_dis))
        formation_err = norm(agent.err)
        '''
        # only when the desire formation position is in front of the agent (0~pi) should it consider the stream line following
        if abs(self.wrap2pi(agent.state.p_ang - world.agents[0].state.p_ang - target_ang)) > np.pi/4:
            if min_dis < 1.5 * agent.size / 100:
                rew -= alpha * (1 / min_dis - 1 / (1.5 * agent.size / 100))
        else:
            rew -= alpha * agent.stream_err
        
        if formation_err < 5 or abs(self.wrap2pi(agent.state.p_ang - world.agents[0].state.p_ang - target_ang)) < np.pi/4:
            rew -= alpha * agent.stream_err
        else:
            if min_dis < 3 * agent.size / 100:
                rew -= alpha * (1 / max(min_dis, agent.size / 100) - 1 / (3 * agent.size / 100))
        '''
        #rew -= alpha * agent.stream_err
        # reward for formation
        c = 1
        #rew -= c*(50*(agent.err - agent.err_prev))
        #rew -= c*formation_err
        #print(rew)
        return -formation_err, -agent.stream_err


    def dis2goal_reward(self, agent, world):
        # Agents are rewarded based on the distance between itself and the navigation goal
        # TODO: to change the reward alpha and beta
        # TODO: 相对定位坐标
        import pdb
        pdb.set_trace()
        rew = 0
        nav_rew_weight = self.args.nav_rew_weight
        avoid_rew_weight = self.args.avoid_rew_weight
        form_reward_weight = self.args.form_rew_weight
        dist_rew_weight = self.args.dist_rew_weight

        dis2goal = norm(agent.state.p_pos - world.landmarks[0].state.p_pos) / 100 # cm->m
        navigation_reward = - nav_rew_weight * (dis2goal - agent.dis2goal_prev)
        #navigation_reward = 0
        avoidance_reward = - avoid_rew_weight * self.collide_this_time

        # if avoidance_reward != 0:
        #     print('Ops, collision exists. avoidance_reward = ' + str(avoidance_reward))
        
        all_agents = world.agents
        agent_index = world.agents.index(agent)
        neighbor_two_agent = [world.agents[agent_index-1], world.agents[((agent_index+1) % len(all_agents))]]
        dis_with_two_neighbor = [norm(agent.state.p_pos - neighbor_two_agent[0].state.p_pos), norm(agent.state.p_pos - neighbor_two_agent[1].state.p_pos)]

        neighbor_dist_reward = - dist_rew_weight * np.abs(dis_with_two_neighbor[0] + dis_with_two_neighbor[1] - 2 * self.args.ideal_side_len)

        pos_rel = [[],[]] # real relative position

        for any_agent in world.agents:
            pos_rel[0].append(any_agent.state.p_pos[0] - agent.state.p_pos[0])
            pos_rel[1].append(any_agent.state.p_pos[1] - agent.state.p_pos[1])
        
        topo_err = MDS.error_rel_g(np.array(world.ideal_topo_point), np.array(pos_rel), len(world.agents))
        formation_reward = - form_reward_weight * topo_err
        # ideal topo: [-15,0] [0,15] [15,0] [0,-15]
        # pos_rel = [0,0] [9,6] [-15,4] [12,7]
        #formation_reward = 

        rew = navigation_reward + avoidance_reward + formation_reward + neighbor_dist_reward
       
        return rew


    def leader_reward(self, agent, world):
        # Agent are rewarded based on minimum agent distance to landmark
        rew = -0.3
        agents = self.follower_agents(world)
        leader = agent
        static_obs = self.get_static_obstacles(world)
        dynamic_obs = self.get_dynamic_obstacles(world)
        obs = static_obs + dynamic_obs
        alpha = 5
        beta = 2

        for o in obs:
            if self.is_collision(agent, o):
                #rew -= 5
                pass
        min_obs_dis = agent.ray[agent.min_ray, 0]
        min_agt_dis = 50 * agent.size / 100
        for entity in agents:
            if entity != agent:
                dis2agt = np.array([min(norm(entity.state.p_pos - agent.state.p_pos) / 100, 50 * agent.size / 100)])
                if dis2agt < min_agt_dis:
                    min_agt_dis = dis2agt
        rew -= alpha * math.exp(-beta * min(min(min_obs_dis), min_agt_dis))
        # leader's task is to navigate toward the goal
        sigma = 100
        rew -= sigma*(leader.dis2goal - leader.dis2goal_prev)
        return rew, 0.

    def kalman_filter(self, world, agent):
        '''
        kalman filter for followers to predict leader trajectory
        leader_state = [pos_x,pos_y,vel_x,vel_y]
        follower_obs = [pos_x,pos_y]
        his_size = 10
        :return: predicted leader state
        '''
        x = agent.dis2leader*np.cos(agent.ang2leader)
        y = agent.dis2leader*np.sin(agent.ang2leader)
        v_x = 0
        v_y = 0
        if len(agent.his):
            v_x = x - agent.his[-1][2]
            v_y = y - agent.his[-1][3]
        state_cur = np.array([x,y,v_x,v_y]).T
        agent.his.append(state_cur)
        det = world.dt
        Z = np.array([x,y]).T
        F = np.array([[1,0,det,0],[0,1,0,det],[0,0,1,0],[0,0,0,1]])
        H = np.array([[1,0,0,0],[0,1,0,0]])
        R = 0
        if len(agent.his):
            R = np.cov(np.array(agent.his).T)

        state_pre = F*state_cur
        P_pre = F*agent.P*F.T
        K = P_pre*H.T*np.invert(H*P_pre*H.T+R)
        state_new = state_pre + K*(Z-H*state_pre)
        agent.P = (np.eye(4)-K*H)*P_pre
        return state_new

    def observation2(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)

    def observation(self, agent, world):
        # update desire center position
        self.outside_boundary(agent)
        p_pos = agent.state.p_pos


        if agent.leader:
            agent.dis2goal_prev = agent.dis2goal
            agent.dis2goal = norm(p_pos - world.landmarks[0].state.p_pos) / 100
            agent.ang2goal = self.get_relAngle(agent, world.landmarks[0])
        else:
            # leader = self.leader_agents(world) # None
            # agent.dis2leader_prev = None
            # agent.dis2leader = None
            # agent.ang2leader = None
            # agent.p_rel = None
            agent.dis2goal_prev = agent.dis2goal
            agent.dis2goal = norm(p_pos - world.landmarks[0].state.p_pos) / 100
            agent.ang2goal = self.get_relAngle(agent, world.landmarks[0])
        # get distance and relative angle of all entities in this agent's reference frame
        agt_dis = []
        agt_ang = []
        obs_dis = []
        obs_ang = []
        obs_size = []
        obs_type = []
        self.collide_this_time = 0
        collide = []
        obs = world.agents + world.static_obs + world.dynamic_obs
        for entity in obs:
            if entity == agent:
                continue
            collide.append(self.is_collision(agent, entity))
            dis2obs = (norm(entity.state.p_pos - p_pos) - entity.size - agent.size)/100
            ang = self.get_relAngle(agent, entity)
            if 'agent' in entity.name and entity != agent:
                # TODO: 这个地方怎么设计比较好（目前的做法是设计了一个最大通信距离）
                dis2agt = np.array([min(norm(entity.state.p_pos - p_pos)/100, 50*agent.size/100)])
                ang = self.get_relAngle(agent, entity)
                agt_dis.append(dis2agt)
                agt_ang.append(np.array([ang]))
            if dis2obs < 25*agent.size/100 and entity != agent:
                if 'agent' in entity.name and entity.leader:
                    continue	
                obs_dis.append(dis2obs)
                obs_ang.append(ang)
                obs_size.append(entity.size/100)
                obs_type.append(entity.name)
        # change the order of the observation distance and angle
        current_agent_num = int(agent.name[-1])
        agt_dis = agt_dis[current_agent_num:] + agt_dis[:current_agent_num]
        agt_ang = agt_ang[current_agent_num:] + agt_ang[:current_agent_num]
        if True in collide:
            agent.crash += 1
        self.collide_this_time += np.sum(collide)

        target_dis = [np.array([norm(world.landmarks[0].state.p_pos - p_pos) / 1000])] 
        target_ang = [np.array([self.get_relAngle(agent, world.landmarks[0])])]
        self.set_agent_ray(agent, obs_dis, obs_ang, obs_size)
        #sensor_ray = [np.array([agent.ray[i][j]]) for i in range(len(agent.ray)) for j in range(len(agent.ray[i]))]
        if not agent.leader:
            agent.stream_err = self.get_stream_value(agent)

        start_ray = [np.array([agent.ray[agent.start_ray[i]][j]]) if agent.start_ray[i] >= 0 else np.array([agent.ray[agent.min_ray[i]][j]]) for i in range(len(agent.start_ray)) for j in range(len(agent.ray[i]))]
        end_ray = [np.array([agent.ray[agent.end_ray[i]][j]]) if agent.end_ray[i] >= 0 else np.array([agent.ray[agent.min_ray[i]][j]]) for i in range(len(agent.end_ray)) for j in range(len(agent.ray[i]))]
        min_ray = [np.array([agent.ray[agent.min_ray[i]][j]]) for i in range(len(agent.min_ray)) for j in range(len(agent.ray[i]))]

        obs_dis = [np.array([agent.obs_dis[i]]) if agent.avoiding[i] else np.array([2.0]) for i in range(len(agent.obs_dis))]
        obs_ang = [np.array([agent.obs_ang[i]]) if agent.avoiding[i] else np.array([0.0]) for i in range(len(agent.obs_ang))]
        obs_r = [np.array([agent.obs_r[i]]) if agent.avoiding[i] else np.array([0.0]) for i in range(len(agent.obs_r))]

        # communication of all other agents, now assume the communication graph is fully connected
        formation_pos_x = []
        formation_pos_y = []
        formation_pos_x.append(p_pos[0])
        formation_pos_y.append(p_pos[1])
        comm = [world.agents[0].state.c]
        for other in world.agents:
            if other is agent: continue
            formation_pos_x.append(other.state.p_pos[0])
            formation_pos_y.append(other.state.p_pos[1])
            agent.agents_ctr = np.array([np.mean(formation_pos_x), np.mean(formation_pos_y)])

        vel = []
        omg = []
        # ang = []
        # if not agent.leader:
        #     ang.append(np.array([self.wrap2pi(agent.ang2leader-agent.w_des) / np.pi]))
        # else:
        #     ang.append(np.array([agent.ang2goal / np.pi]))
        vel.append(np.array([norm(agent.state.p_vel)/100]))
        omg.append(np.array([agent.state.p_omg / np.pi]))
        err = [np.array([agent.err[i]]) for i in range(len(agent.err))] 

        #return np.concatenate([np.array([agent.dis2leader - agent.d_des])] + ang + vel + omg + agt_dis + agt_ang + sensor_ray)
        #return np.concatenate(err + vel + omg + agt_dis + agt_ang + start_ray + end_ray + min_ray + obs_dis + obs_ang + obs_r + target_dis + target_ang)
        # TODO: target_dis can be represented as d_cur - d_pre?
        #return np.concatenate(agt_dis + agt_ang + start_ray + end_ray + min_ray + obs_dis + obs_ang + obs_r + target_dis + target_ang)
        return np.concatenate(agt_dis + agt_ang + min_ray)

    def constraint(self, agent, world):
        return []

    def done(self, agent, world):
        #return not np.any(world.err > 2e-2)
        return np.min([world.agents[i].dis2goal for i in range(len(world.agents))]) < 5e-1