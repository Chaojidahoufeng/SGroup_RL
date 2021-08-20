import numpy as np
import math as math
# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        #self.p_pos_prev = None
        # physical velocity
        self.p_vel = None
        # physical angle
        self.p_ang = None
        #self.p_ang_prev = None
        # physical angular velocity
        self.p_omg = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 5
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 5

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()


class Static_obs(Entity):
    def __init__(self):
        super(Static_obs, self).__init__()
        self.initial_mass = 50

class Dynamic_obs(Entity):
    def __init__(self):
        super(Dynamic_obs, self).__init__()
        self.movable = True
        self.initial_mass = 50
        self.action = Action()

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # formation center
        self.agents_ctr = np.array([0, 0])  # 20190711
        self.agents_ctr_prev = [] # 20190711
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        self.err = []
        self.err_prev = []
        # constraint space
        self.constraint = []

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        self.static_obs = []
        self.dynamic_obs = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1
        self.contact_margin = 1e-6
        # arguments for goal tracking
        self.path = []
        self.start = []
        self.goal = []
        self.station = []
        self.station_num = []
        self.time = 0
        self.distance = 0
        self.flag = 0.5

        self.lam = None
    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks + self.static_obs + self.dynamic_obs

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        self.time += self.dt
        #np.random.seed(int(self.time*10))
        dis = np.linalg.norm(self.agents[0].state.p_vel)*self.dt
        #self.landmarks[0].state.p_pos[0] += dis_x
        #self.landmarks[0].state.p_pos[1] += dis_y
        self.distance += dis / 100  # transfer to meter
        for obs in self.static_obs:
            if abs(self.agents[0].state.p_pos[0] - obs.state.p_pos[0]) >= 600:
                obs.state.p_pos[0] += np.sign(self.agents[0].state.p_pos[0] - obs.state.p_pos[0])*1200#np.random.uniform(1200, 1500)
                obs.state.p_pos[1] = self.agents[0].state.p_pos[1] + 45 * np.random.randint(-10, 10) # training 45, testing 60
                obs.size = np.random.uniform(10.0, 50.0)
            elif abs(self.agents[0].state.p_pos[1] - obs.state.p_pos[1]) >= 600:
                obs.state.p_pos[0] = self.agents[0].state.p_pos[0] + 45 * np.random.randint(-10, 10)
                obs.state.p_pos[1] += np.sign(self.agents[0].state.p_pos[1] - obs.state.p_pos[1])*1200#np.random.uniform(1200, 1500)
                obs.size = np.random.uniform(10.0, 50.0)

        for obs in self.dynamic_obs:
            obs.state.p_pos += obs.state.p_vel
            if abs(self.agents[0].state.p_pos[0] - obs.state.p_pos[0]) >= 500:
                obs.state.p_vel[0] *= -1
            if abs(self.agents[0].state.p_pos[1] - obs.state.p_pos[1]) >= 700:
                obs.state.p_pos[1] += np.sign(self.agents[0].state.p_pos[1] - obs.state.p_pos[1])*1200

        for agent in self.agents:
            if not agent.leader:
                agent.err_prev = agent.err
                #agent.err = np.linalg.norm(agent.p_des-agent.p_rel)
                agent.err = agent.p_des - agent.p_rel

        #np.random.seed()
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # apply environment forces
        self.apply_environment_force(p_force)
        #print(p_force)

        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i,agent in enumerate(self.agents):
            if agent.movable:
                if any(agent.action.u):
                    noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                else:
                    noise = 0.0
                p_force[i] = agent.action.u + noise

        for i in range(len(self.agents)+len(self.landmarks)+len(self.static_obs), len(self.agents)+len(self.landmarks)+len(self.static_obs)+len(self.dynamic_obs)):
            if self.entities[i].movable:
                p_force[i] = self.entities[i].action.u

        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a,entity_a in enumerate(self.entities):
            for b,entity_b in enumerate(self.entities):
                if(b <= a): continue
                #if entity_a not in self.agents: continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)

                if(f_a is not None):
                    entity_a.state.p_vel += 100*f_a
                    entity_a.state.p_pos += 100*f_a*self.dt


    # integrate physical state
    def integrate_state(self, p_force):
        for i,entity in enumerate(self.entities):
            if not entity.movable: continue
            if (p_force[i] is not None):
                #entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
                # 20190717 modify the control of agents
                # entity.state.p_omg = np.pi*p_force[i][0] / (800*entity.mass)
                # entity.state.p_vel[0] += (p_force[i][1] * np.cos(entity.state.p_ang) / entity.mass) * self.dt
                # entity.state.p_vel[1] += (p_force[i][1] * np.sin(entity.state.p_ang) / entity.mass) * self.dt
                entity.state.p_omg = p_force[i][0]
                entity.state.p_vel[0] = 100*(p_force[i][1] * np.cos(entity.state.p_ang))
                entity.state.p_vel[1] = 100*(p_force[i][1] * np.sin(entity.state.p_ang))
                #print(entity.name, entity.state.p_omg, entity.state.p_vel)
            speed = 0
            if entity.max_speed is not None:
                speed = np.linalg.norm(entity.state.p_vel)
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.linalg.norm(entity.state.p_vel) * entity.max_speed
            entity.state.p_pos_prev = entity.state.p_pos
            entity.state.p_ang_prev = entity.state.p_ang

            if abs(entity.state.p_omg) > 1e-6:
                entity.state.p_pos[1] += speed / entity.state.p_omg * (
                            np.cos(entity.state.p_ang) - np.cos(entity.state.p_ang + self.dt * entity.state.p_omg))
                entity.state.p_pos[0] += speed / entity.state.p_omg * (
                            np.sin(entity.state.p_ang + self.dt * entity.state.p_omg) - np.sin(entity.state.p_ang))
                entity.state.p_ang -= entity.state.p_omg * self.dt
            else:
                entity.state.p_pos[1] += speed * np.sin(entity.state.p_ang) * self.dt
                entity.state.p_pos[0] += speed * np.cos(entity.state.p_ang) * self.dt

            # entity.state.p_pos += entity.state.p_vel * self.dt
            # entity.state.p_ang -= entity.state.p_omg * self.dt
            if abs(entity.state.p_ang) >= np.pi:
                entity.state.p_ang -= np.sign(entity.state.p_ang) * 2 * np.pi

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise


    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        # compute actual distance between entities
        delta_pos = (entity_a.state.p_pos - entity_b.state.p_pos)/100
        if (entity_a.state.p_pos[0] - entity_b.state.p_pos[0]) != 0:
            ang = math.atan((entity_a.state.p_pos[1] - entity_b.state.p_pos[1])/(entity_a.state.p_pos[0] - entity_b.state.p_pos[0]))
        else:
            ang = np.pi/2
        dist = np.linalg.norm(delta_pos) - entity_a.size/100 - entity_b.size/100
        # minimum allowable distance
        dist_min = (entity_a.size + entity_b.size)*math.cos(abs(ang)-np.pi/4)/100
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min + 0.001)/k)*k

        '''force = self.contact_force * delta_pos / dist * penetration

        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        print([force_a, force_b])'''
        mass_a = entity_a.mass
        mass_b = entity_b.mass

        '''if entity_a.movable and not entity_b.movable:
            force_a = self.contact_force * entity_a.mass * -abs(entity_a.state.p_vel) * penetration / self.dt
        elif entity_b.movable and not entity_a.movable:
            force_b = self.contact_force * entity_b.mass * -abs(entity_b.state.p_vel) * penetration / self.dt'''
        force_a = np.zeros(self.dim_p)
        force_b = np.zeros(self.dim_p)

        if dist < 0 or abs(dist) < self.contact_margin:
            if entity_b.movable:
                force_a = self.contact_force * (((
                                                             mass_a - mass_b) * entity_a.state.p_vel/100 + 2 * mass_b * entity_b.state.p_vel/100) / (
                                                            mass_a + mass_b)) if entity_a.movable else None
            else:
                force_a = 10 * self.contact_force * (((
                                                         mass_a - mass_b) * entity_a.state.p_vel / 100 + 2 * mass_b * entity_b.state.p_vel / 100) / (
                                                        mass_a + mass_b)) if entity_a.movable else None

        return [force_a, force_b]
