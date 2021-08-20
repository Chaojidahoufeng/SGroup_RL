import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from .multi_discrete import MultiDiscrete

import pyglet
from pyglet import gl

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, constraint_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True, discrete_action=False):

        self.world = world
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.constraint_callback = constraint_callback #2020/02/20
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.constraint_space = []
        share_obs_dim = 0
        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(3) # foward, left right steer
            else:
                u_action_space = spaces.Box(low=0, high=+agent.u_range, shape=(3,), dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
            if not agent.silent:
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)
            share_obs_dim += obs_dim
        self.share_observation_space = [spaces.Box(
            low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32)] * self.n
        #2020/02/20 add constraint space
        con_dim = len(constraint_callback(agent, self.world))
        self.constraint_space.append(spaces.Box(low=0, high=+np.inf, shape=(con_dim,), dtype=np.float32))

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def step(self, action_n, constraint_n=None):
        obs_n = []
        formation_reward_n = []
        avoidance_reward_n = []
        done_n = []
        info_n = {'n': []}
        crash_n = []
        detected_n = []
        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
            if constraint_n:
                agent.constraint = constraint_n[i]

        # advance world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            formation_rew, avoidance_rew = self._get_reward(agent)
            formation_reward_n.append(formation_rew)
            avoidance_reward_n.append(avoidance_rew)
            done_n.append(self._get_done(agent))
            info_n['n'].append(self._get_info(agent))
            crash_n.append(agent.crash)
            detected_n.append(any(agent.avoiding))
        '''
        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.n
        '''
        return obs_n, formation_reward_n, avoidance_reward_n,  done_n, info_n, crash_n, detected_n

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        agent.action.u[1] = 0.
        agent.agents_ctr_prev = agent.agents_ctr # record previous state
        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index+s)])
                index += s
            action = act
        else:
            action = [action]

        #print(action)
        #action = [action]


        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1: agent.action.u[0] = -1.0
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
            else:
                if agent.leader:
                    agent.action.u[0] += 0.0  # omega
                    #agent.action.u[0] += 0.3*(action[0][0])  # omega
                    agent.action.u[1] += 0.35
                    #agent.action.u[1] += 0.35*(action[0][1])
                else:
                    agent.action.u[0] += action[0][0] - action[0][1]  # omega
                    #agent.action.u[0] += action[0][0]  # omega
                    agent.action.u[1] += 0.5 * (action[0][2])
                    #agent.action.u[1] += 0.5 * (action[0][1])
            #print(agent.action.u)
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    # get constraint bound for a particular agent
    def _get_constraint(self, agent):
        if self.constraint_callback is None:
            return False
        return self.constraint_callback(agent, self.world)

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, mode='human'):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            #print(message)
        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            WINDOW_W = 700
            WINDOW_H = 700
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(WINDOW_W, WINDOW_H)

        # create rendering geometry
        #if self.render_geoms is None:
        # import rendering only if we need it (and don't import for headless machines)
        #from gym.envs.classic_control import rendering
        from multiagent import rendering

        self.agents_geoms = []
        self.agents_geoms_xform = []
        self.render_geoms = []
        self.render_geoms_xform = []
        entities_rearrange = self.world.entities
        agents = []
        for entity in self.world.entities:
            if 'agent' in entity.name:
                agents.append(entity)
                entities_rearrange.pop(0)
        entities_rearrange += agents
        for entity in entities_rearrange:
            if 'agent' in entity.name:
                geom = rendering.make_square(entity.size, angle=entity.state.p_ang+np.pi/4)
            elif 'landmark' in entity.name:
                if entity.center:
                    geom = rendering.make_circle(entity.size)
                else:
                    geom = rendering.make_circle(entity.size)
            else:
                geom = rendering.make_circle(entity.size)
            geom.set_color(*entity.color)
            xform = rendering.Transform()
            geom.add_attr(xform)
            self.render_geoms.append(geom)
            self.render_geoms_xform.append(xform)

        import copy
        agents_end = copy.deepcopy(agents)
        agents_end.pop(0)
        agents_begin = copy.deepcopy(agents)
        agents_begin.pop(0)
        agents_begin.insert(0, agents_begin.pop())
        lines = []
        lines_length_text = []
        for i in range(len(agents_begin)):
            line = rendering.Line((agents_begin[i].state.p_pos[0], agents_begin[i].state.p_pos[1]),
                                  (agents_end[i].state.p_pos[0], agents_end[i].state.p_pos[1]))
            lines.append(line)
            agent_begin_name = agents_begin[i].name
            agent_end_name = agents_end[i].name
            tmp_text = agent_begin_name + ' to ' + agent_end_name + ' : '
            length = np.linalg.norm(agents_begin[i].state.p_pos - agents_end[i].state.p_pos)
            tmp_text = tmp_text + str(length)
            lines_length_text.append(tmp_text)

        # add geoms to viewer
        for viewer in self.viewers:
            viewer.geoms = []
            viewer.labels = []

            for geom in self.render_geoms:
                viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from multiagent import rendering
            # update bounds to center around agent
            cam_range = 500
            '''if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos'''
            pos = self.world.agents[0].state.p_pos
            self.viewers[i].set_bounds(pos[0]-cam_range, pos[0]+cam_range, pos[1]-cam_range, pos[1]+cam_range)
            # update geometry positions
            for e, entity in enumerate(entities_rearrange):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # add goal to render
            goal = []
            goal_xform = []
            goal = rendering.make_circle(5)
            goal.set_color(1.0, 0, 0)
            goal_xform = rendering.Transform()
            goal.add_attr(goal_xform)
            goal_xform.set_translation(*self.world.landmarks[0].state.p_pos)
            self.viewers[i].add_geom(goal)

            goal_ang = self.world.agents[0].state.p_ang + self.world.agents[0].ang2goal
            goal_dir = 20 * np.array([np.cos(goal_ang), np.sin(goal_ang)])
            arrow = rendering.make_triangle(5, angle=goal_ang)
            arrow.set_color(1.0, 0., 0.)
            arrow_xform = rendering.Transform()
            arrow.add_attr(arrow_xform)
            arrow_xform.set_translation(*self.world.agents[0].state.p_pos + goal_dir)
            self.viewers[i].add_geom(arrow)
            # add formation center to render
            ctr = []
            ctr_xform = []
            ctr = rendering.make_circle(1)
            ctr_xform = rendering.Transform()
            ctr.add_attr(ctr_xform)
            ctr_xform.set_translation(*self.agents[0].agents_ctr)
            self.viewers[i].add_geom(ctr)
            # add previous formation center to render
            ctr_prev = []
            ctr_xform_prev = []
            ctr_prev = rendering.make_circle(1)
            ctr_prev.set_color(0.5, 0.5, 0.5)
            ctr_xform_prev = rendering.Transform()
            ctr_prev.add_attr(ctr_xform_prev)
            ctr_xform_prev.set_translation(*self.agents[0].agents_ctr_prev)
            self.viewers[i].add_geom(ctr_prev)

            # add head to agents
            for e, agent in enumerate(self.agents):
                if e > 0:
                    for j in range(agent.start_ray[0], agent.end_ray[0] + 1):
                        # 105 for compensating square's rendering error
                        if 100 * agent.ray[j][0] < 200:
                            ray_pos = 105 * agent.ray[j][0] * np.array(
                                [np.cos(agent.ray[j][1] + agent.state.p_ang), np.sin(agent.ray[j][1] + agent.state.p_ang)])
                            ray = rendering.make_line(agent.state.p_pos, agent.state.p_pos + ray_pos)
                            ray.set_color(1., 0., 0.)
                            ray_xform = rendering.Transform()
                            ray.add_attr(ray_xform)
                            self.viewers[i].add_geom(ray)
                    for j in range(agent.start_ray[1], agent.end_ray[1] + 1):
                        # 105 for compensating square's rendering error
                        if 100 * agent.ray[j][0] < 200:
                            ray_pos = 105 * agent.ray[j][0] * np.array(
                                [np.cos(agent.ray[j][1] + agent.state.p_ang), np.sin(agent.ray[j][1] + agent.state.p_ang)])
                            ray = rendering.make_line(agent.state.p_pos, agent.state.p_pos + ray_pos)
                            ray.set_color(1., 0., 0.)
                            ray_xform = rendering.Transform()
                            ray.add_attr(ray_xform)
                            self.viewers[i].add_geom(ray)
                    err = rendering.make_text(text='error of agent %d = %f meters' % (e, np.linalg.norm(agent.err)), font_size=15,
                                              x=self.world.agents[0].state.p_pos[0] - WINDOW_W // 1.5,
                                              y=self.world.agents[0].state.p_pos[1] - WINDOW_H // 2.0 - 20 * (e + 1),
                                              anchor_x='left',
                                              color=(0, 0, 0, 255))
                    self.viewers[i].add_label(err)
                head = rendering.make_circle(agent.size / 8)
                head_xform = rendering.Transform()
                head.set_color(0.0, .0, 1.0)
                head.add_attr(head_xform)
                displacement = 0.6*agent.size*np.array([np.cos(agent.state.p_ang), np.sin(agent.state.p_ang)])
                head_xform.set_translation(*agent.state.p_pos + displacement)
                self.viewers[i].add_geom(head)
                label = rendering.make_text(text='%d' % e, font_size=12, x=agent.state.p_pos[0], y=agent.state.p_pos[1], color=(0, 0, 0, 255))
                self.viewers[i].add_label(label)
                    
            time = rendering.make_text(text='time = %f sec' % self.world.time, font_size=15,
                                           x=self.world.agents[0].state.p_pos[0] - WINDOW_W // 1.5,
                                           y=self.world.agents[0].state.p_pos[1] - WINDOW_H // 2.0,
                                           anchor_x='left',
                                           color=(0, 0, 0, 255))
            distance = rendering.make_text(text='distance = %f meters' % self.world.distance, font_size=15,
                                           x=self.world.agents[0].state.p_pos[0] - WINDOW_W // 1.5,
                                           y=self.world.agents[0].state.p_pos[1] - WINDOW_H // 2.0-20,
                                           anchor_x='left',
                                           color=(0, 0, 0, 255))
            self.viewers[i].add_label(distance)
            self.viewers[i].add_label(time)
            # render to display or array
            # render to display or array
            for line in lines:
                self.viewers[i].add_geom(line)
            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))


            '''for j in range(len(self.viewers)):
                self.viewers[i].geoms.pop(-1)'''
        #print(pos[0])
        return results

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x,y]))
        return dx


# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i+env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n
