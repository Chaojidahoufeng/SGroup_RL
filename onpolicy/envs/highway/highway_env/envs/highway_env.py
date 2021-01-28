import numpy as np
from typing import Tuple
from gym.envs.registration import register

from onpolicy.envs.highway.highway_env import utils
from onpolicy.envs.highway.highway_env.envs.common.abstract import AbstractEnv
from onpolicy.envs.highway.highway_env.envs.common.action import Action
from onpolicy.envs.highway.highway_env.road.road import Road, RoadNetwork
from onpolicy.envs.highway.highway_env.vehicle.controller import ControlledVehicle

import random

class HighwayEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """
    RIGHT_LANE_REWARD: float = 0.1
    """The reward received when driving on the right-most lanes, linearly mapped to zero for other lanes."""

    HIGH_SPEED_REWARD: float = 0.9
    """The reward received when driving at full speed, linearly mapped to zero for lower speeds according to config["reward_speed_range"]."""

    LANE_CHANGE_REWARD: float = 0
    """The reward received at each lane change action."""

    def default_config(self) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 1,
            "collision_reward": -1,  # The reward received when colliding with a vehicle.
            "reward_speed_range": [20, 30],
            "offroad_terminal": False
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()
        ######## Jianming Jan 29 New feature -> bubble test: control handover
        # self.ctrl_change_vehicle_num = True
        # self.num_test = 0
        ########

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.straight_road_network(self.config["lanes_count"]),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        # the number of agent with initialized postions overlapping with each other
        self.controlled_vehicles = []
        number_overlap = 0
        for i in range(self.config["controlled_vehicles"]):
            # vehicle = self.action_type.vehicle_class.create_random(self.road,
            #                                                        speed=25,
            #                                                        lane_id=self.config["initial_lane_id"],
            #                                                        spacing=self.config["ego_spacing"],
            #                                                        )
            default_spacing = 12.5 # 0.5 * speed
            longitude_position = 40+5*np.random.randint(1)
            initial_lane_idx = random.choice( [4*i for i in range(self.config["lanes_count"])] )
            # To separate cars in different places to avoid collision
            for vehicle_ in self.controlled_vehicles:
                if abs(longitude_position - vehicle_.position[0]) < 5 and initial_lane_idx == 4*vehicle_.lane_index[2]:
                    longitude_position = longitude_position - (number_overlap+1)*default_spacing
                    number_overlap = number_overlap + 1
            vehicle = self.action_type.vehicle_class(road=self.road, position=[longitude_position, initial_lane_idx], heading=0, speed=25)
            
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)


        vehicles_type = utils.class_from_path(self.config["npc_vehicles_type"])
        for _ in range(self.config["vehicles_count"]):
            vehicle = vehicles_type.create_random(self.road, spacing=1 / self.config["vehicles_density"])
            self.road.vehicles.append(vehicle)
            self.controlled_vehicles.append(vehicle)
            # observation size depents on the firstly controlled_vehicles in the initilization process
            # but after defined the observation type doesn't change.

    def _reward(self, action: Action) :#-> float: now we return a list
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """

        # -> float: now we change it to return a list!!!!!

        ######## Jianming Jan 29 New feature -> bubble test: control handover
        # change the control method between action level and continous steering-acceleration level.
        # if self.ctrl_change_vehicle_num:
        #     if self.num_test >= 5:
        #         self.temp_vehicle = self.controlled_vehicles[-1]
        #         self.controlled_vehicles.remove(self.temp_vehicle)
        #         self.temp_vehicle.use_action_level_behavior = False
        #         self.ctrl_change_vehicle_num = False
        #         self.define_spaces()
        #         self.num_test = 0
        #     self.num_test += 1
        # else:
        #     self.temp_vehicle.use_action_level_behavior = True
        #     self.controlled_vehicles.append(self.temp_vehicle)
        #     self.ctrl_change_vehicle_num = True
        #     self.define_spaces()
        ########
            
        rewards=[]
        for vehicle in self.controlled_vehicles:
        
            neighbours = self.road.network.all_side_lanes(vehicle.lane_index)
            lane = vehicle.target_lane_index[2] if isinstance(vehicle, ControlledVehicle) \
                else vehicle.lane_index[2]
            scaled_speed = utils.lmap(vehicle.speed, self.config["reward_speed_range"], [0, 1])
            reward = \
                + self.config["collision_reward"] * vehicle.crashed \
                + self.RIGHT_LANE_REWARD * lane / max(len(neighbours) - 1, 1) \
                + self.HIGH_SPEED_REWARD * np.clip(scaled_speed, 0, 1)
            #reward = utils.lmap(reward,
            #              [self.config["collision_reward"], self.HIGH_SPEED_REWARD + self.RIGHT_LANE_REWARD],
            #              [0, 1])
            #reward = 0 if not vehicle.on_road else reward
            reward = -1 if not vehicle.on_road else reward
            rewards.append(reward)
        return rewards

    def _is_terminal(self) :#-> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        #now we change it to return a list
        dones = []
        for vehicle in self.controlled_vehicles:
            dones.append(vehicle.crashed or \
                         self.steps >= self.config["duration"] or \
                         (self.config["offroad_terminal"] and not vehicle.on_road))

        defender_done = dones[:self.config["n_defenders"]]
        attacker_done = dones[self.config["n_defenders"]:self.config["n_defenders"] + self.config["n_attackers"]]

        if np.all(defender_done):
            for i in range(len(dones)):
                dones[i] = True
        elif len(attacker_done) > 0 and np.all(attacker_done):
            for i in range(len(dones)):
                dones[i] = True
        return dones

    def adv_rew(self) :#-> bool:
        dones = []
        for vehicle in self.controlled_vehicles:
            dones.append(vehicle.crashed or \
                         self.steps >= self.config["duration"] or \
                         (self.config["offroad_terminal"] and not vehicle.on_road))

        defender_done = dones[:self.config["n_defenders"]]
        attacker_done = dones[self.config["n_defenders"]:self.config["n_defenders"] + self.config["n_attackers"]]

        if np.all(defender_done) and (not np.all(attacker_done)):
            return 1
        else:
            return 0

    def _is_done(self) :#-> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        #now we change it to return a list
        dones = []
        for vehicle in self.controlled_vehicles:
            dones.append(vehicle.crashed or \
                         self.steps >= self.config["duration"] or \
                         (self.config["offroad_terminal"] and not vehicle.on_road))

        defender_done = dones[:self.config["n_defenders"]]
        attacker_done = dones[self.config["n_defenders"]:self.config["n_defenders"] + self.config["n_attackers"]]

        if np.all(defender_done):
            return True
        elif len(attacker_done)>0 and np.all(attacker_done):
            return True
        else:
            return False


    def _cost(self, action: int) -> float:
        """The cost signal is the occurrence of collision."""
        return float(self.vehicle.crashed)

    def get_available_actions(self):
        """
        Get the list of currently available actions.

        Lane changes are not available on the boundary of the road, and speed changes are not available at
        maximal or minimal speed.

        :return: the list of available actions
        """
        from onpolicy.envs.highway.highway_env.envs.common.action import  DiscreteMetaAction,MultiAgentAction

        if isinstance(self.action_type, DiscreteMetaAction):
            actions = [self.action_type.actions_indexes['IDLE']]
            for l_index in self.road.network.side_lanes(self.vehicle.lane_index):
                if l_index[2] < self.vehicle.lane_index[2] \
                        and self.road.network.get_lane(l_index).is_reachable_from(self.vehicle.position) \
                        and self.action_type.lateral:
                    actions.append(self.action_type.actions_indexes['LANE_LEFT'])
                if l_index[2] > self.vehicle.lane_index[2] \
                        and self.road.network.get_lane(l_index).is_reachable_from(self.vehicle.position) \
                        and self.action_type.lateral:
                    actions.append(self.action_type.actions_indexes['LANE_RIGHT'])
            if self.vehicle.speed_index < self.vehicle.SPEED_COUNT - 1 and self.action_type.longitudinal:
                actions.append(self.action_type.actions_indexes['FASTER'])
            if self.vehicle.speed_index > 0 and self.action_type.longitudinal:
                actions.append(self.action_type.actions_indexes['SLOWER'])
            return actions

        elif isinstance(self.action_type, MultiAgentAction):
            multi_actions=[]
            for vehicle,action_type in zip(self.controlled_vehicles,self.action_type.agents_action_types):
                actions = [action_type.actions_indexes['IDLE']]
                for l_index in self.road.network.side_lanes(vehicle.lane_index):
                    if l_index[2] < vehicle.lane_index[2] \
                            and self.road.network.get_lane(l_index).is_reachable_from(self.vehicle.position) \
                            and action_type.lateral:
                        actions.append(action_type.actions_indexes['LANE_LEFT'])
                    if l_index[2] > vehicle.lane_index[2] \
                            and self.road.network.get_lane(l_index).is_reachable_from(self.vehicle.position) \
                            and action_type.lateral:
                        actions.append(action_type.actions_indexes['LANE_RIGHT'])
                if vehicle.speed_index < vehicle.SPEED_COUNT - 1 and action_type.longitudinal:
                    actions.append(action_type.actions_indexes['FASTER'])
                if vehicle.speed_index > 0 and action_type.longitudinal:
                    actions.append(action_type.actions_indexes['SLOWER'])
                multi_actions.append(actions)
            return multi_actions

register(
    id='highway-v0',
    entry_point='onpolicy.envs.highway.highway_env.envs:HighwayEnv',
)
