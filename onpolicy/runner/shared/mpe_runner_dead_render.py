    
import time
import wandb
import os
import numpy as np
from itertools import chain
import torch
import imageio

from onpolicy.utils.util import update_linear_schedule
from onpolicy.runner.shared.base_runner import Runner

import pickle
from matplotlib import pyplot as plt 

def _t2n(x):
    return x.detach().cpu().numpy()

class MPERunner(Runner):
    def __init__(self, config):
        self.data_dir = './data/formation_change/'
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        super(MPERunner, self).__init__(config)

    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):

                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)

            # import pdb
            # pdb.set_trace()

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                # import pdb
                # pdb.set_trace()

                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == "MPE":
                    env_infos = {}
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        form_rews = []
                        avoi_rews = []
                        nav_rews = []
                        self_nav_rews = []
                        for info in infos:
                            if 'individual_reward' in info[agent_id].keys():
                                idv_rews.append(info[agent_id]['individual_reward'])
                                agent_k = 'agent%i/individual_rewards' % agent_id
                                env_infos[agent_k] = idv_rews
                            if 'formation_reward' in info[agent_id].keys():
                                form_rews.append(info[agent_id]['formation_reward'])
                                agent_k_form = 'agent%i/formation_reward' % agent_id
                                env_infos[agent_k_form] = form_rews
                            if 'avoidance_reward' in info[agent_id].keys():
                                avoi_rews.append(info[agent_id]['avoidance_reward'])
                                agent_k_avoi = 'agent%i/avoidance_reward' % agent_id
                                env_infos[agent_k_avoi] = avoi_rews
                            if 'navigation_reward' in info[agent_id].keys():
                                nav_rews.append(info[agent_id]['navigation_reward'])
                                agent_k_nav = 'agent%i/navigation_reward' % agent_id
                                env_infos[agent_k_nav] = nav_rews
                            if 'self_navigation_reward' in info[agent_id].keys():
                                self_nav_rews.append(info[agent_id]['self_navigation_reward'])
                                agent_k_selfnav = 'agent%i/self_navigation_reward' % agent_id
                                env_infos[agent_k_selfnav] = self_nav_rews


                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()

        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout() # ??????emodel???val()??????
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                            np.concatenate(self.buffer.obs[step]),
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.masks[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        # rearrange action
        # import pdb
        # pdb.set_trace()
        if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        elif self.envs.action_space[0].__class__.__name__ == 'Box':
            actions_env = actions
        else:
            raise NotImplementedError

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i]+1)[eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
            elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            else:
                raise NotImplementedError

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        envs = self.envs

        formation_rewards = []
        
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            # if self.all_args.render_sight == 'first-person':
            #     if self.all_args.save_gifs:
            #         image = envs.render(mode='rgb_array', sight='first-person')[0][0]
            #         all_frames.append(image)
            #     else:
            #         envs.render(mode='human',sight='first-person')
            # else:
            #     if self.all_args.save_gifs:
            #         image = envs.render(mode='rgb_array', sight='global')[0][0]
            #         all_frames.append(image)
            #     else:
            #         envs.render(mode='human',sight='globals')
            if self.all_args.save_gifs:
                image = envs.render(mode='rgb_array')[0][0]
                all_frames.append(image)
            else:
                envs.render(mode='human')

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            
            episode_rewards = []

            self.episode_length_1 = 150
            self.episode_length_2 = 120
            self.episode_length_3 = 120

            from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
            from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

            # First Stage

            share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]

            self.policy = Policy(self.all_args,
                                self.envs.observation_space[0],
                                share_observation_space,
                                self.envs.action_space[0],
                                device = self.device)

            self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)

            self.model_dir = "/home/yanyz/yanyz/gitlab/onpolicy/onpolicy/scripts/results/MPE/rel_formation_form_error/rmappo/08-26-rel-formation-form-selfnav10-train-mpe-obs0-poly5/run2/models"
            self.restore()

            
            for step in range(self.episode_length_1):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                    np.concatenate(rnn_states),
                                                    np.concatenate(masks),
                                                    deterministic=True)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i]+1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == 'Discrete':
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                elif self.envs.action_space[0].__class__.__name__ == 'Box':
                    actions_env = actions
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                if step != self.episode_length_1 - 1:
                    obs, rewards, dones, infos = envs.step(actions_env)
                    formation_rewards.append(-infos[0][0]['formation_reward']/self.all_args.form_rew_weight)

                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                # if self.all_args.save_gifs:
                #     if self.all_args.render_sight == 'first-person':
                #         image = envs.render(mode='rgb_array', sight='first-person')[0][0]
                #     else:
                #         image = envs.render(mode='rgb_array', sight='global')[0][0]
                #     all_frames.append(image)
                #     calc_end = time.time()
                #     elapsed = calc_end - calc_start
                #     if elapsed < self.all_args.ifi:
                #         time.sleep(self.all_args.ifi - elapsed)
                # else:
                #     if self.all_args.render_sight == 'first-person':
                #         image = envs.render(mode='human', sight='first-person')[0][0]
                #     else:
                #         image = envs.render(mode='human', sight='global')[0][0]
                if self.all_args.save_gifs:
                    image = envs.render(mode='rgb_array')[0][0]
                    all_frames.append(image)
                else:
                    envs.render(mode='human')

            self.envs.envs[0].agents[-1].dead = True

            self.envs.renew()

            share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]
            
            self.policy = Policy(self.all_args,
                                 self.envs.observation_space[0],
                                 share_observation_space,
                                 self.envs.action_space[0],
                                 device = self.device)

            self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)

            self.model_dir = "/home/yanyz/yanyz/gitlab/onpolicy/onpolicy/scripts/results/MPE/rel_formation_form_error/rmappo/08-26-rel-formation-form-selfnav10-train-mpe-obs0-square/run1/models"
            self.restore()

            obs, rewards, dones, infos = envs.step(actions_env)
            self.num_agents_living = 4
            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents_living, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents_living, 1), dtype=np.float32)


            for step in range(self.episode_length_2):
                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                    np.concatenate(rnn_states),
                                                    np.concatenate(masks),
                                                    deterministic=True)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                      for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i]+1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == 'Discrete':
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                elif self.envs.action_space[0].__class__.__name__ == 'Box':
                    actions_env = actions
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                if step != self.episode_length_2 - 1:
                    obs, rewards, dones, infos = envs.step(actions_env)
                    formation_rewards.append(-infos[0][0]['formation_reward']/self.all_args.form_rew_weight)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents_living, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render(mode='rgb_array')[0][0]
                    all_frames.append(image)
                else:
                    envs.render(mode='human')

            self.envs.envs[0].agents[-2].dead = True

            self.envs.renew()

            share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]
            
            self.policy = Policy(self.all_args,
                                 self.envs.observation_space[0],
                                 share_observation_space,
                                 self.envs.action_space[0],
                                 device = self.device)

            self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)

            self.model_dir = "/home/yanyz/yanyz/gitlab/onpolicy/onpolicy/scripts/results/MPE/rel_formation_form_error/rmappo/08-26-rel-formation-form-selfnav10-train-mpe-obs0-triangle/run1/models"
            self.restore()

            obs, rewards, dones, infos = envs.step(actions_env)
            self.num_agents_living = 3
            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents_living, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents_living, 1), dtype=np.float32)


            for step in range(self.episode_length_3):
                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                    np.concatenate(rnn_states),
                                                    np.concatenate(masks),
                                                    deterministic=True)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                      for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i]+1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == 'Discrete':
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                elif self.envs.action_space[0].__class__.__name__ == 'Box':
                    actions_env = actions
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                # if step != self.episode_length_3 - 1:
                obs, rewards, dones, infos = envs.step(actions_env)
                formation_rewards.append(-infos[0][0]['formation_reward']/self.all_args.form_rew_weight)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents_living, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render(mode='rgb_array')[0][0]
                    all_frames.append(image)
                else:
                    envs.render(mode='human')

        # x = np.arange(len(formation_rewards))
        # y = formation_rewards
        # plt.plot(x,y)
        # plt.show()

        file_name = self.data_dir + 'our_model_' + str(self.all_args.seed) + '_eval_formation_change.pkl'
        with open(file_name, 'wb') as fp:
            pickle.dump(formation_rewards, fp)



            #print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        # if self.all_args.save_gifs:
        #     imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)