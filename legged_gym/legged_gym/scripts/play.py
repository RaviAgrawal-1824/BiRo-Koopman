# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import pickle as pkl

class Queue:
  def __init__(self):
    self.items = []

  def enqueue(self, item):
    """Adds an item to the back of the queue."""
    self.items.append(item)

  def dequeue(self):
    """Removes and returns the item at the front of the queue."""
    if not self.is_empty():
      return self.items.pop(0)
    return None

  def is_empty(self):
    """Returns True if the queue is empty, False otherwise."""
    return len(self.items) == 0
  
def flatten_list(list_of_lists):
  """
  Flattens a list of lists into a single list using nested loops.
  """
  flat_list = []
  for sublist in list_of_lists:
    for element in sublist:
      flat_list.append(element)
  return flat_list

def save_data(full_data):
    filename = f"data_dridh_lgm/dridh_lgm_data{full_data.shape[0]}instances_diff_v2_obsset3_mlst_slopes_1ms.pickle"
    with open(filename, "wb") as f:
        pkl.dump(full_data, f)
    print("Done")
    print("Collected data of Total ",full_data.shape[0],"instances")


def play(args):

    #########################################
    record_data = False
    policy_type = "diffusion_v2"  # "diffusion_v2" or "lgm"

    ##########################################

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    if policy_type == "diffusion_v2":
        env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    else:
        env_cfg.env.num_envs = min(env_cfg.env.num_envs, 250)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False




    ########################################################
    full_data = {}

    obs_hist = Queue()
    for i in range(5):
        obs_hist.enqueue(np.zeros((env_cfg.env.num_envs, 30)))

    if policy_type == "diffusion_v2":
        exp_name = "dridh_250250x2eps_batch4000_lr1*1e-4_numtps60_l7hidden256_obsebd48l3_10kepochs_slopes"
        from dridh_diff_v2 import test_dridh_diff_v2
        diff_policy_v2 = test_dridh_diff_v2.diff_policy_v2(path = f"/home/mothish/MtechProject/tiny-diffusion/exps/{exp_name}/model_10000.pth")
        


    ##########################################
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0


    if policy_type == "diffusion_v2":
       print("Deploying diffusion policy")
    #    env.set_camera([2,-3,1],[0,0,1])
       env.set_camera([20,0,1],[30,25,0])
    else:
       env.set_camera([15,-3,1],[10,3,0])
    for i in range(1*int(env.max_episode_length)):
        if policy_type == "lgm":
            actions = policy(obs.detach())

        #######################################
        # print(obs.detach().cpu().numpy().shape)
        # print(actions.detach().cpu().numpy().shape)
        # print(np.concatenate((obs.detach().cpu().numpy(), actions.detach().cpu().numpy()), axis=1).shape)
        # combined_array = np.concatenate((obs.detach().cpu().numpy(), actions.detach().cpu().numpy()), axis=1)
        # print(np.vstack((combined_array, combined_array)).shape)
        # # exit()
        # print(i)
        if policy_type == "diffusion_v2" or record_data == True:
            _ = obs_hist.dequeue()
            obs_hist.enqueue(obs[:, :30].detach().cpu().numpy())
            # obs_hist.enqueue(obs.detach().cpu().numpy())


            # Create an empty NumPy array to store the combined data
            obs_with_hist = np.zeros((env_cfg.env.num_envs, 150))

            # Iterate through the queue (without modifying it) and copy elements
            for j, arr in enumerate(obs_hist.items):
                obs_with_hist[:, j * 30: (j + 1) * 30] = arr.copy()

        # Print the combined array (optional)
        # print(obs_with_hist.shape)
        ########################################
        # print(i)
        if record_data:
            if i==0:
                full_data = np.concatenate((obs_with_hist, actions.detach().cpu().numpy()), axis=1)
            else:
                present_data = np.concatenate((obs_with_hist, actions.detach().cpu().numpy()), axis=1)
                full_data = np.vstack((full_data, present_data))

        if policy_type == "diffusion_v2":
        #    print(torch.squeeze(obs.detach().cpu()).shape)
        #    print(torch.squeeze(torch.tensor(obs_with_hist,device="cuda:0")).shape)
        #    diff_actions = diff_policy_v2.sample_action(torch.squeeze(obs.detach().cpu())).to(device="cuda:0")
           actions = diff_policy_v2.sample_action(torch.squeeze(torch.tensor(obs_with_hist))).to(device="cuda:0")
           obs, _, rews, dones, infos = env.step(actions)
        elif policy_type == "lgm":
           obs, _, rews, dones, infos = env.step(actions.detach())
        else:
           print("Check the policy type")
           break
        ########################################
        if policy_type == "diffusion_v2":
           if dones:
                break

        
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()
    if record_data:
        save_data(full_data)

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
