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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class DridhRoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 217 # 169 with rough terrain
        num_actions = 6

    
    class terrain( LeggedRobotCfg.terrain):
        # measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5] # 1mx1m rectangle (without center line)
        # measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        
        mesh_type = 'trimesh'
        measure_heights = True

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.55] # x,y,z [0.0, 0.0, 0.533] [m] 
        # init angles from mpc 0, -0.58, -1.13, 0, -0.7, -1.05
        # default_joint_angles = { # = target angles [rad] when action = 0.0
        #     'hip_left': -0.078,
        #     'thigh_left': -0.32,
        #     'shank_left': -1.1,

        #     'hip_right': -0.16,
        #     'thigh_right': 0.45,
        #     'shank_right': -0.82
        # }

        default_joint_angles = { # = target angles [rad] when action = 0.0
            'hip_left': -0.078,
            'thigh_left': 0.024,
            'shank_left': -0.28,

            'hip_right': -0.16,
            'thigh_right': 0.12,
            'shank_right': 0.061
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        # stiffness = {   'hip':400.0, 'thigh':220.0,
        #                 'shank':220.}  # [N*m/rad]
        # damping = { 'hip': 16, 'thigh': 1.8,
        #             'shank': 0.3}  # [N*m*s/rad]     # [N*m*s/rad]
        
        # PD Drive parameters:
        stiffness = {   'hip':400.0, 'thigh':220.0,
                        'shank':150.}  # [N*m/rad]
        damping = { 'hip': 16, 'thigh': 2.4,
                    'shank': 1.3}  # [N*m*s/rad]     # [N*m*s/rad]
        
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        
    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/dridh/urdf/Dridh_URDF.urdf'
        name = "dridh"
        foot_name = 'Shank'
        terminate_after_contacts_on = ['Torso','Thigh','Hip']
        flip_visual_attachments = False
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
    class commands(LeggedRobotCfg.commands) :
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            # lin_vel_x = [-0.5, 0.5] # min max [m/s]
            # lin_vel_y = [-0.5, 0.5]   # min max [m/s]

            lin_vel_x = [0.5, 0.5] # min max [m/s]
            lin_vel_y = [0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [0, 0]    # min max [rad/s]
            heading = [0, 0]
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.60
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        base_height_target = 0.60
        max_contact_force = 1.
        only_positive_rewards = False
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 11.0
            tracking_ang_vel = 10.0
            termination = -100
            torques = -5.e-6
            dof_vel = -0.56
            dof_acc = -2.e-7 #-2.e-7
            lin_vel_z = -0.6
            feet_air_time = 10.
            dof_pos_limits = -1.
            base_height = 20
            orientation = -0.1
            no_fly = 3.35
            dof_vel = -0.0
            ang_vel_xy = -0.80
            action_rate = -0.80
            feet_contact_forces = -0.009 #-0.002
            difference_in_feet_contact_forces = -0.00000

class DridhRoughCfgPPO( LeggedRobotCfgPPO ):
    
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_dridh'
        max_iterations = 5000 # number of policy updates

    class algorithm( LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01



  