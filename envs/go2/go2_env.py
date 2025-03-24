from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch


class Go2Robot(LeggedRobot):
    # def _get_noise_scale_vec(self, cfg):
    #     """ Sets a vector used to scale the noise added to the observations.
    #         [NOTE]: Must be adapted when changing the observations structure

    #     Args:
    #         cfg (Dict): Environment config file

    #     Returns:
    #         [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
    #     """
    #     noise_vec = torch.zeros_like(self.obs_buf[0])
    #     self.add_noise = self.cfg.noise.add_noise
    #     noise_scales = self.cfg.noise.noise_scales
    #     noise_level = self.cfg.noise.noise_level
    #     # noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
    #     noise_vec[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
    #     noise_vec[3:6] = noise_scales.gravity * noise_level
    #     noise_vec[6:9] = 0. # commands
    #     noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
    #     noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
    #     noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions


    #     return noise_vec
    

    # def _init_foot(self):
    #     self.feet_num = len(self.feet_indices)

    #     rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
    #     self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
    #     self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
    #     self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
    #     self.feet_pos = self.feet_state[:, :, :3]
    #     self.feet_vel = self.feet_state[:, :, 7:10]

    # def _init_buffers(self):
    #     super()._init_buffers()
    #     str_rng = self.cfg.domain_rand.motor_strength_range
    #     kp_str_rng = self.cfg.domain_rand.kp_range
    #     kd_str_rng = self.cfg.domain_rand.kd_range

    #     self.motor_strength = (str_rng[1] - str_rng[0]) * torch.rand(self.num_envs, self.num_actions, dtype=torch.float,
    #                                                                  device=self.device, requires_grad=False) + str_rng[
    #                               0]
    #     self.kp_factor = (kp_str_rng[1] - kp_str_rng[0]) * torch.rand(self.num_envs, self.num_actions,
    #                                                                   dtype=torch.float, device=self.device,
    #                                                                   requires_grad=False) + kp_str_rng[0]
    #     self.kd_factor = (kd_str_rng[1] - kd_str_rng[0]) * torch.rand(self.num_envs, self.num_actions,
    #                                                                   dtype=torch.float, device=self.device,
    #                                                                   requires_grad=False) + kd_str_rng[0]
    #     self._init_foot()

    # def update_feet_state(self):
    #     self.gym.refresh_rigid_body_state_tensor(self.sim)

    #     self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
    #     self.feet_pos = self.feet_state[:, :, :3]
    #     self.feet_vel = self.feet_state[:, :, 7:10]

    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    self.last_actions
                                    ),dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    # def _process_rigid_body_props(self, props, env_id):
    #     if self.cfg.domain_rand.randomize_base_mass:
    #         rng = self.cfg.domain_rand.added_mass_range
    #         props[0].mass += np.random.uniform(rng[0], rng[1])

    #     if self.cfg.domain_rand.randomize_base_com:
    #         rng_com = self.cfg.domain_rand.added_com_range
    #         rand_com = np.random.uniform(rng_com[0], rng_com[1], size=(3, ))
    #         props[0].com += gymapi.Vec3(*rand_com)
    #     return props

    # def _compute_torques(self, actions):
    #     """ Compute torques from actions.
    #         Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
    #         [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

    #     Args:
    #         actions (torch.Tensor): Actions

    #     Returns:
    #         [torch.Tensor]: Torques sent to the simulation
    #     """
    #     #pd controller
    #     actions_scaled = actions * self.cfg.control.action_scale
    #     control_type = self.cfg.control.control_type
    #     if control_type=="P":  #根据控制模式来计算力矩
    #         if not self.cfg.domain_rand.randomize_kpkd:  # TODO add strength to gain directly
    #             torques = self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains * self.dof_vel
    #         else:
    #             torques = self.kp_factor * self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos) - self.kd_factor * self.d_gains * self.dof_vel
    #     elif control_type=="V":
    #         torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
    #     elif control_type=="T":
    #         torques = actions_scaled
    #     else:
    #         raise NameError(f"Unknown controller type: {control_type}")
    #     torques = torques * self.motor_strength
    #     return torch.clip(torques, -self.torque_limits, self.torque_limits)



    def _reward_powers(self):
        # Penalize torques
        return torch.sum(torch.abs(self.torques) * torch.abs(self.dof_vel), dim=1)

    def _reward_action_smoothness(self):
        return torch.sum(torch.square(
            self.actions - 2*self.last_actions + self.last_last_actions ), dim=1)


    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.08) * ~contact
        return torch.sum(pos_error, dim=(1))

    def _reward_alive(self):
        # Reward for staying alive
        return 1.0

    def _reward_feet_slip(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        foot_velocities = torch.square(torch.norm(self.feet_vel [:, :, 0:2], dim=2).view(self.num_envs, -1))
        rew_slip = torch.sum(contact_filt * foot_velocities, dim=1)
        return rew_slip

    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1, 2))

    # #########################################################################################################################


