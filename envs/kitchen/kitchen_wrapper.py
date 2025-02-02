# a wrapper for Franka Kitchen, supports state description, object positioning, and motion planning
from gymnasium_robotics.envs.franka_kitchen.kitchen_env import KitchenEnv, OBS_ELEMENT_INDICES, OBS_ELEMENT_GOALS
import numpy as np
import random

N_GOALS = 7
GOAL_NAMES = ["bottom burner", "top burner", "light switch", "slide cabinet", "hinge cabinet", "microwave", "kettle"]
BODY_NAMES = ["knob 2", "knob 4", "lightswitchroot", "slidelink", "hingerightdoor", "microdoorroot", "kettleroot"]
GRASP_POS_TO_BODY = np.array([[0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [-0.183,-0.123,0.],
    [-0.302,-0.128,0.], [0.475,-0.108,0.], [0.,0.,0.259]]) # target pos = body xpos + grasp_pos_to_body
END_EFFECTOR_NAMES = ["panda0_leftfinger", "panda0_rightfinger"]

'''
a Kitchen wrapper that can easily access end-effector and object positions
'''
class KitchenWrapper(KitchenEnv):
    def __init__(self, tasks_to_complete, **kwargs):
        self.tasks_to_complete_list = list(tasks_to_complete)
        super().__init__(tasks_to_complete, **kwargs)

    def goal2idx(self, goal):
        return GOAL_NAMES.index(goal)

    # estimate end-effector pos with average pos of the two fingers
    def get_endeffector_pos(self):
        return np.mean([self.robot_env.data.body(n).xpos for n in END_EFFECTOR_NAMES], axis=0)

    # get pos for all goals
    def get_all_goal_pos(self):
        return [self.robot_env.data.body(BODY_NAMES[i]).xpos+GRASP_POS_TO_BODY[i] for i in range(N_GOALS)]

    # compute distance to all goals
    def get_dis_to_all_goals(self):
        end_pos = self.get_endeffector_pos()
        all_goal_pos = self.get_all_goal_pos()
        dis = [np.linalg.norm(end_pos-p) for p in all_goal_pos]
        return dis

    # convert 59-dim obs to 30-dim qpos & 29-dim qvel
    def obs_to_qpos_qvel(self, obs):
        robot_pos, robot_vel, obj_pos, obj_vel = obs[0:9], obs[9:18], obs[18:39], obs[39:]
        qpos, qvel = np.concatenate((robot_pos, obj_pos)), np.concatenate((robot_vel, obj_vel))
        return qpos, qvel

    # design a state description to tell LLM: current position, goal positions, goal completion
    def describe_state(self):
        ret = "Robot end-effector position: {}.\nObject positions: ".format(self.get_endeffector_pos())
        all_goal_pos = self.get_all_goal_pos()
        for i in range(N_GOALS):
            ret += "{} {}; ".format(GOAL_NAMES[i], all_goal_pos[i])
        ret += ".\nAll tasks to solve: {}, completed tasks: {}.".format(self.tasks_to_complete_list, self.episode_task_completions)
        return ret

    # input index for a goal name, return the task_complete pose of the goal, and the current pose of the goal
    def get_goal_state_and_current_state(self, goal_idx):
        return OBS_ELEMENT_GOALS[GOAL_NAMES[goal_idx]], self.data.qpos[OBS_ELEMENT_INDICES[GOAL_NAMES[goal_idx]]]

    # compute distance between current goal pose and the target goal pose
    def get_dis_to_goal_pose_completion(self, goal_idx):
        target, now = self.get_goal_state_and_current_state(goal_idx)
        return np.linalg.norm(target-now)

    # convert the planned position to the 6-dim goal for the motion-controller
    def plan2goal(self, pos):
        pos_now = self.get_endeffector_pos()
        return np.concatenate((pos, pos-pos_now))



if __name__=="__main__": 
    env = KitchenWrapper(tasks_to_complete=["microwave", "kettle", "light switch", "hinge cabinet"], 
        render_mode="human")

    import minari
    import matplotlib.pyplot as plt
    dataset = minari.load_dataset("kitchen-mixed-v1")
    #dataset.set_seed(seed=123)
    #episode = dataset.sample_episodes(n_episodes=1)[0]
    for episode in dataset:
        obs_list = episode.observations['observation']
        distances = [[] for i in range(N_GOALS)]

        env.reset()
        for obs in obs_list:
            q_pos, q_vel = env.obs_to_qpos_qvel(obs)
            env.robot_env.set_state(q_pos, q_vel)
            env.render()
            #print(env.describe_state())
            #print(env.get_goal_state_and_current_state(0))

            dis = env.get_dis_to_all_goals()
            for i in range(N_GOALS):
                distances[i].append(dis[i])

        x = [i for i in range(len(distances[0]))]
        colors = ['r','g','b','purple', 'c', 'orange', 'grey']
        for i in range(N_GOALS):
            plt.plot(x, distances[i], label=GOAL_NAMES[i], c=colors[i])
        plt.xlabel('step')
        plt.ylabel('end-effector distance to goal')
        plt.legend()
        plt.show()
        plt.cla()