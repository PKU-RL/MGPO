''' a Kitchen wrapper for sequential execution of subtasks
Speficy tasks that require sequential execution of some subtasks in order
'''

from gymnasium_robotics.envs.franka_kitchen.kitchen_env import KitchenEnv
import numpy as np
import random
#from gymnasium import spaces

N_GOALS_TEST = 5
GOAL_NAMES_TEST = ["bottom burner", "light switch", "slide cabinet", "hinge cabinet", "microwave"]

class KitchenSeqEnv(KitchenEnv):
    def __init__(self, task=None, **kwargs): # task is a list of obj indices (in the order)
        self.env_name = 'kitchen'
        self.discrete_action = False
        self.prompt_dim = N_GOALS_TEST
        if task is not None:
            self.subtask_idxs = task
            self.num_subtasks = len(task)
        else:
            self.sample_task()
        self.subtask_names = [GOAL_NAMES_TEST[i] for i in self.subtask_idxs]
        self.max_return = self.num_subtasks
        super().__init__(GOAL_NAMES_TEST, **kwargs) # the inner env considers all subtasks
        #self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(59,))
        #self.action_space = spaces.Box(low=-1, high=1, shape=(9,))

    def sample_task(self, num_subtasks=None):
        if num_subtasks is None:
            num_subtasks = random.randint(1, N_GOALS_TEST)
        self.num_subtasks = num_subtasks
        self.max_return = self.num_subtasks
        self.subtask_idxs = random.sample([i for i in range(N_GOALS_TEST)], k=num_subtasks)
        self.subtask_names = [GOAL_NAMES_TEST[i] for i in self.subtask_idxs]
        return self.subtask_idxs

    # get the whole prompt sequence
    def get_optimal_prompt(self):
        prompt = []
        for i in range(self.num_subtasks):
            subgoal = np.zeros(N_GOALS_TEST, dtype=np.float32)
            subgoal[self.subtask_idxs[i]] = 1
            prompt.append(subgoal)
        prompt += self.get_task_prompt()
        return prompt

    # get the final goal of the task
    def get_task_prompt(self):
        completion = [0 for i in range(N_GOALS_TEST)]
        for i in range(self.num_subtasks):
            completion[self.subtask_idxs[i]] = 1
        return [np.array(completion, dtype=np.float32)]

    def print_episode_stats(self, prompt=None):
        prompt_list=[]
        if prompt is not None:
            prompt_np = (prompt[0].cpu().numpy()[0]).astype(int)
            mask_np = prompt[1].cpu().numpy()[0].astype(int)
            prompt_np = prompt_np[(mask_np!=0).argmax():]
            prompt_list = []
            for i, p in enumerate(prompt_np):
                if i<len(prompt_np)-1:
                    prompt_list.append(np.argmax(p))
                else:
                    prompt_list.append(p)
        print('task: {} {}, prompt: {}, visited: {}'.format(self.subtask_names, self.subtask_idxs, prompt_list, self.episode_visited_goal_ids))

    def reset(self):
        self.obj_states = [0 for i in range(N_GOALS_TEST)]
        self.active_goal_idx = 0
        self._enforce_reset = False
        self.episode_visited_goals, self.episode_visited_goal_ids = [], [] # save prompt goals in episode experience, for online adaptation
        obs, _ = super().reset()
        return obs['observation']

    def step(self, act):
        assert not self._enforce_reset, "Reset the environment with `env.reset()`"
        
        obs, _, _, _, info = super().step(act)
        step_task_completions = info['step_task_completions']
        #assert len(step_task_completions)<=1

        # when any subtask is completed, add to visited goals
        if len(step_task_completions)>0:
            #print(step_task_completions)
            obj = GOAL_NAMES_TEST.index(step_task_completions[0])
            self.obj_states[obj] = 1
            subgoal = np.zeros(N_GOALS_TEST, dtype=np.float32)
            subgoal[obj] = 1
            if obj not in self.episode_visited_goal_ids:
                self.episode_visited_goals.append(subgoal) # if an obj goal state is triggered, add to visited goals
                self.episode_visited_goal_ids.append(obj)

        success = self.detect_success() 
        terminated, task_success = False, False
        rew = float(success) / self.max_return # normalized score as reward
        if success:
            if self.active_goal_idx == self.num_subtasks - 1:
                terminated = True
                task_success = True
                self._enforce_reset = True
            self.active_goal_idx += 1

        return obs['observation'], rew, terminated, {"success": success, "task_success": task_success}

    # subtask success: subtasks before current should be done; subtasks after current should not be done
    def detect_success(self):
        success = True
        for i in range(self.active_goal_idx+1):
            if self.obj_states[self.subtask_idxs[i]] != 1:
                success = False
                break
        for i in range(self.active_goal_idx+1, self.num_subtasks):
            if self.obj_states[self.subtask_idxs[i]] == 1:
                success = False
                break
        return success


    def render(self, return_rgb=True):
        return super().render()