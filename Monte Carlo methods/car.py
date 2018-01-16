from collections import defaultdict
import random

class Car():
    def __init__(self, environment, select_action_fn):
        self.environment = environment
        self._reset()
        self.Q = defaultdict(lambda: defaultdict(lambda: float("-inf")))
        self.returns_sum = defaultdict(float)
        self.returns_count = defaultdict(float)
        self.actions = []
        self.greedy_probability = 0.9
        self.select_action_fn = select_action_fn

        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0: continue
                self.actions.append((i, j))
                
        self.P = {}
    
    def __str__(self):
        return "#"
    
    def _reset(self):
        self.speed = (0, 0)
        self.position = self.environment.select_start_position()
    
    def select_action(self):
        return self.select_action_fn(self)
    
    def train(self, episodes=10, update_policy_each=10):
        self.Q = defaultdict(lambda: defaultdict(float))
        # self.P = {}
        self.returns_sum = defaultdict(float)
        self.returns_count = defaultdict(float)
        
        for i in range(episodes):
            print("\rStarted iteration {}    ".format(i), end="")
    
            steps, rewards = self.play()
            R = self._calculate_returns(steps, rewards)

            for (state, action) in R:
                reward = R[(state, action)]
                self.returns_sum[state] += reward
                self.returns_count[state] += 1
                q_state = self.Q[state]
                q_state[action] = self.returns_sum[state] / self.returns_count[state]

            print("\rFinished iteration {}     ".format(i), end="")

            if i % update_policy_each == 0:
                self.P = self.calculate_policy(self.Q)

        print("")
    
    def _calculate_returns(self, steps, rewards):
        total_reward = 0
        R = defaultdict(float)
        for i in range(len(rewards)-1, -1, -1):
            state, action = steps[i]
            total_reward += rewards[i]
            R[(state, action)] = total_reward
            
        return R
    
    def calculate_policy(self, Q):
        P = {}
        for state in Q:
            max_value = float("-inf")
            max_action = None
            
            for action in Q[state]:                    
                if Q[state][action] > max_value:
                    max_value = Q[state][action]
                    max_action = action
                    
            if max_action:
                P[state] = max_action

        return P
    
    def play(self):
        steps = []
        rewards = []
        count = 0
        self._reset()

        while True:
            reward, action = self.step()
            if action == (0, 0): continue

            state = (self.position, self.speed)
            steps.append((state, action))
            rewards.append(reward)
            count += 1
            
            if self.environment.is_finish(self.position):
                break
            
        return steps, rewards
    
    def step(self):
        old_position = self.position
        action = self.select_action()
        self.speed = (self.speed[0] + action[0], self.speed[1] + action[1])
        self.speed = (min(4, self.speed[0]), min(4, self.speed[1]))
        self.speed = (max(-4, self.speed[0]), max(-4, self.speed[1]))

        new_position = (self.position[0] + self.speed[0], self.position[1] + self.speed[1])
        new_position, _path = self.environment.move_to(self, new_position)
        self.position = new_position

        if not self.environment.is_start(old_position) and self.environment.is_start(new_position):
            self.speed = (0, 0)
        
        return self.reward(new_position), action
    
    def reward(self, new_position):
        if self.environment.is_finish(new_position):
            return 0

        return -1
