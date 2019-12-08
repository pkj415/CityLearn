import numpy as np

class ValueBasedMethods:
    def __init__(self, levels, min_action, max_action, num_buildings=1):
        self.tile_width = np.round((max_action - min_action) / (levels - 1), 4)
        self.Q = np.zeros((num_buildings, 24, levels, levels))
        self.epsilon = 0.01
        self.gamma = 0.9999
        self.alpha = 0.01
        self.min_action = min_action
        self.max_action = max_action
        self.n_actions = levels

        self.Q[:,0:1, :, self.discretize_actions(np.array([0.]))[0]] = 0.001# * 10
        self.Q[:,1:9, :, self.discretize_actions(np.array([0.2]))[0]] = 0.001#* 10
        self.Q[:,9:11, :, self.discretize_actions(np.array([0.]))[0]] = 0.001#* 10
        self.Q[:,11:19, :, self.discretize_actions(np.array([-0.34]))[0]] = 0.001#* 10
        self.Q[:,19:, :, self.discretize_actions(np.array([0.]))[0]] = 0.001#* 10

    def discretize_states(self, states):
        states_copy = np.copy(states)
        states_copy[:,2] *= (self.n_actions - 1)
        states_copy[:,1] -= 17  # 17 is the minimum temperature
        states_copy[:,0] -= 1   # Convert range from 1-24 to 0-23 for hours
        return states_copy.astype(np.int)
        
    def discretize_actions(self, actions):
        return np.array((actions - self.min_action) // self.tile_width, dtype=np.int)

    def undiscretize_actions(self, actions):
        return self.min_action + actions * self.tile_width

    def get_q_value(self, states, actions):
        states = self.discretize_states(states[:])
        actions = self.discretize_actions(actions[:])
        return np.array([self.Q[i, state[0], state[2], action[0]] for i, (state, action) in enumerate(zip(states, actions))])

# class Q_Learning:
#     def __init__(self, levels, min_action, max_action):
#         super().__init__()
#         self.tile_width = np.round((max_action - min_action) / (levels - 1), 4)
#         # self.Q = np.zeros((24, levels, levels))
#         self.Q = np.zeros((24, levels))
#         self.epsilon = 0.01
#         self.gamma = 0.9999
#         self.alpha = 0.01
#         self.min_action = min_action
#         self.max_action = max_action
#         self.n_actions = levels

#     def discretize_states(self, states):
#         states_copy = np.copy(states)
#         states_copy[:,2] *= (self.n_actions - 1) #(self.tile_width / (self.max_action - self.min_action))
#         # states_copy[:,2] = np.floor(states_copy[:,2] * (self.n_actions - 1))
#         states_copy[:,1] -= 17  # 17 is the minimum temp[]
#         states_copy[:,0] -= 1
#         return states_copy.astype(np.int)
        
#     def discretize_actions(self, actions):
#         return np.array((actions - self.min_action) // self.tile_width, dtype=np.int)

#     def undiscretize_actions(self, actions):
#         return self.min_action + actions * self.tile_width

#     def select_action(self, states, p=0, greedy=False):
#         states = self.discretize_states(states)
#         actions = np.zeros(states.shape[0])
#         for i, state in enumerate(states):
#             # actions[i] = np.random.choice(np.flatnonzero(self.Q[state[0], state[2], :] == self.Q[state[0], state[2], :].max()))
#             actions[i] = np.random.choice(np.flatnonzero(self.Q[state[0], :] == self.Q[state[0], :].max()))
#             if not greedy and np.random.random() < self.epsilon * (1 - p + 0.01):
#                 actions[i] = np.random.choice(np.arange(self.n_actions))
#         return np.array([np.expand_dims(self.undiscretize_actions(actions), axis=0)])

#     def select_greedy_action(self, states):
#         return self.select_action(states, greedy=True)

#     def add_to_batch(self, states, actions, reward, next_states, dones, p=0):
#         # pass
#         # print(actions, self.discretize_actions(actions))
#         states = self.discretize_states(states[:])
#         next_states = self.discretize_states(next_states[:])
#         actions = self.discretize_actions(actions)
#         for i in range(states.shape[0]):
#             # self.Q[states[i,0], states[i,2], actions[i]] += \
#             #     self.alpha * (1 - p + 0.01) * (reward + \
#             #                     self.gamma * np.max(self.Q[next_states[i,0], next_states[i,2], :]) - \
#             #                     self.Q[states[i,0], states[i,2], actions[i]])
#             self.Q[states[i,0], actions[i]] += \
#                 self.alpha * (1 - p + 0.01) * (reward + \
#                                 self.gamma * np.max(self.Q[next_states[i,0], :]) - \
#                                 self.Q[states[i,0], actions[i]])
#         # print(states[0,0], states[0,2], actions[0,0,0], self.Q[states[i,0], states[i,2], :])
#         print(states[0,0], actions[0,0,0], self.Q[states[i,0], :], reward)

# class Q_Learning_Mult:
#     def __init__(self, levels, min_action, max_action, num_buildings):
#         self.tile_width = np.round((max_action - min_action) / (levels - 1), 4)
#         self.Q = np.zeros((num_buildings, 24, levels, levels))
#         # self.Q = np.zeros((24, levels))
#         self.epsilon = 0.01
#         self.gamma = 0.9999
#         self.alpha = 0.01
#         self.min_action = min_action
#         self.max_action = max_action
#         self.n_actions = levels

#         self.Q[:,0:1, :, self.discretize_actions(np.array([0.]))[0]] = 0.001# * 10
#         self.Q[:,1:9, :, self.discretize_actions(np.array([0.2]))[0]] = 0.001#* 10
#         self.Q[:,9:11, :, self.discretize_actions(np.array([0.]))[0]] = 0.001#* 10
#         self.Q[:,11:19, :, self.discretize_actions(np.array([-0.34]))[0]] = 0.001#* 10
#         self.Q[:,19:, :, self.discretize_actions(np.array([0.]))[0]] = 0.001#* 10

#     def discretize_states(self, states):
#         states_copy = np.copy(states)
#         states_copy[:,2] *= (self.n_actions - 1) #(self.tile_width / (self.max_action - self.min_action))
#         # states_copy[:,2] = np.floor(states_copy[:,2] * (self.n_actions - 1))
#         states_copy[:,1] -= 17  # 17 is the minimum temp[]
#         states_copy[:,0] -= 1
#         return states_copy.astype(np.int)
        
#     def discretize_actions(self, actions):
#         return np.array((actions - self.min_action) // self.tile_width, dtype=np.int)

#     def undiscretize_actions(self, actions):
#         return self.min_action + actions * self.tile_width

#     def select_action(self, states, p=0, greedy=False):
#         states = self.discretize_states(states)
#         actions = np.zeros((states.shape[0], 1))
#         for i, state in enumerate(states):
#             actions[i, 0] = np.random.choice(np.flatnonzero(self.Q[i, state[0], state[2], :] == self.Q[i, state[0], state[2], :].max()))
#             # actions[i] = np.random.choice(np.flatnonzero(self.Q[state[0], :] == self.Q[state[0], :].max()))
#             if not greedy and np.random.random() < self.epsilon * (1 - p + 0.01):
#                 actions[i, 0] = np.random.choice(np.arange(self.n_actions))
#         # print(actions)
#         return self.undiscretize_actions(actions)
#         # return np.array([self.undiscretize_actions(actions)])

#     def select_greedy_action(self, states):
#         return self.select_action(states, greedy=True)

#     def add_to_batch(self, states, actions, rewards, next_states, dones, p=0):
#         # pass
#         # print(actions, self.discretize_actions(actions))
#         states = self.discretize_states(states[:])
#         next_states = self.discretize_states(next_states[:])
#         actions = self.discretize_actions(actions)
#         # print(reward)
#         for i, (state, next_state, action, reward) in enumerate(zip(states, next_states, actions, rewards)):
#             # print(i, state, next_state, action)
#             # print(self.alpha * (1 - p + 0.01) * (reward + \
#             #                     self.gamma * np.max(self.Q[i, next_state[0], next_state[2], :]) - \
#             #                     self.Q[i, state[0], state[2], action[0]]))
#             # print(self.Q[i, next_states[i,0], next_states[i,2], :])
#             self.Q[i, state[0], state[2], action[0]] += \
#                 self.alpha * (1 - p + 0.01) * (reward + \
#                                 self.gamma * np.max(self.Q[i, next_state[0], next_state[2], :]) - \
#                                 self.Q[i, state[0], state[2], action[0]])
#             # self.Q[states[i,0], actions[i]] += \
#             #     self.alpha * (1 - p + 0.01) * (reward + \
#             #                     self.gamma * np.max(self.Q[next_states[i,0], :]) - \
#             #                     self.Q[states[i,0], actions[i]])
#         # print(states[0,0], states[0,2], actions[0,0,0], self.Q[states[i,0], states[i,2], :])
#         # print(states[0,0], actions[0,0,0], self.Q[states[i,0], :], reward)

class Q_Learning_Mult(ValueBasedMethods):
    def __init__(self, levels, min_action, max_action, num_buildings=1):
        super().__init__(levels, min_action, max_action, num_buildings)
        # self.tile_width = np.round((max_action - min_action) / (levels - 1), 4)
        # self.Q = np.zeros((num_buildings, 24, levels, levels))
        # # self.Q = np.zeros((24, levels))
        # self.epsilon = 0.01
        # self.gamma = 0.9999
        # self.alpha = 0.01
        # self.min_action = min_action
        # self.max_action = max_action
        # self.n_actions = levels

        # self.Q[:,0:1, :, self.discretize_actions(np.array([0.]))[0]] = 0.001# * 10
        # self.Q[:,1:9, :, self.discretize_actions(np.array([0.2]))[0]] = 0.001#* 10
        # self.Q[:,9:11, :, self.discretize_actions(np.array([0.]))[0]] = 0.001#* 10
        # self.Q[:,11:19, :, self.discretize_actions(np.array([-0.34]))[0]] = 0.001#* 10
        # self.Q[:,19:, :, self.discretize_actions(np.array([0.]))[0]] = 0.001#* 10

    def select_action(self, states, p=0, greedy=False):
        states = self.discretize_states(states)
        actions = np.zeros((states.shape[0], 1))
        for i, (state, action) in enumerate(zip(states, actions)):
            action[0] = np.random.choice(np.flatnonzero(self.Q[i, state[0], state[2], :] == self.Q[i, state[0], state[2], :].max()))
            if not greedy and np.random.random() < self.epsilon * (1 - p + 0.01):
                action[0] = np.random.choice(np.arange(self.n_actions))
        return self.undiscretize_actions(actions)

    def select_greedy_action(self, states):
        return self.select_action(states, greedy=True)

    def add_to_batch(self, states, actions, rewards, next_states, dones, p=0):
        states = self.discretize_states(states[:])
        next_states = self.discretize_states(next_states[:])
        actions = self.discretize_actions(actions)
        for i, (state, next_state, action, reward) in enumerate(zip(states, next_states, actions, rewards)):
            self.Q[i, state[0], state[2], action[0]] += \
                self.alpha * (1 - p + 0.01) * (reward + \
                                self.gamma * np.max(self.Q[i, next_state[0], next_state[2], :]) - \
                                self.Q[i, state[0], state[2], action[0]])

class Sarsa:
    def __init__(self, levels, min_action, max_action, num_buildings=1):
        self.tile_width = np.round((max_action - min_action) / (levels - 1), 4)
        self.Q = np.zeros((num_buildings, 24, levels, levels))
        # self.Q = np.zeros((24, levels))
        self.epsilon = 0.01
        self.gamma = 0.9999
        self.alpha = 0.01
        self.min_action = min_action
        self.max_action = max_action
        self.n_actions = levels

    def discretize_states(self, states):
        states_copy = np.copy(states)
        states_copy[:,2] *= (self.n_actions - 1) #(self.tile_width / (self.max_action - self.min_action))
        # states_copy[:,2] = np.floor(states_copy[:,2] * (self.n_actions - 1))
        states_copy[:,1] -= 17  # 17 is the minimum temp[]
        states_copy[:,0] -= 1
        return states_copy.astype(np.int)

    def discretize_actions(self, actions):
        return np.array((actions - self.min_action) // self.tile_width, dtype=np.int)

    def undiscretize_actions(self, actions):
        return self.min_action + actions * self.tile_width

    def select_action(self, states, p=0, greedy=False):
        states = self.discretize_states(states)
        actions = np.zeros((states.shape[0], 1))
        for i, (state, action) in enumerate(zip(states, actions)):
            action[0] = np.random.choice(np.flatnonzero(self.Q[i, state[0], state[2], :] == self.Q[i, state[0], state[2], :].max()))
            # actions[i] = np.random.choice(np.flatnonzero(self.Q[state[0], :] == self.Q[state[0], :].max()))
            if not greedy and np.random.random() < self.epsilon * (1 - p + 0.01):
                action[0] = np.random.choice(np.arange(self.n_actions))
        return self.undiscretize_actions(actions)

    def select_greedy_action(self, states):
        return self.select_action(states, greedy=True)

    def add_to_batch(self, states, actions, rewards, next_states, next_actions, dones, p=0):
        states = self.discretize_states(states[:])
        next_states = self.discretize_states(next_states[:])
        actions = self.discretize_actions(actions[:])
        next_actions = self.discretize_actions(next_actions[:])
        for i, (state, next_state, action, next_action, reward) in enumerate(zip(states, next_states, actions, next_actions, rewards)):
            self.Q[i, state[0], state[2], action[0]] += \
                self.alpha * (1 - p + 0.01) * (reward + \
                                self.gamma * self.Q[i, next_state[0], next_state[2], next_action[0]] - \
                                self.Q[i, state[0], state[2], action[0]])


# class N_Sarsa:
#     def __init__(self, levels, min_action, max_action, num_buildings=1):
#         self.tile_width = np.round((max_action - min_action) / (levels - 1), 4)
#         self.Q = np.zeros((num_buildings, 24, levels, levels))
#         self.epsilon = 0.01
#         self.gamma = 0.9999
#         self.alpha = 0.01
#         self.min_action = min_action
#         self.max_action = max_action
#         self.n_actions = levels

#         self.Q[:,0:1, :, self.discretize_actions(np.array([0.]))[0]] = 0.001# * 10
#         self.Q[:,1:9, :, self.discretize_actions(np.array([0.2]))[0]] = 0.001#* 10
#         self.Q[:,9:11, :, self.discretize_actions(np.array([0.]))[0]] = 0.001#* 10
#         self.Q[:,11:19, :, self.discretize_actions(np.array([-0.34]))[0]] = 0.001#* 10
#         self.Q[:,19:, :, self.discretize_actions(np.array([0.]))[0]] = 0.001#* 10

#     def discretize_states(self, states):
#         states_copy = np.copy(states)
#         states_copy[:,2] *= (self.n_actions - 1) #(self.tile_width / (self.max_action - self.min_action))
#         # states_copy[:,2] = np.floor(states_copy[:,2] * (self.n_actions - 1))
#         states_copy[:,1] -= 17  # 17 is the minimum temp[]
#         states_copy[:,0] -= 1
#         return states_copy.astype(np.int)

#     def discretize_actions(self, actions):
#         return np.array((actions - self.min_action) // self.tile_width, dtype=np.int)

#     def undiscretize_actions(self, actions):
#         return self.min_action + actions * self.tile_width

#     def select_action(self, states, p=0, greedy=False):
#         states = self.discretize_states(states)
#         actions = np.zeros((states.shape[0], 1))
#         for i, (state, action) in enumerate(zip(states, actions)):
#             action[0] = np.random.choice(np.flatnonzero(self.Q[i, state[0], state[2], :] == self.Q[i, state[0], state[2], :].max()))
#             if not greedy and np.random.random() < self.epsilon * (1 - p + 0.01):
#                 action[0] = np.random.choice(np.arange(self.n_actions))
#         return self.undiscretize_actions(actions)

#     def select_greedy_action(self, states):
#         return self.select_action(states, greedy=True)

#     def get_q_value(self, states, actions):
#         states = self.discretize_states(states[:])
#         actions = self.discretize_actions(actions[:])
#         return np.array([self.Q[i, state[0], state[2], action[0]] for i, (state, action) in enumerate(zip(states, actions))])

#     def add_to_batch(self, states, actions, returns, dones, p=0):
#         states = self.discretize_states(states[:])
#         actions = self.discretize_actions(actions[:])
#         for i, (state, action, return_g) in enumerate(zip(states, actions, returns)):
#             self.Q[i, state[0], state[2], action[0]] += \
#                 self.alpha * (1 - p + 0.01) * (return_g - self.Q[i, state[0], state[2], action[0]])

class N_Sarsa(ValueBasedMethods):
    def __init__(self, levels, min_action, max_action, num_buildings=1):
        super().__init__(levels, min_action, max_action, num_buildings)

    def undiscretize_actions(self, actions):
        return self.min_action + actions * self.tile_width

    def select_action(self, states, p=0, greedy=False):
        states = self.discretize_states(states)
        actions = np.zeros((states.shape[0], 1))
        for i, (state, action) in enumerate(zip(states, actions)):
            action[0] = np.random.choice(np.flatnonzero(self.Q[i, state[0], state[2], :] == self.Q[i, state[0], state[2], :].max()))
            if not greedy and np.random.random() < self.epsilon * (1 - p + 0.01):
                action[0] = np.random.choice(np.arange(self.n_actions))
        return self.undiscretize_actions(actions)

    def select_greedy_action(self, states):
        return self.select_action(states, greedy=True)

    def add_to_batch(self, states, actions, returns, dones, p=0):
        states = self.discretize_states(states[:])
        actions = self.discretize_actions(actions[:])
        for i, (state, action, return_g) in enumerate(zip(states, actions, returns)):
            self.Q[i, state[0], state[2], action[0]] += \
                self.alpha * (1 - p + 0.01) * (return_g - self.Q[i, state[0], state[2], action[0]])

class Random(ValueBasedMethods):
    def __init__(self, levels, min_action, max_action, num_buildings=1):
        super().__init__(levels, min_action, max_action, num_buildings)

    def select_action(self, states, p=0, greedy=False):
        actions = np.zeros((states.shape[0], 1))
        for i in range(len(states)):
            actions[i, 0] = np.random.choice(np.arange(self.n_actions))
        return self.undiscretize_actions(actions)

    def select_greedy_action(self, states):
        return self.select_action(states, greedy=True)

    def add_to_batch(self, states, actions, rewards, next_states, dones, p=0):
        pass