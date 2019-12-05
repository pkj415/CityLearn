from reward_function import reward_function

import matplotlib.pyplot as plt
import numpy as np

class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_actions:int,
                 num_tilings:int,
                 tile_width:np.array,
                 max_action,
                 min_action
                 ):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        self.state_low = state_low
        self.num_actions = num_actions
        self.levels = num_actions
        self.num_tilings = num_tilings
        self.tile_width = tile_width
        self.min_action = min_action
        self.max_action = max_action
        self.n_dimensions = state_low.shape[0]
        self.n_tiles = np.ceil(np.round((state_high - state_low) / tile_width, 2)).astype(np.int) + 2
        print(self.n_tiles)
        print(self.tile_width)
        print(self.state_low)
        # print(np.ceil(np.round((state_high - state_low) / tile_width, 2)).astype(np.int))
        # exit()
        self.action_tile_width = np.round((max_action - min_action) / (self.levels - 1), 4)
        # self.offset = np.zeros((self.num_tilings, self.n_dimensions))
        # for dim in range(self.n_dimensions):
        #     self.offset[:,dim] = np.random.uniform(-1 * tile_width[dim], 0, size=(num_tilings))
        self.offset = np.linspace(-1 * tile_width, 0, num=num_tilings)
        self.all_dimensions = np.concatenate((np.array([self.num_actions, self.num_tilings]), self.n_tiles), axis=0)
        # print(self.all_dimensions)
        # exit()

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        return self.num_actions * self.num_tilings * np.prod(self.n_tiles)

    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        def get_index(idx, a, n_tile):
            position = 0
            super_idx = np.concatenate((np.array([a, idx]), n_tile), axis=0)
            for dimi, super_idxi in zip(self.all_dimensions, super_idx):
                position *= dimi
                position += super_idxi
            # print(super_idx, position)
            return position
        s = s[0]
        x = np.zeros((self.feature_vector_len()))
        if done: return x
        # print(s)
        # print(self.offset)
        indices = np.array((np.array([s[0], s[-1]]) - self.offset - self.state_low) // self.tile_width, dtype=np.int)
        for i, idx in enumerate(indices):
            # print(i,a,idx, self.all_dimensions)
            x[get_index(i, a, idx)] = 1.
        return x

    def discretize_actions(self, actions, min_action, n_actions):
        return np.array( (actions - self.min_action) // self.action_tile_width, dtype=np.int)

    def undiscretize_actions(self, actions, min_action, n_actions):
        a = self.min_action + actions * self.action_tile_width
        if np.any(a > self.max_action):
            print(a)
        return a 
        
def SarsaLambda(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X:StateActionFeatureVectorWithTile,
    num_episode:int,
    num_action: int,
    min_action: float,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """

    def epsilon_greedy_policy(s,done,w,epsilon=.0):
        nA = num_action
        Q = [np.dot(w, X(s,done,a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    w = np.zeros((X.feature_vector_len()))
    cost, cum_reward = np.zeros((num_episode,)), np.zeros((num_episode,))

    epsilon = 0.2
    k = 0
    for episode in range(num_episode):
        if episode % 100 == 0: print(episode)
        cum_reward[episode] = 0
        s, done = env.reset(), False
        a = epsilon_greedy_policy(s, done, w, epsilon)
        x = X(s, done, a)
        z, q_old = np.zeros(x.shape), 0
        while not done:
            if (k)%10000==0:
                    print('hour: '+str(k+1)+' of '+str(2500*num_episode)+'\r', end='')
            s_dash, r, done, _ = env.step([np.expand_dims(X.undiscretize_actions(a, min_action, num_action), axis=0)])
            reward = reward_function(r)
            a_dash = epsilon_greedy_policy(s_dash, done, w, epsilon)
            x_dash = X(s_dash, done, a_dash)
            q, q_dash = w.dot(x), w.dot(x_dash)
            delta = reward + gamma * q_dash - q
            z = gamma * lam * z + (1 - alpha * gamma * lam * z.dot(x)) * x
            w += alpha * (delta + q - q_old) * z - alpha * (q - q_old) * x
            q_old, x, a = q_dash, x_dash, a_dash
            cum_reward[episode] += reward[0]
            k+=1
        cost[episode] = env.cost()
    print(cost)
    print(cum_reward)
    print('Best Cost:', min(cost))
    print(env.action_track[8][-100:])

    s, done = env.reset(), False
    a = epsilon_greedy_policy(s, done, w)
    while not done:
        s_dash, r, done, _ = env.step([np.expand_dims(X.undiscretize_actions(a, min_action, num_action), axis=0)])
        reward = reward_function(r)
        a = epsilon_greedy_policy(s_dash, done, w)
    print('Greedy Cost:', env.cost())

    plt.plot(env.action_track[8][-100:])
    plt.plot(env.buildings[0].cooling_device.cop_cooling_list[2400:2500])
    plt.legend(['RL Action','Heat Pump COP'])
    plt.show()

    plt.plot(env.buildings[0].cooling_storage.soc_list[2400:])
    plt.plot(env.buildings[0].cooling_storage.energy_balance_list[2400:])
    plt.legend(['State of Charge','Storage device energy balance'])
    plt.show()