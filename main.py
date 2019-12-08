from citylearn import  CityLearn, building_loader, auto_size
from energy_models import HeatPump, EnergyStorage, Building
from utils import create_env, get_agents, parse_arguments
from reward_function import reward_function
from sarsa import SarsaLambda, StateActionFeatureVectorWithTile
from ddp import run_dp

import matplotlib.pyplot as plt
import numpy as np

import logging
from pathlib import Path
import sys
import time
from itertools import count

logger = logging.getLogger('spam_application')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

# TODO: Extend to multiple buildings (assuming we have the same cooling demand pattern for the buildings for a time period).
# TODO: Execute the optimal policy with the environment to assert the cost we have found.

# TODO: Ensure positive transfer irrespective of start state

def reset_all(entities):
    for entity in entities.values():
        entity.reset()

def get_cost_of_building(building_uids, **kwargs):
    '''
    Get the cost of a single building from start_time to end_time using DP and discrete action and charge levels.
    '''
    env, buildings, heat_pump, heat_tank, cooling_tank = create_env(building_uids, **kwargs)
    agents = get_agents(buildings, heat_pump, cooling_tank, **kwargs)

    # print(agents.undiscretize_actions(agents.discretize_actions(np.array([0.34]))))

    # Add different agents below.
    if kwargs["agent"] == "RBC":
        state = env.reset()
        done = False
        while not done:
            action = agents.select_action(state)
            next_state, rewards, done, _ = env.step(action)
            state = next_state
        cost_rbc = env.cost()
        print(cost_rbc)
        logger.info("{0}, {1}".format(cost_rbc, env.get_total_charges_made()))
    elif kwargs["agent"] == "DDP":
        learning_start_time = time.time()
        optimal_action_val = run_dp(heat_pump[building_uids[-1]],
          cooling_tank[building_uids[-1]], buildings[-1], **kwargs)
        learning_end_time = time.time()
        # print("DDP Learning time ")

        # env = CityLearn(demand_file, weather_file, buildings = [buildings[-1]], time_resolution = 1,
        #   simulation_period = (kwargs["start_time"]-1, kwargs["end_time"]))
        done = False
        time_step = 0
        while not done:
          _, rewards, done, _ = env.step([[optimal_action_val[time_step]]])
          time_step += 1
        cost_via_dp = env.cost()
        print("Cost via DDP - {0}, Total charges made - {1}, Learning time - {2}".format(cost_via_dp, env.get_total_charges_made(),
          learning_end_time - learning_start_time))
    elif kwargs["agent"] == "Q":
        k = 0
        episodes = kwargs["episodes"]
        cost, cum_reward, greedy_cost, greedy_reward = \
            np.zeros((episodes,)), np.zeros((episodes,)), np.zeros((episodes,)), np.zeros((episodes,))

        for e in range(episodes): #A stopping criterion can be added, which is based on whether the cost has reached some specific threshold or is no longer improving
            cum_reward[e] = 0
            state = env.reset()

            done = False
            while not done:
                if (k)%10000==0:
                    print('hour: '+str(k+1)+' of '+str(2*2500*episodes)+'\r', end='')
                actions = agents.select_action(state, e/episodes)
                # print(actions)
                next_state, rewards, done, _ = env.step(actions)
                reward = reward_function(rewards) #See comments in reward_function.py
                agents.add_to_batch(state, actions, reward, next_state, done, e/episodes)
                state = next_state
                cum_reward[e] += reward[0]
                k+=1
            cost[e] = env.cost()

            # Greedy Run
            greedy_reward[e] = 0
            state = env.reset()
            done = False
            while not done:
                action = agents.select_greedy_action(state)
                next_state, rewards, done, _ = env.step(action)
                reward = reward_function(rewards) #See comments in reward_function.py
                state = next_state
                greedy_reward[e] += reward[0]
                k+=1
            curr_cost = env.cost()
            print(str(curr_cost) + '             \r', end='')
            greedy_cost[e] = curr_cost

            # plt.plot(env.action_track[8][-100:])
            # plt.plot(env.buildings[0].cooling_device.cop_cooling_list[2400:2500])
            # plt.legend(['RL Action','Heat Pump COP'])
            # plt.show()
            # print(env.action_track[8][-100:])

            # plt.plot(env.buildings[0].cooling_storage.soc_list[2400:])
            # plt.plot(env.buildings[0].cooling_storage.energy_balance_list[2400:])
            # plt.legend(['State of Charge','Storage device energy balance'])
            # plt.show()
        print(cost)
        print(greedy_cost)
        print("Best Cost", min(greedy_cost))
        # print(env.action_track[8][-100:])
        # print(env.buildings[0].cooling_storage.energy_balance_list[2400:])

        # plt.plot(env.action_track[8][-100:])
        # plt.plot(env.buildings[0].cooling_device.cop_cooling_list[-100:])
        # plt.legend(['RL Action','Heat Pump COP'])
        # plt.show()
        # print(env.action_track[building_uids[0]][-100:])

        # plt.plot(env.buildings[0].cooling_storage.soc_list[-100:])
        # plt.plot(env.buildings[0].cooling_storage.energy_balance_list[-100:])
        # plt.legend(['State of Charge','Storage device energy balance'])
        # plt.show()
    elif kwargs["agent"] == "SarsaLambda":
        X = StateActionFeatureVectorWithTile(
        np.array([1, kwargs["min_charge_val"]]),
        np.array([24, kwargs["max_charge_val"]]),
        kwargs["action_levels"],
        num_tilings=1,
        tile_width=np.array([1., (kwargs["max_charge_val"] - kwargs["min_charge_val"])/(kwargs["charge_levels"]-1)]),
        max_action=kwargs["max_action_val"],
        min_action=kwargs["min_action_val"]
        )
        gamma = 0.9999

        SarsaLambda(env, gamma, 0.85, 0.01, X, kwargs["episodes"], kwargs["action_levels"], kwargs["min_action_val"])
    elif kwargs["agent"] == "Sarsa":
        k = 0
        episodes = kwargs["episodes"]
        cost, cum_reward, greedy_cost, greedy_reward = \
            np.zeros((episodes,)), np.zeros((episodes,)), np.zeros((episodes,)), np.zeros((episodes,))

        for e in range(episodes): #A stopping criterion can be added, which is based on whether the cost has reached some specific threshold or is no longer improving
            cum_reward[e] = 0
            state = env.reset()
            action = agents.select_action(state) #, e/episodes)
            done = False
            while not done:
                if (k)%10000==0:
                    print('hour: '+str(k+1)+' of '+str(2*2500*episodes)+'\r', end='')
                    # exit()
                next_state, rewards, done, _ = env.step(action)
                reward = reward_function(rewards) #See comments in reward_function.py
                next_action = agents.select_action(next_state) #, e/episodes)
                agents.add_to_batch(state, action, reward, next_state, next_action, done) #, e/episodes)
                state = next_state
                action = next_action
                cum_reward[e] += reward[0]
                k+=1
            curr_cost = env.cost()
            print(str(curr_cost) + '             \r', end='')
            cost[e] = curr_cost
            # Greedy Run
            greedy_reward[e] = 0
            state = env.reset()
            done = False
            while not done:
                action = agents.select_greedy_action(state)
                next_state, rewards, done, _ = env.step(action)
                reward = reward_function(rewards) #See comments in reward_function.py
                state = next_state
                greedy_reward[e] += reward[0]
                k+=1
            curr_cost = env.cost()
            print(str(curr_cost) + '             \r', end='')
            greedy_cost[e] = curr_cost
        print(cost)
        print(greedy_cost)
        print("Best Cost", min(cost), min(greedy_cost))
        # print(env.action_track[8][-100:])
        # print(env.buildings[0].cooling_storage.energy_balance_list[2400:])

        # plt.plot(env.action_track[8][-100:])
        # plt.plot(env.buildings[0].cooling_device.cop_cooling_list[2400:2500])
        # plt.legend(['RL Action','Heat Pump COP'])
        # plt.show()

        # plt.plot(env.buildings[0].cooling_storage.soc_list[2400:])
        # plt.plot(env.buildings[0].cooling_storage.energy_balance_list[2400:])
        # plt.legend(['State of Charge','Storage device energy balance'])
        # plt.show()
    elif kwargs["agent"] == "DPDiscr":
        learning_start_time = time.time()
        optimal_action_val = run_dp(heat_pump[building_uids[-1]],
        cooling_tank[building_uids[-1]], buildings[-1], **kwargs)
        learning_end_time = time.time()

        env = CityLearn(demand_file, weather_file, buildings = [buildings[-1]], time_resolution = 6,
        simulation_period = (kwargs["start_time"]-1, kwargs["end_time"]))
        done = False
        time_step = 0
        while not done:
            _, rewards, done, _ = env.step([[optimal_action_val[time_step]]])
        time_step += 1
        cost_via_dp = env.cost()
        logger.info("{0}, {1}, {2}".format(cost_via_dp, env.get_total_charges_made(),
        learning_end_time - learning_start_time))
    elif kwargs["agent"] == "TD3":
        k = 0
        episodes = kwargs["episodes"]
        cost, cum_reward, greedy_cost, greedy_reward = \
            np.zeros((episodes,)), np.zeros((episodes,)), np.zeros((episodes,)), np.zeros((episodes,))

        for e in range(episodes): #A stopping criterion can be added, which is based on whether the cost has reached some specific threshold or is no longer improving
            cum_reward[e] = 0
            state = env.reset()

            done = False
            while not done:
                if (k)%1000==0:
                    print('hour: '+str(k+1)+' of '+str(2*2500*episodes)+'\r', end='')
                actions = agents.select_action(state)
                next_state, rewards, done, _ = env.step(actions)
                rewards = reward_function(rewards) #See comments in reward_function.py
                agents.add_to_batch(state, actions, rewards, next_state, done)
                state = next_state
                cum_reward[e] += rewards[0]
                k+=1
            cost[e] = env.cost()

            # plt.plot(env.action_track[8][-100:])
            # plt.plot(env.buildings[0].cooling_device.cop_cooling_list[2400:2500])
            # plt.legend(['RL Action','Heat Pump COP'])
            # plt.show()
            # print(env.action_track[8][-100:])

            # plt.plot(env.buildings[0].cooling_storage.soc_list[2400:])
            # plt.plot(env.buildings[0].cooling_storage.energy_balance_list[2400:])
            # plt.legend(['State of Charge','Storage device energy balance'])
            # plt.show()
        print(cost)
        # print(greedy_cost)
        print("Best Cost", min(cost))
        # print(env.action_track[8][-100:])
        # print(env.buildings[0].cooling_storage.energy_balance_list[2400:])

        # plt.plot(env.action_track[8][-100:])
        # plt.plot(env.buildings[0].cooling_device.cop_cooling_list[-100:])
        # plt.legend(['RL Action','Heat Pump COP'])
        # plt.show()
        # print(env.action_track[building_uids[0]][-100:])

        # plt.plot(env.buildings[0].cooling_storage.soc_list[-100:])
        # plt.plot(env.buildings[0].cooling_storage.energy_balance_list[-100:])
        # plt.legend(['State of Charge','Storage device energy balance'])
        # plt.show()
    elif kwargs["agent"] == "N_Sarsa":
        k = 0
        episodes = kwargs["episodes"]
        cost, cum_reward, greedy_cost, greedy_reward = \
            np.zeros((episodes,)), np.zeros((episodes,)), np.zeros((episodes,)), np.zeros((episodes,))
        gamma = 0.9999
        n = kwargs["n"]
        print(n)

        for e in range(episodes): #A stopping criterion can be added, which is based on whether the cost has reached some specific threshold or is no longer improving
            if (k)%10000==0:
                print('hour: '+str(k+1)+' of '+str(2*2500*episodes)+'\r', end='')
            cum_reward[e] = 0
            state = env.reset()
            action = agents.select_action(state) #, e/episodes)
            traj_states, traj_actions, traj_rewards = [state], [action], [np.zeros((len(state),))]
            T = 2500
            done = False
            for t in count(0):
                if t < T:
                    next_state, rewards, done, _ = env.step(action)
                    rewards = reward_function(rewards) #See comments in reward_function.py
                    traj_states.append(next_state)
                    traj_rewards.append(rewards)
                    if done != True:
                        next_action = agents.select_action(next_state) #, e/episodes)
                        traj_actions.append(next_action)
                        action = next_action
                tau = t - n + 1
                if tau >= 0:
                    # print(type(traj_rewards[1]))
                    # print(traj_rewards)
                    _return_g = np.zeros((len(state)))
                    for i in range(tau+1, min(tau+n, T)+1):
                        _return_g += gamma**(i-tau-1) * traj_rewards[i]
                    if tau + n < T:
                        _return_g += (gamma ** n) * agents.get_q_value(traj_states[tau+n], traj_actions[tau+n])
                    agents.add_to_batch(traj_states[tau], traj_actions[tau], _return_g, done) #, e/episodes)
                if tau == T-1:
                    break
                k+=1
            curr_cost = env.cost()
            # print(T)
            print(str(curr_cost) + '             \r', end='')
            cost[e] = curr_cost
            # Greedy Run
            state = env.reset()
            done = False
            while not done:
                action = agents.select_greedy_action(state)
                next_state, rewards, done, _ = env.step(action)
                reward = reward_function(rewards) #See comments in reward_function.py
                state = next_state
                k+=1
            curr_cost = env.cost()
            print(str(curr_cost) + '             \r', end='')
            greedy_cost[e] = curr_cost
        # print(cost)
        # print(greedy_cost)
        print("Best Cost", min(cost), min(greedy_cost))

args = parse_arguments()

# logger.info("Cost, Total charging done, Learning time")
get_cost_of_building(args.building_uids, start_time=args.start_time, end_time=args.end_time,
    action_levels=args.action_levels, min_action_val=args.min_action_val, max_action_val=args.max_action_val,
    charge_levels=args.action_levels, min_charge_val=args.min_action_val, max_charge_val=args.max_action_val,
    agent=args.agent, episodes=args.episodes, n=args.n, target_cooling=args.target_cooling)
