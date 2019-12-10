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


def reset_all(entities):
    for entity in entities.values():
        entity.reset()

def get_cost_of_building(building_uids, **kwargs):
    env, buildings, heat_pump, heat_tank, cooling_tank = create_env(building_uids, **kwargs)
    agents = get_agents(buildings, heat_pump, cooling_tank, **kwargs)

    # Add different agents below.
    if kwargs["agent"] in ["RBC", "Random", "Degenerate"]:
        state = env.reset()
        done = False
        while not done:
            action = agents.select_action(state)
            next_state, rewards, done, _ = env.step(action)
            state = next_state
        cost = env.cost()
        print("Cost: " + str(cost))
    elif kwargs["agent"] == "DDP":
        learning_start_time = time.time()
        optimal_action_val = run_dp(heat_pump[building_uids[-1]],
          cooling_tank[building_uids[-1]], buildings[-1], **kwargs)
        learning_end_time = time.time()

        done = False
        time_step = 0
        while not done:
          _, rewards, done, _ = env.step([[optimal_action_val[time_step]]])
          time_step += 1
        cost_via_dp = env.cost()
        print("Cost via DDP - {0}, Total charges made - {1}, Learning time - {2}".format(cost_via_dp, env.get_total_charges_made(),
          learning_end_time - learning_start_time))
    elif kwargs["agent"] == "Q":
        episodes = kwargs["episodes"]
        cost, cum_reward, greedy_cost, greedy_reward = \
            np.zeros((episodes,)), np.zeros((episodes,)), np.zeros((episodes,)), np.zeros((episodes,))

        for e in range(episodes): 
            print('Episode: '+str(e+1)+' of '+str(episodes)+'\r', end='')

            cum_reward[e] = 0
            state = env.reset()

            done = False
            while not done:
                actions = agents.select_action(state, e/episodes)
                next_state, rewards, done, _ = env.step(actions)
                reward = reward_function(rewards) #See comments in reward_function.py
                agents.add_to_batch(state, actions, reward, next_state, done, e/episodes)
                state = next_state
                cum_reward[e] += reward[0]
            cost[e] = env.cost()

            # Greedy Run
            greedy_reward[e] = 0
            state = env.reset()
            done = False
            while not done:
                action = agents.select_greedy_action(state)
                next_state, rewards, done, _ = env.step(action)
                reward = reward_function(rewards)
                state = next_state
                greedy_reward[e] += reward[0]
            curr_cost = env.cost()
            greedy_cost[e] = curr_cost

        print("Best Cost", min(greedy_cost))
    elif kwargs["agent"] == "N_Sarsa":
        episodes = kwargs["episodes"]
        cost, cum_reward, greedy_cost, greedy_reward = \
            np.zeros((episodes,)), np.zeros((episodes,)), np.zeros((episodes,)), np.zeros((episodes,))
        gamma = 0.9999
        n = kwargs["n"]

        for e in range(episodes): 
            print('Episode: '+str(e+1)+' of '+str(episodes)+'\r', end='')
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
                    _return_g = np.zeros((len(state)))
                    for i in range(tau+1, min(tau+n, T)+1):
                        _return_g += gamma**(i-tau-1) * traj_rewards[i]
                    if tau + n < T:
                        _return_g += (gamma ** n) * agents.get_q_value(traj_states[tau+n], traj_actions[tau+n])
                    agents.add_to_batch(traj_states[tau], traj_actions[tau], _return_g, done) #, e/episodes)
                if tau == T-1:
                    break
            curr_cost = env.cost()
            cost[e] = curr_cost

            # Greedy Run
            state = env.reset()
            done = False
            while not done:
                action = agents.select_greedy_action(state)
                next_state, rewards, done, _ = env.step(action)
                reward = reward_function(rewards)
                state = next_state
            curr_cost = env.cost()
            greedy_cost[e] = curr_cost
        print("Best Cost: ", min(greedy_cost))
    elif kwargs["agent"] == "SarsaLambda":
        X = StateActionFeatureVectorWithTile(
        state_low=np.array([1, kwargs["min_charge_val"]]),
        state_high=np.array([24, kwargs["max_charge_val"]]),
        num_actions=kwargs["action_levels"],
        num_tilings=1,
        tile_width=np.array([1., (kwargs["max_charge_val"] - kwargs["min_charge_val"])/(kwargs["charge_levels"]-1)]),
        max_action=kwargs["max_action_val"],
        min_action=kwargs["min_action_val"]
        )
        gamma = 0.9999
        SarsaLambda(env, gamma, kwargs["lamda"], 0.01, X, kwargs["episodes"], kwargs["action_levels"], kwargs["min_action_val"])
    elif kwargs["agent"] == "QPlanningTiles":
        from q_planning_tiles import QPlanningTiles

        cop_cooling = buildings[-1].cooling_device.eta_tech*(buildings[-1].cooling_device.t_target_cooling + 273.15)/(buildings[-1].sim_results['t_out'] - buildings[-1].cooling_device.t_target_cooling)
        elec_consump = max(buildings[-1].sim_results['cooling_demand']/cop_cooling)
        max_storing_consump = max(buildings[-1].cooling_storage.capacity/cop_cooling)
        print("------- Configuraiton for QPlanner -------")
        print("Setting elec_consump to {0:.2f}+{1:.2f}={2:.2f}".format(elec_consump, max_storing_consump, max_storing_consump+elec_consump))

        agents = QPlanningTiles(storage_capacity=cooling_tank[building_uids[-1]].capacity, elec_consump=elec_consump+max_storing_consump,
            parameterize_actions=kwargs["use_parameterized_actions"], use_adaptive_learning_rate=kwargs["use_adaptive_learning_rate"],
            level_cnt=kwargs["action_levels"])

        e_num = 1
        num_episodes = kwargs["episodes"]
        while True:
            if num_episodes != 0 and e_num > num_episodes:
                break

            agents.replay_buffer = []

            done = False
            state = env.reset()
            episode_start_time = time.time()
            while not done:
                # Note: Do not consider this as the agent using environment information directly (env object is used here just for
                # convenience now, that should change, as it seems from the look of it that we are using env information).
                # It is only using the cooling demand of the previous time step which it has already taken an action on, and an actual
                # controller can actually measure this. We are not violating the fact that we don't know the environment dynamics.

                # TODO: Fix the abstraction to not use env object to get this information. This can cause misinterpretations.
                # print("Going to select action")

                # action = [[0.0]]
                action = agents.select_action(state)

                next_state, rewards, done, _ = env.step(action)
                # print("Env: For state {0}, {1} -> {2}, {3}".format(state, action, next_state, rewards))

                # print("Chose action {0} for time_step {1}".format(action, env.time_step))
                print("state {0}, time {1}, reward^2 {2}".format(state, env.time_step, rewards[-1]*rewards[-1]))
                cooling_demand_prev_step = env.buildings[-1].sim_results['cooling_demand'][env.time_step-1]
                
                agents.update_prev_cooling_demand(cooling_demand_prev_step)
                agents.update_on_transition(rewards[-1], next_state, done)

                state = next_state

            episode_end_time = time.time()
            cost = env.cost()
            print("Episode {0}: {1}, {2}, {3}".format(e_num, cost, env.get_total_charges_made(),
                episode_end_time - episode_start_time))

            # Plots
            # soc = [i/env.buildings[0].cooling_storage.capacity for i in env.buildings[0].cooling_storage.soc_list]

            # Plots for the last 100 hours of the simulation
            # plt.plot([20*action for action in env.action_track[args.building_uids[-1]][:]])
            # plt.plot(env.buildings[0].cooling_device.cop_cooling_list[:])
            # plt.plot(soc[:]) #State of the charge
            # plt.legend(['RL Action','Heat Pump COP', 'SOC'])
            # plt.show()

            e_num += 1
    elif kwargs["agent"] in ["TD3", "DDPG"]:
        episodes = kwargs["episodes"]
        cost, cum_reward = np.zeros((episodes,)), np.zeros((episodes,))

        for e in range(episodes): 
            print('Episode: '+str(e+1)+' of '+str(episodes)+'\r', end='')

            cum_reward[e] = 0
            state = env.reset()

            done = False
            while not done:
                actions = agents.select_action(state)
                next_state, rewards, done, _ = env.step(actions)
                rewards = reward_function(rewards) #See comments in reward_function.py
                agents.add_to_batch(state, actions, rewards, next_state, done)
                state = next_state
                cum_reward[e] += rewards[0]
            cost[e] = env.cost()

        print("Best Cost", min(cost))
    
args = parse_arguments()

get_cost_of_building(args.building_uids, start_time=args.start_time, end_time=args.end_time,
    action_levels=args.action_levels, min_action_val=args.min_action_val, max_action_val=args.max_action_val,
    charge_levels=args.charge_levels, min_charge_val=args.min_charge_val, max_charge_val=args.max_charge_val,
    agent=args.agent, episodes=args.episodes, n=args.n, lamda=args.lamda, target_cooling=args.target_cooling,
    use_adaptive_learning_rate=args.use_adaptive_learning_rate, use_parameterized_actions=args.use_parameterized_actions)
