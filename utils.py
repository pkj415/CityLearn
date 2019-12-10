from citylearn import  CityLearn, building_loader, auto_size
from energy_models import HeatPump, EnergyStorage, Building

import numpy as np

import argparse
from pathlib import Path

class Discretizer():
	def __init__(self, min_val, max_val, level_cnt):
		self.min_val = min_val
		self.max_val = max_val
		self.level_cnt = level_cnt
		vals = []
		for i in range(level_cnt):
			vals.append(self.get_val(i))
		print("Disc values from {0} to {1} are {2}".format(min_val, max_val, vals))


	# Gives level just below val (flooring)
	def get_level(self, val):
		slab_size = (self.max_val - self.min_val)/(self.level_cnt-1)
		return int((val - self.min_val)/slab_size)

	# Gives val of level
	def get_val(self, level):
		slab_size = (self.max_val - self.min_val)/(self.level_cnt-1)
		return slab_size*level + self.min_val

def create_env(building_uids, **kwargs):
    
    data_folder = Path("data/")
    demand_file = data_folder / "AustinResidential_TH.csv"
    weather_file = data_folder / 'Austin_Airp_TX-hour.csv'

    max_action_val = kwargs["max_action_val"]
    min_action_val = kwargs["min_action_val"]
    target_cooling = kwargs["target_cooling"]

    heat_pump, heat_tank, cooling_tank = {}, {}, {}

    loss_coeff, efficiency = 0.19/24, 1.

    # Ref: Assessment of energy efficiency in electric storage water heaters (2008 Energy and Buildings)
    buildings = []
    for uid in building_uids:
        heat_pump[uid] = HeatPump(nominal_power = 9e12, eta_tech = 0.22, 
                                    t_target_heating = 45, t_target_cooling=target_cooling)
        heat_tank[uid] = EnergyStorage(capacity = 9e12, loss_coeff = loss_coeff)
        cooling_tank[uid] = EnergyStorage(capacity = 9e12, loss_coeff = loss_coeff)
        buildings.append(Building(uid, heating_storage = heat_tank[uid], 
                            cooling_storage = cooling_tank[uid], heating_device = heat_pump[uid],
                            cooling_device = heat_pump[uid], sub_building_uids=[uid]))
        buildings[-1].state_space(np.array([24.0, 40.0, 1.001]), np.array([1.0, 17.0, -0.001]))
        buildings[-1].action_space(np.array([max_action_val]), np.array([min_action_val]))
        
    building_loader(demand_file, weather_file, buildings)
    auto_size(buildings, t_target_heating = 45, t_target_cooling=target_cooling)

    env = CityLearn(demand_file, weather_file, buildings = buildings, time_resolution = 1,
                    simulation_period = (kwargs["start_time"]-1, kwargs["end_time"]))

    return env, buildings, heat_pump, heat_tank, cooling_tank

def get_agents(buildings, heat_pumps, cooling_tanks, **kwargs):
    agent = kwargs["agent"]
    # Add different agents below.
    if agent == "RBC":
        # RULE-BASED CONTROLLER (Stores energy at night and releases it during the day)
        from policy_grad_agent import RBC_Agent
        agents = RBC_Agent(degenerate=False)
    elif agent == "Degenerate":
        from policy_grad_agent import RBC_Agent
        agents = RBC_Agent(degenerate=True)
    elif agent == "Random":
        from value_approx_agent import Random
        agents = Random(kwargs["action_levels"], kwargs["min_action_val"], kwargs["max_action_val"], len(buildings))
    elif agent in ["DDP", "QPlanningTiles"]:
    	return None
    elif agent == "Q":
        from value_approx_agent import Q_Learning_Mult
        assert kwargs["action_levels"] == kwargs["charge_levels"], "For Q Learning action_levels and charge_levels must be same"
        agents = Q_Learning_Mult(kwargs["action_levels"], kwargs["min_action_val"], kwargs["max_action_val"], len(buildings))
    elif agent == "N_Sarsa":
        from value_approx_agent import N_Sarsa
        assert kwargs["action_levels"] == kwargs["charge_levels"], "For Q Learning action_levels and charge_levels must be same"
        agents = N_Sarsa(kwargs["action_levels"], kwargs["min_action_val"], kwargs["max_action_val"], len(buildings))
    elif agent == "TD3":
        from policy_grad_agent import TD3_Agents
        # Extracting the state-action spaces from the buildings to feed them to the agent(s)
        observations_space, actions_space = [],[]
        for building in buildings:
            observations_space.append(building.observation_spaces)
            actions_space.append(building.action_spaces)
        agents = TD3_Agents(observations_space,actions_space)
    elif agent == "DDPG":
        from policy_grad_agent import RL_Agents
        # Extracting the state-action spaces from the buildings to feed them to the agent(s)
        observations_space, actions_space = [],[]
        for building in buildings:
            observations_space.append(building.observation_spaces)
            actions_space.append(building.action_spaces)
        agents = RL_Agents(observations_space,actions_space)
    elif agent in ["SarsaLambda"]:
        return None
    else:
        raise Exception("Unsupported Agent")
    return agents

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action_levels',
                        help='number of action levels. Choose odd number if you want zero charging action',
                        type=int, default=49)
    parser.add_argument('--min_action_val', help='min action value >= -1.', type=float,
                        default=-1.)
    parser.add_argument('--max_action_val', help='max action value <= 1.', type=float,
                        default=1.)
    parser.add_argument('--charge_levels',
                        help='number of charge levels. Choose odd number if you want zero charge value allowed',
                        type=int, default=49)
    parser.add_argument('--min_charge_val', help='min charge value >= 0.', type=float,
                        default=0.)
    parser.add_argument('--max_charge_val', help='max charge value <= 1.', type=float,
                        default=1.)
    parser.add_argument('--start_time',
                        help='Start hour. Note: For less than 3500 hr, there seems to be no data for a building 8, check this',
                        type=int, default=3500)
    parser.add_argument('--end_time', help='End hour', type=int, default=6000)
    parser.add_argument('--building_uids', nargs='+', type=int, required=True)
    parser.add_argument('--agent', type=str,
                        choices=['RBC', 'DDP', 'TD3', 'Q', 'DDPG', 'SarsaLambda', 'N_Sarsa', 'QPlanningTiles', 'Degenerate', 'Random'], required=True)
    parser.add_argument('--episodes', type=int, help="Num episodes", default=10)
    parser.add_argument('--n', help='n Step', type=int, default=1)
    parser.add_argument('--target_cooling', type=int, help="Indoor temperature", default=10)
    parser.add_argument('--use_adaptive_learning_rate', type=bool, help="Applies only for QPlanner", default=False)
    parser.add_argument('--use_parameterized_actions', type=bool, help="Applies only for QPlanner", default=True)
    parser.add_argument('--lamda', help='Lambda for Sarsa Lambda', type=float, default=0.9)

    args = parser.parse_args()
    assert args.min_action_val <= 0., "Can't discharge as min_action_val <= 0."
    assert args.max_action_val >= 0., "Can't charge as max_action_val >= 0."
    return args