This repository contains code developed for CS394R: Reinforcement Learning: Theory and Practice project offered in Fall 2019 at University of Texas at Austin. 

This project uses Reinforcement Learning based agents on the [CityLearn](https://github.com/intelligent-environments-lab/CityLearn) envrionment. This code was developed by [Vaibhav Sinha](https://vbsinha.github.io) and [Piyush Jain](https://github.com/pkj415/).

## Requirements

This code has been written in Python 3 and requires numpy, gym, matplotlib and pandas.

## Running Experiments

The exceutable file is `main.py`. Here are the main command line parameters for the code:
```
usage: main.py [-h] 
               --building_uids BUILDING_UIDS [BUILDING_UIDS ...]
               --agent {RBC,DDP,TD3,Q,DDPG,SarsaLambda,N_Sarsa,QPlanningTiles,Degenerate}
               [--action_levels ACTION_LEVELS]
               [--min_action_val MIN_ACTION_VAL]
               [--max_action_val MAX_ACTION_VAL]
               [--charge_levels CHARGE_LEVELS]
               [--min_charge_val MIN_CHARGE_VAL]
               [--max_charge_val MAX_CHARGE_VAL] 
               [--start_time START_TIME]
               [--end_time END_TIME] 
               [--episodes EPISODES]
               [--n N]
               [--target_cooling TARGET_COOLING]
               [--use_adaptive_learning_rate USE_ADAPTIVE_LEARNING_RATE]
               [--use_parameterized_actions USE_PARAMETERIZED_ACTIONS]
```

As an example to run Q-Learning with reduced action space -0.5 to 0.5 on building 8 with 5 levels of discretization for charge and action for 80 episodes run:

```
python main.py --agent Q --building_uids 8 --max_action_val 0.5 --min_action_val=-0.5 --action_levels 5 --charge_levels 5 --episodes 80
 ```

 To run Q-Learning or any other algorithms on multiple buildings simply add the building uids sequentially
 ```
python main.py --agent Q --building_uids 8 21 67 --max_action_val 0.5 --min_action_val=-0.5 --action_levels 5 --charge_levels 5 --episodes 80
 ```

To run Sarsa, use N Step Sarsa and set N=1. For general N step Sarsa use N appropriately. N is ignored if the agent is not N_Sarsa
```
python main.py --agent N_Sarsa --building_uids 8 --max_action_val 0.5 --min_action_val=-0.5 --action_levels 5 --charge_levels 5 --episodes 80 --n 1
```

Similarly for Sarsa Lamda pass lamda parameter. Lamda is ignored if the agent is not SarsaLambda.
```
python main.py --agent SarsaLambda --building_uids 8 --max_action_val 0.5 --min_action_val=-0.5 --action_levels 5 --charge_levels 5 --episodes 80 --lamda 0.9
```

To find the performance of a Random Q-Learning/Sarsa agent use
```
python main.py --agent Random --building_uids 8 --max_action_val 0.5 --min_action_val=-0.5 --action_levels 5 --charge_levels 5
```
Notice that this does not use episodes. It always runs for one.

To get the RBC Baseline values, use:
```
python main.py --agent RBC --building_uids 8
```
This does not take any additional parameters.

To get the Degenerate Baseline values, use:
```
python main.py --agent Degenerate --building_uids 8
```
This does not take any additional parameters.


Sample command to run DDP -
python main.py --agent DDP --building_uids 8 --action_level 9 --start_time 3500 --end_time 3600 --target_cooling 1

Sample command to run QLearningTiles with adaptive tile coding and action parameterization -
python main.py --agent QPlanningTiles --building_uids 8 --action_level 9 --start_time 3500 --end_time 3600 --target_cooling 1 --use_adaptive_learning_rate True --True

## Code Structure

The files `citylearn.py`, `energy_models.py` and `reward_function.py` contains code for the environment.

`ddp.py` has code for obatining theoretical best cost (DDP baseline).

`policy_grad_agent.py` has code for RBC and Degenerate agents as well as TD3 and DDPG agents (code adapted from the [official implementation](https://github.com/sfujim/TD3)).

`sarsa.py` has code for Sarsa Lambda agent.

`main.py`, `main_piyush.py` amd `utils.py` handle the interfacing.

`value_approx_agent.py` implements Q-Learning, N Step Sarsa and Random agent.

The data used can be found in `data` directory.

## License
This code is provided using the [MIT License](LICENSE).
