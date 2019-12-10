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


Sample command to run DDP -
python main.py --agent DDP --building_uids 8 --action_level 9 --start_time 3500 --end_time 3600 --target_cooling 1

Sample command to run QLearningTiles with adaptive tile coding and action parameterization -
python main.py --agent QPlanningTiles --building_uids 8 --action_level 9 --start_time 3500 --end_time 3600 --target_cooling 1 --use_adaptive_learning_rate True --True