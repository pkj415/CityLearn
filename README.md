Sample command to run DDP -
python main.py --agent DDP --building_uids 8 --action_level 9 --start_time 3500 --end_time 3600 --target_cooling 1

Sample command to run QLearningTiles with adaptive tile coding and action parameterization -
python main.py --agent QPlanningTiles --building_uids 8 --action_level 9 --start_time 3500 --end_time 3600 --target_cooling 1 --use_adaptive_learning_rate True --True