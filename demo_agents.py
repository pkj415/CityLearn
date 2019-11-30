import copy
from energy_models import HeatPump
import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import signal
import utils

np.random.seed(1)

class QLearningTiles:
    # All "Out tweak" comments are to note down the design choices we made for the algorithm. It will help
    # us keep track of them and if we should change them. Plus, for noting in the final report.
    def __init__(self, storage_capacity, elec_consump, gamma=0.9, alpha=0.1, epsilon=0.0, level_cnt=9,
            efficiency=1.0, loss_coefficient=0.0, parameterize_actions=False): #0.19/24
        self.num_updates = 0
        # TODO: Our Tweak -> Wrap around Generalize for the hour of the day in a circular fashion

        # Configurables
        # [Hour of the day, outside temperature, charge available, the action level]

        if parameterize_actions:
          self.state_low = [1, 0, 0.0, -0.5]
          self.state_high = [24, 20, 1.0, 0.5]
          self.tile_widths = [2, 4, 0.2, 0.2]
        else:
          self.state_low = [1, -6.4, 0.0]
          self.state_high = [24, 39.1, 1.0]
          self.tile_widths = [2, 4, 0.2]

        self.level_cnt = level_cnt

        from tc import ValueFunctionWithTile

        self.num_tilings = 10
        self.initial_weight_value = -1 * (elec_consump*elec_consump) / self.num_tilings

        self.parameterize_actions = parameterize_actions
        if not self.parameterize_actions:
            self.Q_sa = []
            for _ in range(level_cnt):
                self.Q_sa.append(ValueFunctionWithTile(
                    self.state_low, self.state_high, num_tilings=self.num_tilings,
                    tile_width=self.tile_widths, initial_weight_value=self.initial_weight_value,
                    wrap_around=[False, False, False], use_standard_tile_coding=False))
        else:
            self.Q_sa = ValueFunctionWithTile(
                    self.state_low, self.state_high, num_tilings=self.num_tilings,
                    tile_width=self.tile_widths, initial_weight_value=self.initial_weight_value,
                    wrap_around=[False, False, False, False], use_standard_tile_coding=False)

        # Learning method params
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

        self.action_disc = utils.Discretizer(min_val=-0.5, max_val=0.5, level_cnt=level_cnt)
        self.charge_disc = utils.Discretizer(min_val=0.0, max_val=1.0, level_cnt=level_cnt)

        # Format of one state in the replay buffer -
        # Hour of day, t_outdoor, cooling_demand. Below will have a list of these states.
        self.replay_buffer = []

        # Note that we have not included charge on ES, we will be updating Q values for states at
        # all charge levels In essence, we will be making updates for unseen states which can be
        # done as there is no correlation on which charge level you start with and the cooling
        # demand at that time step

        # Environment specific variables that are allowed to be used by the agent.
        self.storage_capacity = storage_capacity
        self.efficiency = efficiency
        self.loss_coefficient = loss_coefficient

        self.max_action_val_seen_till_now = 0.0

        self.num_visits = {}

        self.got_stop_signal = False

    def update_prev_cooling_demand(self, cooling_demand):
        self.replay_buffer[-1]["cooling_demand"] = abs(cooling_demand)

    def get_max_action(self, state, action_val=None, Q_sa_copy=None):
        max_action = 0
        max_action_val = float('-inf')

        if not Q_sa_copy:
            Q_sa_copy = self.Q_sa

        for action in range(self.level_cnt):
            if self.action_disc.get_val(action) > 0:
                if self.action_disc.get_val(action) > 1 - state[2]:
                    continue
            else:
                if -1 * self.action_disc.get_val(action) > state[2]:
                    continue

            # Testing Hack
            if action_val is not None and action_val != self.action_disc.get_val(action):
                continue

            state_action = list(state)
            state_action.append(action)

            q_sa_val = None
            if not self.parameterize_actions:
                q_sa_val = Q_sa_copy[action](list(state))
            else:
                input_ = list(state)
                input_.append(self.action_disc.get_val(action))
                q_sa_val = Q_sa_copy(input_)

            if q_sa_val > max_action_val:
                max_action_val = q_sa_val
                max_action = action

        return max_action, max_action_val

    def plan_on_replay_buffer(self, num_iterations=1, without_updates=False, rel_delta=True):
        delta = float('inf')
        idx = 0

        if without_updates:
            num_iterations = 1

        print("Replay buffer - {0}".format(len(self.replay_buffer)))
        alpha = 1.0

        # if not without_updates:
        #     alpha = float(input("What alpha value?"))

        self.max_action_val_seen_till_now = 0.0
        self.min_action_val_seen_till_now = float('inf')

        prev_delta = float('inf')
        alpha_ceil = 1.0

        self.got_stop_signal = False
        self.num_times_delta_inc = 0

        max_delta_ratio = 0.0
        while True: #* self.min_action_val_seen_till_now: #idx < num_iterations or (
            if self.got_stop_signal:
                break

            self.max_action_val_seen_till_now = 0.0
            idx += 1

            prev_state = {}

            delta = 0.0
            Q_sa_copy = copy.deepcopy(self.Q_sa)
            for state in self.replay_buffer:
                if self.got_stop_signal:
                    break

                if not prev_state:
                    prev_state = state
                    continue

                # Our Tweak -> We can make updates for different states which we haven't even seen!
                for charge_level in range(self.level_cnt): # int(self.level_cnt/2)
                    if self.got_stop_signal:
                        break

                    charge_val = self.charge_disc.get_val(charge_level)

                    # To account for losses when storing energy.
                    charge_val = charge_val*(1-self.loss_coefficient)

                    for action in range(self.level_cnt):
                        if self.got_stop_signal:
                            break

                        action_val = self.action_disc.get_val(action)
                        # Testing hack
                        # if charge_level != 0 or action_val != 0:
                        #     continue

                        # print("Checking state {0} charge {1} action_val {2}".format(state, charge_val, action_val))
                        if action_val < 0 and -1 * action_val > charge_val:
                            continue
                        if action_val > 0 and action_val > 1 - charge_val:
                            continue

                        cooling_pump = HeatPump(nominal_power = 9e12, eta_tech = 0.22, t_target_heating = 45, t_target_cooling = 1)
                        cooling_pump.set_cop(t_source_cooling = prev_state["t_out"])

                        # TODO: Not handling cases where without charge in ES, we can't satisfy our cooling demand.
                        assert(cooling_pump.get_max_cooling_power(t_source_cooling = prev_state["t_out"]) > prev_state["cooling_demand"])
                        cooling_power_avail = cooling_pump.get_max_cooling_power(t_source_cooling = prev_state["t_out"]) - prev_state["cooling_demand"]

                        # Don't accept charge values which require charging more than possible, and which discharge more than required.
                        # Not updating weights for these values will help in not using this for updates and focussing representational power
                        # in valid actions only. TODO: Trying removing and confirm the benefits.

                        if action_val >= 0:
                            # Note the different in action_val in this and DDP. In DDP an action means drawing energy from pump to get us that
                            # much of charging and hence we use /efficiency and not * efficiency in that.
                            if action_val*self.storage_capacity*self.efficiency > cooling_power_avail:
                                continue

                            cooling_energy_to_storage = action_val*self.storage_capacity*self.efficiency
                        else:
                            if -1 * action_val*self.storage_capacity*self.efficiency > prev_state["cooling_demand"]:
                                continue
                            cooling_energy_to_storage = action_val*self.storage_capacity*self.efficiency


                        next_charge_val = charge_val + cooling_energy_to_storage/self.storage_capacity
                        cooling_energy_drawn_from_heat_pump = cooling_energy_to_storage + prev_state["cooling_demand"]
                        elec_demand_cooling = cooling_pump.get_electric_consumption_cooling(cooling_supply = cooling_energy_drawn_from_heat_pump)

                        q_val = None
                        if not self.parameterize_actions:
                            q_val = self.Q_sa[action]([prev_state["hour_of_day"], prev_state["t_out"], self.charge_disc.get_val(charge_level)])
                        else:
                            q_val = self.Q_sa([prev_state["hour_of_day"], prev_state["t_out"], self.charge_disc.get_val(charge_level), self.action_disc.get_val(action)])

                        max_action, max_action_val = self.get_max_action(
                            [state["hour_of_day"], state["t_out"], next_charge_val], action_val=None, Q_sa_copy=Q_sa_copy)

                        if max_action_val == self.initial_weight_value * self.num_tilings:
                            print("Found you!!! State {0} max_action {1} max_action_val {2}, coming from charge_val of prev state {3}".format(
                                [state["hour_of_day"], state["t_out"], next_charge_val],
                                self.action_disc.get_val(max_action),
                                max_action_val, charge_val))
                        self.max_action_val_seen_till_now = max(self.max_action_val_seen_till_now, abs(max_action_val))
                        self.min_action_val_seen_till_now = min(self.min_action_val_seen_till_now, abs(max_action_val))
                        
                        # if self.charge_disc.get_val(charge_level) == 0.0 and self.action_disc.get_val(action) == 0.0:
                        #     print("Qs, a for {0} is {1}".format(
                        #         [prev_state["hour_of_day"], prev_state["t_out"], self.charge_disc.get_val(charge_level), action],
                        #         q_val))
                        # if (charge_level == 0 or charge_level == 1) and (action_val == 0.0 or action == 10):
                        # print("Qs, a for {0} is {1}, Target {2}, Q*s' for {3} is {4} with action {5},{6}".format(
                        #         [prev_state["hour_of_day"], prev_state["t_out"], self.charge_disc.get_val(charge_level), action], q_val,
                        #         - 1 * (elec_demand_cooling*elec_demand_cooling) + self.gamma * max_action_val,
                        #         [state["hour_of_day"], state["t_out"], next_charge_val], max_action_val, max_action,self.action_disc.get_val(max_action)))

                        curr_delta = abs(- 1 * (elec_demand_cooling*elec_demand_cooling) +
                            self.gamma * max_action_val - q_val)
                        delta = max(delta, curr_delta)
                        delta_ratio = curr_delta/abs(q_val)
                        max_delta_ratio = max(max_delta_ratio, delta_ratio)
                        prev_state_action = [prev_state["hour_of_day"], prev_state["t_out"], self.charge_disc.get_val(charge_level), action]
                        next_state_action = [state["hour_of_day"], state["t_out"], next_charge_val, max_action]
                        
                        # Testing hack 
                        # tupl = (prev_state["hour_of_day"], prev_state["t_out"], self.charge_disc.get_val(charge_level), action)
                        # if tupl not in self.num_visits:
                        #     self.num_visits[tupl] = 0
                        # self.num_visits[tupl] += 1

                        if not without_updates:
                            if not self.parameterize_actions:
                                self.Q_sa[action].update(alpha, - 1 * (elec_demand_cooling*elec_demand_cooling) + self.gamma * max_action_val,
                                    [prev_state["hour_of_day"], prev_state["t_out"], self.charge_disc.get_val(charge_level)])
                            else:
                                self.Q_sa.update(alpha, - 1 * (elec_demand_cooling*elec_demand_cooling) + self.gamma * max_action_val,
                                    [prev_state["hour_of_day"], prev_state["t_out"], self.charge_disc.get_val(charge_level),
                                    self.action_disc.get_val(action)])
                prev_state = state

            print("Done Planning iteration {0}: {1} with buffer size {2}, delta={3}, max_action_value={4} min_action_val={5} "
                "delta_ratio={6}".format(
                "Without updates" if without_updates else "normal",
                idx, len(self.replay_buffer), delta,
                self.max_action_val_seen_till_now,
                self.min_action_val_seen_till_now,
                max_delta_ratio))

            # Testing hack
            if without_updates:
                return max_delta_ratio

            max_delta_ratio = self.plan_on_replay_buffer(num_iterations=1, without_updates=True)

            if max_delta_ratio < 0.001:
                print("Breaking as max delta ratio < 0.01")
                break

            if delta > prev_delta:
                self.num_times_delta_inc += 1
                if self.num_times_delta_inc <= 3:
                    prev_delta = delta
                    continue

                self.num_times_delta_inc = 0
                print("Delta {0} > prev_delta {1}. Changing alpha {2} -> {3}".format(delta, prev_delta, alpha, alpha/2))
                alpha /= 2
                if alpha < 0.001:
                    break

                alpha_ceil = alpha
            else:
                prev_alpha = alpha
                alpha = min(2*alpha, alpha_ceil)
                print("Delta {0} <= prev_delta {1}. Changing alpha {2} -> {3}".format(delta, prev_delta, prev_alpha, alpha))

            alpha = 0.1
            prev_delta = delta

            # if idx == num_iterations:
            #     while True:
            #         try:
            #             num_iterations = int(input("Plan for how many more iterations?"))
            #             alpha = float(input("What alpha value?"))
            #             break
            #         except Exception as exc:
            #             import traceback
            #             print("Error, try again ... {0}".format(traceback.format_exc()))
            #     idx = 0

    def select_action(self, state):
        # State - Hour of day, temperatue, charge on ES
        self.current_state = state[0]

        # TODO: Change below for multi building case later
        max_action, max_action_val = self.get_max_action(self.current_state)

        # print("Selecting action for state {0}, {1} -> {2}".format(state,
        #     max_action, max_action_val), flush=True)

        if (random.random() < self.epsilon):
            print("Chose randomly but")
            self.chosen_action = random.randint(0, self.level_cnt-1)
        else:
            self.chosen_action = max_action

        # Maintain a replay buffer of just 25 hours for now.
        if len(self.replay_buffer) >= 25:
            self.replay_buffer = self.replay_buffer[1:]

        self.replay_buffer.append(
            {
                "cooling_demand": -1,
                "hour_of_day": self.current_state[0],
                "t_out": self.current_state[1]
            })

        return [[self.action_disc.get_val(self.chosen_action)]]

    def update_on_transition(self, r, s_next, done):
        self.num_updates += 1
        max_action_next_val = 0.0

        # TODO: Change below for multi building case later
        next_state = s_next[0]
        
        max_action = -1
        if not done:
            max_action, max_action_next_val = self.get_max_action(next_state)

        state_action = list(self.current_state)
        state_action.append(self.chosen_action)
        next_state_action = list(next_state)
        next_state_action.append(max_action)

        q_val = None
        if not self.parameterize_actions:
            q_val = self.Q_sa[self.chosen_action](list(self.current_state))
        else:
            input_ = list(self.current_state)
            input_.append(self.action_disc.get_val(self.chosen_action))
            q_val = self.Q_sa(input_)

        delta = \
            abs(
                -1 * r * r + self.gamma * max_action_next_val - q_val)
        # print("Update: {0}, {1} -> {2}, {3} Diff {4}".format(state_action,
        #     self.Q_sa[self.chosen_action](list(self.current_state)),
        #     next_state_action, max_action_next_val,
        #     delta))
        # self.Q_sa[self.chosen_action].update(
        #     self.alpha, -1 * r * r + self.gamma * max_action_next_val, list(self.current_state))

        if len(self.replay_buffer) >= 25:
            print("New planning buffer")
            self.plan_on_replay_buffer()
