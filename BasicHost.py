import pandas as pd
import pygame
import numpy as np
from main import GameManager
from utils.helperFunctions import action_number_into_function, function_into_action_number
import time

class BasicHost(object):

    def __init__(self,env,ep_num):
        self.env = env
        self.ep_num = ep_num

    def calc_action(self,env):
        return 0

    def run_episode(self):


        running = True
        self.env.reset()
        done = 0
        df = pd.DataFrame()
        self.action_dict = action_number_into_function(self.env.tables,self.env.unique_combos,self.env.immutable_config)
        self.function_dict = function_into_action_number(self.env.tables,self.env.unique_combos,self.env.immutable_config)
        pygame.init()
        raw_obs = []
        raw_actions = []
        while(running):
            events = pygame.event.get()
            action = self.env.handle_events(events,self.function_dict)

            if action != -1:
                #If we have done an action that inpacts environment then we step the env
                obs, rew, done, _, _ = self.env.step(action)
                #Write the raw action and observation for BC later
                raw_obs.append(obs)
                raw_actions.append(action)

                # Write action and reward
                row_dict = dict()
                raw_action = action
                row_dict['action'] = self.action_dict[raw_action]
                row_dict['reward'] = rew

                row_dict['curr_time'] = self.env.universal_clock.current_time
                for i, table in enumerate(self.env.tables):
                    row_dict[f'table_{i}_status'] = table.status
                    if table.party:
                        row_dict[f'table_{i}_party_status'] = table.party.status
                df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)
                if done:
                    running = False

            # Drawing code
            pygame.display.flip()
            self.env.render()

        pygame.quit()
        df.to_csv(f"solves/BasicHostSolve_{self.ep_num}.csv", index=False)
        return np.array(raw_obs), np.array(raw_actions)

