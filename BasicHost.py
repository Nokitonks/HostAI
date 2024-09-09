import pandas as pd
import pygame

from utils.helperFunctions import action_number_into_function
import time

class BasicHost(object):

    def __init__(self,env,ep_num):
        self.env = env
        self.ep_num = ep_num

    def calc_action(self,env):
        return 0

    def run_episode(self):
        self.env.reset()
        done = 0
        df = pd.DataFrame()
        self.action_dict = action_number_into_function(self.env.tables,self.env.unique_combos,self.env.immutable_config)
        pygame.init()
        while(not done):
            action = self.calc_action(self.env)
            obs, rew, done, _, _ = self.env.step(action)
            events = pygame.event.get()
            time.sleep(0.1)
            self.env.render()

            # Write action and reward
            row_dict = dict()
            raw_action = action
            row_dict['action'] = self.action_dict[raw_action]
            row_dict['reward'] = rew

            # Show tables
            row_dict['curr_time'] = self.env.universal_clock.current_time
            for i, table in enumerate(self.env.tables):
                row_dict[f'table_{i}_status'] = table.status
                if table.party:
                    row_dict[f'table_{i}_party_status'] = table.party.status
            df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)


        df.to_csv(f"solves/BasicHostSolve_{self.ep_num}.csv", index=False)

