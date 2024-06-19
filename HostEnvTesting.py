from HostEnv import HostWorldEnv
import pygame
import time
from ClassDefinitions import *
if __name__ == "__main__":

    table1 = Table(((4, 4), (6, 6)), 4, 8, 'regular table', [], None, TableStatus.READY)
    table2 = Table(((8, 8), (10, 10)), 4, 8, 'regular table', [], None, TableStatus.READY)
    table3 = Table(((8, 4), (10, 6)), 4, 8, 'regular table', [], None, TableStatus.READY)
    table4 = Table(((4, 8), (6, 10)), 4, 8, 'regular table', [], None, TableStatus.READY)
    config = {
        'tables' : [table1,table2,table3,table4],
        'max_party_size' : 8,
        'max_time': 320,
        'max_wait_list': 20,
        'max_reservation_list' : 5,
        'window_size' : (800, 600),
        'grid_size' : 40
    }
    env = HostWorldEnv(config)
    observation = env.reset()

    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # Use the action mask to filter valid actions
        action_mask = env.get_action_mask()

        # Check if all values are False by directly passing the list to `all`

        valid_actions = [i for i, valid in enumerate(action_mask) if valid]
        action = env.action_space.sample()
        if not any(action_mask):
            action = None
        else:
            while not action_mask[action]:
                action = env.action_space.sample()

        observation, reward, done, info = env.step(action)
        env.render(mode='human')
        time.sleep(0.1)
    env.close()