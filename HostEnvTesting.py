from HostEnv import HostWorldEnv
import pygame
if __name__ == "__main__":
    config = {

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
        valid_actions = [i for i, valid in enumerate(action_mask) if valid]
        action = env.action_space.sample()

        while not action_mask[action]:
            action = env.action_space.sample()

        observation, reward, done, info = env.step(action)
        env.render()

    env.close()