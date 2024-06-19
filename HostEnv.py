import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces
from ClassDefinitions import *
def create_observation_space(num_tables, max_party_size, max_time,max_wait_list,max_reservation_list):
    # Define space for party info
    party_space = spaces.Dict({
        'size': spaces.Discrete(max_party_size + 1),
        'status': spaces.Discrete(len(PartyStatus)),
        'arrival_time': spaces.Discrete(max_time),
        'sat_time': spaces.Discrete(max_time),
        'leave_time': spaces.Discrete(max_time),
        'reservation': spaces.Dict({
            'time_of_reservation': spaces.Discrete(max_time),
            'reservation_status': spaces.Discrete(len(ReservationStatus))
        })
    })

    # Define space for all tables
    tables_space = spaces.Dict({
        f'table_{i}': spaces.Dict({
            'status': spaces.Discrete(len(TableStatus)),
            'party': party_space
        }) for i in range(num_tables)
    })

    # Define the full observation space
    observation_space = spaces.Dict({
        'tables': tables_space,
        'waitlist': spaces.Tuple([party_space for _ in range(max_wait_list)]),
        'reservation_list': spaces.Tuple([party_space for _ in range(max_reservation_list)])
    })

    return observation_space

class HostWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self,config, render_mode=None):

        super(HostWorldEnv, self).__init__()
        self.config = config
        """
        Agent section
        """

        """
        Valid actions is computed as follows. There are 4 different "Bins" of possible parties to seat. Each bin 
        represents a size (2,4,6,8). each bin will automatically give the party to seat when get_party is called 
        on that bin. 
        
        There is n tables that exist in a configuration of the restaurant so the size of valid actions will be 
        of length 4*n
        
        **NOTE** 
        At all times some of these actions will be unavailble for different reasons (no 2 person parties waiting,
         full table etc.)
        """
        # The list containing tables objects to setup the grid of tables
        self.tables = config['tables']

        # Define action space
        self.action_space = spaces.Discrete(4*len(self.tables))

        self.action_handlers = {
        }
        cnt = 0
        for pools in range(4):
            for table_index in len(self.tables):
                self.action_handlers[cnt] = (self.assign_party_to_table,{
                    'party_pool':pools,
                    'table_index':table_index})
                cnt += 1


        """
        Our observation space is complex and can carry a lot of information about the sate of the game 
        the most important things to consider are as follows:
        - Status of all the tables (Dirty, clean, occupied)
        - Of the occupied tables (the party that is sitting at what table)
        - party info for table and for waitlist
        - reservation list 
        - wait list
        
        - Party observation is as follows: 
            Size -> Discrete
            Status -> Discrete
            Arrival Time -> discrete
            Sat time -> discrete
            Leave time -> discrete  
            Resrvation -> To Reservation Desc
                Time of Reservation -> Discrete
                Reservation Status -> Discrete
        """

        self.observation_space = create_observation_space(len(self.tables), config['max_party_size'], config['max_time'], config['max_wait_list'],config['max_reservation_list'])
        self.state = None
        self.reset()

        # Setup Pygame
        pygame.init()
        self.screen = pygame.display.set_mode(self.config['window_size'])
        pygame.display.set_caption("Restaurant Environment")
        self.font = pygame.font.SysFont(None, 24)

    def step(self,action):

        # Call the handler function for the action with parameters
        if action in self.action_handlers:
            handler, params = self.action_handlers[action]
            reward, done = handler(**params)
        else:
            print(f"Unknown acction: {action}")

        # For debugging etc purposes
        info = {}

        return self.state, reward, done, info

    def render(self):
        pass

    def reset(self):

        self.start_time = 0
        self.end_time = self.config['max_time']
        self.window_size = self.config['window_size'] # The size of the Pygame Window
        self.grid_size = self.config['grid_size']  # Size of the grid cells
        self.ROWS = self.window_size // self.grid_size
        self.COLS = self.window_size // self.grid_size


        self.TABLE_SECTION = pygame.Rect(self.window_size // 4, 0, self.window_size, self.window_size)
        self.PARTY_SECTION = pygame.Rect(0, 0, self.window_size // 4, self.window_size)

        # Misc variables for GUI sizes
        self.PARTY_RECT_SIZE_Y = self.window_size // 10
        self.PARTY_PADDING_X = self.PARTY_RECT_SIZE_Y // 4


        self.show_reservations = False
        self.waitlist = []
        # Parties are added to this list once they have finished eating and left
        self.served = []

        # Beginning of game we read in the reservations and walk-ins for the evening
        reservations = self.read_reservations('reservations.csv')
        self.reservations = sorted(reservations, key=lambda  x: x.reservation_time)

        self.walk_ins = self.read_walk_ins('walk_ins.csv')

        # Convert reservations to have their own party object tied to the reservation
        for reservation in reservations:
            party = Party(reservation.party_name,reservation.num_people, reservation, None,
                          PartyStatus.NONE, reservation.reservation_time,None, None,
                          None,reservation.num_people*10)
            self.walk_ins.append(party)

        self.universal_clock = UniversalClock(self.start_time,5)

        pass

    def assign_party_to_table(self,party_pool,table_index):
        party = self.party_pools[party_pool].pop()
        table = self.tables[table_index]
        table.assign_party(party)
        self.waitlist.remove(party)
        party.sat_time = self.universal_clock.current_time

    def get_action_mask(self):

        # Generate our action mask
        action_mask = [True] * self.action_space.n

        cnt = 0
        val = True
        for pools in range(4):
            pool = self.party_pools[pools]
            if pool.empty():
                val = False
            for table_index in len(self.tables):
                table = self.tables[table_index]

                # Invalid if party size is not going to fit at the table
                if (table.size_px < pool.party_size):
                    val = False

                # Invalid if the table is occupied
                if (table.status != TableStatus.READY):
                    val = False

                action_mask[cnt] = val
                cnt += 1

        return action_mask
