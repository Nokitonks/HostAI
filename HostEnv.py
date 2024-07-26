import numpy as np
import pygame
import csv
from helperFunctions import *
import gymnasium as gym
from gymnasium import spaces
from ClassDefinitions import *

class HostWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self,config, render_mode='human'):

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
        self.tables = config['level_settings'].tables



        # Define action space
        num_assign_actions = 4*len(self.tables)
        num_default_actions = 1

        unique_combos = []
        for table in self.tables:
            for other_table in table.combinable_with:
                combo = [table,other_table]
                combo = sorted(combo)
                if combo not in unique_combos:
                    unique_combos.append(combo)
        num_combine_actions = len(unique_combos)
        self.action_space = spaces.Discrete(num_assign_actions +
                                            num_default_actions +
                                            num_combine_actions + # For combine actions
                                            num_combine_actions   # For un-combine actions
                                            )

        self.action_handlers = {
        }
        cnt = 0
        for pools in range(4):
            for table_index in range(len(self.tables)):
                self.action_handlers[cnt] = (self.assign_party_to_table,{
                    'party_pool':pools,
                    'table_index':table_index})
                cnt += 1
        for combo in unique_combos:
            self.action_handlers[cnt] = (combo[0].combine_with,{
                'other_table': combo[1]
            })
            cnt += 1
            self.action_handlers[cnt] = (combo[0].uncombine_with,{
                'other_table': combo[1]
            })
            cnt += 1
        self.unique_combos = unique_combos
        self.action_handlers[cnt] = (self.default_action,{})


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

        self.observation_space = self.create_observation_space(len(self.tables), config['level_settings'].max_party_size
                                                               , config['level_settings'].max_time,
                                                               config['level_settings'].max_wait_list,
                                                               config['level_settings'].max_res_list)
        self.state = None
        self.reset()

        # Setup Pygame
        pygame.init()
        self.screen = pygame.display.set_mode(self.config['window_size'])
        pygame.display.set_caption("Restaurant Environment")
        self.font = pygame.font.SysFont(None, 24)

    def create_observation_space(self,num_tables, max_party_size, max_time, max_wait_list, max_reservation_list):

        reservation_space = spaces.Dict({
            'time_of_reservation': spaces.Discrete(max_time),
            'reservation_status': spaces.Discrete(len(ReservationStatus))
        })

        # Define space for party info
        party_space = spaces.Dict({
            'size': spaces.Discrete(max_party_size + 1),
            'status': spaces.Discrete(len(PartyStatus)),
            'arrival_time': spaces.Discrete(max_time),
            'reservation': reservation_space
        })


        # Define the full observation space
        observation_space = spaces.Dict({
            'tables': spaces.Dict({
            f'table_{i}': spaces.Dict({
                'status': spaces.Discrete(len(TableStatus)),
                'party': party_space,
                'table_combined_size': spaces.Discrete(max_party_size + 1)
            }) for i in range(num_tables)
                }),
            'waitlist': spaces.Tuple([party_space for _ in range(max_wait_list)]),
            'reservation_list': spaces.Tuple([reservation_space for _ in range(max_reservation_list)]),
            'current_time': spaces.Discrete(max_time*2)
        })

        return observation_space

    def step(self,action):
        reward = 0
        done = False
        # Default action means action will be None

        # Call the handler function for the action with parameters
        handler, params = self.action_handlers[action]
        reward, done = handler(**params)

        # For debugging etc purposes
        info = {}
        self.universal_clock.update()
        reward += self.update_tables()
        self.update_arrivals()
        reward += self.update_parties()
        if self.universal_clock.current_time >= self.end_time:
            # Game Over
            if len(self.waitlist) == 0:
                for table in self.tables:
                    if table.status != TableStatus.READY and table.status != TableStatus.COMBINED:
                        done = False
                        break
                    done = True
            pass

        return self._get_observation(), reward, done, False, info

    def render(self, mode='human'):

        if mode == 'human':
            self.screen.fill(self.colors['BLACK'])

            # Draw party section
            pygame.draw.rect(self.screen, self.colors['BLUE'], self.PARTY_SECTION, 2)
            self.screen.set_clip(self.PARTY_SECTION)
            self.screen.set_clip(None)
            self.draw_parties((self.PARTY_SECTION.x, self.PARTY_SECTION.y))

            # Draw table section
            pygame.draw.rect(self.screen, self.colors['BLUE'], self.TABLE_SECTION, 2)
            table_offset = (self.TABLE_SECTION.x, self.TABLE_SECTION.y)
            self.screen.set_clip(self.TABLE_SECTION)
            self.draw_grid(table_offset)

            for table in self.tables:
                self.draw_table(table, table_offset)
            self.screen.set_clip(None)


            # Draw universal clock
            self.draw_universal_clock(self.universal_clock)
            pygame.display.flip()


    def reset(self,seed=None, options=None):
        super().reset(seed=seed)
        self.colors = {

            'WHITE' : (255, 255, 255),
            'RED' : (255, 0, 0),
            'ORANGE' : (255, 165, 0),
            'PURPLE' : (255, 0, 255),
            'BLUE' : (0, 0, 255),
            'BLACK' : (0, 0, 0),
            'GREEN' : (0, 255, 0),
            'GRAY' : (200, 200, 200),
            'LIGHT_GRAY' : (220, 220, 220)
        }

        self.SCREEN_WIDTH = 800
        self.SCREEN_HEIGHT = 600
        self.start_time = 0
        self.end_time = self.config['max_time']
        self.window_size = self.config['window_size'] # The size of the Pygame Window
        self.GRID_SIZE = self.config['grid_size']  # Size of the grid cells
        self.ROWS = self.window_size[1] // self.GRID_SIZE
        self.COLS = self.window_size[0] // self.GRID_SIZE

        self.party_pool_manager = PartyPoolManager(4,[2,4,6,8])

        self.TABLE_SECTION = pygame.Rect(self.window_size[0] // 4, 0, self.window_size[0], self.window_size[1])
        self.PARTY_SECTION = pygame.Rect(0, 0, self.window_size[0] // 4, self.window_size[1])

        # Reset tables that were combined
        for table in self.tables:
            table.reset()

        # Misc variables for GUI sizes
        self.PARTY_RECT_SIZE_Y = self.window_size[1] // 10
        self.PARTY_PADDING_X = self.PARTY_RECT_SIZE_Y // 4


        self.show_reservations = False
        self.waitlist = []
        # Parties are added to this list once they have finished eating and left
        self.served = []

        # Beginning of game we read in the reservations and walk-ins for the evening
        reservations = self.read_reservations('reservation_files/reservations0.csv')
        self.reservations = sorted(reservations, key=lambda  x: x.reservation_time)

        self.walk_ins = self.read_walk_ins('walk_in_files/walk_ins.csv')

        # Convert reservations to have their own party object tied to the reservation
        for reservation in reservations:
            party = Party(reservation.party_name,reservation.num_people, reservation, None,
                          PartyStatus.NONE, reservation.reservation_time,None, None,
                          None,reservation.dine_time)
            self.walk_ins.append(party)

        self.universal_clock = UniversalClock(self.start_time,1)

        return self._get_observation(),{}

    def read_reservations(self,csv_file):
        reservations = []
        with open(csv_file, mode='r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                reservation = Reservation(row['name'], int(row['num_people']), row['reservation_time'], "",
                                          "", ReservationStatus[row['status']],int(row['dine_time']))
                reservations.append(reservation)
        return reservations

    # Function to read walk-ins from CSV
    def read_walk_ins(self,csv_file):
        walk_ins = []
        with open(csv_file, mode='r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                party = Party(row['name'],int(row['num_people']), None, None, PartyStatus.NONE,
                              row['arrival_time'], None, None, None, int(row['dine_time']))

                walk_ins.append(party)
        return walk_ins

    def default_action(self):
        return 0, False

    def assign_party_to_table(self,party_pool,table_index):
        try:
            party = self.party_pool_manager.pools[party_pool].get_party()
        except:
            return 0, False
        table = self.tables[table_index]
        table.assign_party(party)
        self.waitlist.remove(party)
        party.sat_time = self.universal_clock.current_time

        reward = party.num_people
        # Needs to return a reward and a done
        return reward, False

    def get_action_mask(self):

        # Generate our action mask
        action_mask = [1] * (self.action_space.n )

        cnt = 0
        val = 1
        for pools in range(4):
            pool = self.party_pool_manager.pools[pools]
            for table_index in range(len(self.tables)):
                table = self.tables[table_index]
                val = 1
                if len(pool) == 0:
                    val = 0

                # Invalid if party size is not going to fit at the table
                if (table.get_combined_size([]) < pool.party_size):
                    val = 0

                # Invalid if the table is occupied
                if (table.status != TableStatus.READY):
                    val = 0

                action_mask[cnt] = val
                cnt += 1

        for combo in self.unique_combos:
            # This is the combine action between combo[0] and combo[1]
            if combo[0].can_combine_with(combo[1]):
                val = 1
            else:
                val = 0
            action_mask[cnt] = val
            cnt += 1

            # This is the un-combine action between combo[0] and combo[1]
            # Only allow to uncombine with the node table of a combined set (Is the table that is status ready)
            if(combo[0].can_uncombine_with(combo[1])):
                val = 1
            else:
                val = 0

            action_mask[cnt] = val
            cnt += 1

        return np.array(action_mask,dtype=np.int8)

    def update_parties(self):
        """
        add to the waiting time of all parties currently on wait list
        """

        for party in self.waitlist:
            if self.universal_clock.current_time - (int(party.arrival_time)) >= self.config['wait_tolerance']:
                self.waitlist.remove(party)
                self.party_pool_manager.find_pool_for_size(party.num_people).remove(party)
                return -party.num_people
        return 0

    def update_arrivals(self):

        for party in self.walk_ins:
            clock_min = self.universal_clock.current_time
            arrival_min = int(party.arrival_time)
            if arrival_min <= clock_min:
                self.waitlist.append(party)
                for pool in self.party_pool_manager.pools:
                    if pool.party_size >= party.num_people:
                        pool.add(party)
                        break
                party.status = PartyStatus.ARRIVED
                self.walk_ins.remove(party)

    def update_tables(self):
        reward = 0
        for table in self.tables:
            if table.party:
                time_seated = self.universal_clock.current_time - table.party.sat_time
                if time_seated >= table.party.dine_time:

                    # Update our score
                    reward = self.score_party(table.party)

                    # Add party to served list and set table to dirty
                    table.party.status = PartyStatus.LEFT
                    self.served.append(table.party)
                    table.remove_party()
                    table.status = TableStatus.DIRTY

            elif table.status == TableStatus.DIRTY:
                if table.clean_progress >= table.clean_time:
                    table.clean_progress = 0
                    table.status = TableStatus.READY
                else:
                    table.clean_progress += 1
        return reward

    def score_party(self,party):
        """
        :param party: the party to score

        This is where we will implement how our scoring system works (Very important for eventual AI)
        """
        return 0

    def draw_grid(self,offset):
        for row in range(self.ROWS):
            for col in range(self.COLS):
                rect = pygame.Rect(col * self.GRID_SIZE + offset[0], row * self.GRID_SIZE + offset[1], self.GRID_SIZE, self.GRID_SIZE)
                pygame.draw.rect(self.screen, self.colors['GRAY'], rect, 1)

    def draw_parties(self,offset):
        start_y = offset[1]
        for i, party in enumerate(self.waitlist):
            party_rect = pygame.Rect(offset[0], start_y + i * self.PARTY_RECT_SIZE_Y, self.PARTY_SECTION.width, self.PARTY_RECT_SIZE_Y)
            if self.PARTY_SECTION.colliderect(party_rect):  # Only draw if within the PARTY_SECTION
                if party.status is PartyStatus.SEATED:
                    pygame.draw.rect(self.screen, self.colors['ORANGE'] , party_rect)
                else:
                    pygame.draw.rect(self.screen, self.colors['WHITE'], party_rect)
                pygame.draw.rect(self.screen, self.colors['BLACK'], party_rect, 1)
                party_text = str(party)
                max_font = get_max_font_size(party_text, self.PARTY_SECTION.width - self.PARTY_PADDING_X, self.PARTY_RECT_SIZE_Y, 36)
                text_surface = max_font.render(party_text, True, self.colors['BLACK'])
                text_rect = text_surface.get_rect(center=party_rect.center)
                self.screen.blit(text_surface, text_rect.topleft)

    def draw_universal_clock(self,clock):
        clock_text = clock.get_time_str()
        clock_surface = self.font.render(clock_text, True, self.colors['GREEN'])
        clock_rect = clock_surface.get_rect(center=(self.SCREEN_WIDTH // 2, 30))
        self.screen.blit(clock_surface, clock_rect.topleft)

    def draw_table(self,table,offset):
        x1, y1 = table.footprint[0]
        x2, y2 = table.footprint[1]
        table_rect = pygame.Rect(x1 * self.GRID_SIZE + offset[0], y1 * self.GRID_SIZE + offset[1], (x2 - x1) * self.GRID_SIZE, (y2 - y1) * self.GRID_SIZE)
        if table.status == TableStatus.READY :
            table_color = self.colors['GREEN']
            pygame.draw.rect(self.screen, table_color, table_rect)
            return
        elif table.status == TableStatus.COMBINED:
            table_color = self.colors['ORANGE']
            pygame.draw.rect(self.screen, table_color, table_rect)
            return
        elif table.status == TableStatus.DIRTY:
            table_color = self.colors['RED']
            pygame.draw.rect(self.screen, table_color, table_rect)

            # Draw the foreground rectangle based on progress
            foreground_width = (table_rect.width * table.clean_progress) / table.clean_time
            pygame.draw.rect(self.screen, self.colors['GREEN'], (table_rect.x, table_rect.y, foreground_width, table_rect.height))

            return
        else:
            #If we are here it means that table.party is not null
            if table.party is None: raise ValueError("ERROR: Table is occupied but no Party Object ")
            table_color = self.colors['GRAY']


        #  table.party exists from here on out:-->

        pygame.draw.rect(self.screen, table_color, table_rect)
        #Find status of the party at the table

        if table.party.status == PartyStatus.SEATED:
            party_color = self.colors['GREEN']
        elif table.party.status == PartyStatus.APPS:
            party_color = self.colors['ORANGE']
        elif table.party.status == PartyStatus.MAIN_COURSE:
            party_color = self.colors['RED']
        elif table.party.status == PartyStatus.DESSERT:
            party_color = self.colors['PURPLE']
        elif table.party.status == PartyStatus.CHECK_DROPPED:
            party_color = self.colors['BLUE']

        party_rect = create_scaled_rect(table_rect,60)
        pygame.draw.rect(self.screen, party_color, party_rect)

        text = f"{(self.universal_clock.current_time - table.party.sat_time)}"
        text_surface = self.font.render(text, True, self.colors['BLACK'])
        text_rect = text_surface.get_rect(center=party_rect.center)
        self.screen.blit(text_surface, text_rect)

    def _get_observation(self):
        dummy_party = {
            'size': 0,
            'status': 0,
            'arrival_time': 0,
            'reservation': {
                'time_of_reservation': 0,
                'reservation_status': 0
            }
        }
        dummy_reservation = {
            'time_of_reservation': 0,
            'reservation_status':0
        }
        # Create the observation dictionary based on the current state
        observation = {
            'tables': {},
            'waitlist': [],
            'reservation_list': [],
            'current_time': int(self.universal_clock.current_time)
        }

        for table_idx, table in enumerate(self.tables):
            table_observation = {
                'status': table.status.value,
                'table_combined_size': table.get_combined_size([]),
                'party': None
            }
            if table.party:
                party = table.party
                table_observation['party'] = {
                    'size': party.num_people,
                    'status': party.status.value,
                    'arrival_time': int(party.arrival_time),
                    'reservation': {
                        'time_of_reservation': 0,
                        'reservation_status': 0,
                    }
                }
                if party.reservation:
                    table_observation['party']['reservation']['time_of_reservation'] = int(party.reservation.reservation_time)
                    table_observation['party']['reservation']['reservation_status'] = party.reservation.status.value
            else:
                table_observation['party'] = dummy_party

            observation['tables'][f'table_{table_idx}'] = table_observation

        for party in self.waitlist:
            observation['waitlist'].append({
                'size': party.num_people,
                'status': party.status.value,
                'arrival_time': int(party.arrival_time),
                'reservation': {
                    'time_of_reservation': 0,
                    'reservation_status': 0
                }
             })
            if party.reservation:
                observation['waitlist'][-1]['reservation']['time_of_reservation'] = int(party.reservation.reservation_time)
                observation['waitlist'][-1]['reservation']['reservation_status'] = party.reservation.status.value
        while len(observation['waitlist']) < self.config['max_wait_list']:
            observation['waitlist'].append(dummy_party)
        for reservation in self.reservations:
            observation['reservation_list'].append({
                'time_of_reservation': int(reservation.reservation_time),
                'reservation_status': reservation.status.value
            })
        while len(observation['reservation_list']) < self.config['max_reservation_list']:
            observation['reservation_list'].append(dummy_reservation)

        return observation