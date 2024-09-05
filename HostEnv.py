import csv
from utils.helperFunctions import *
import gymnasium as gym
from gymnasium import spaces
from ClassDefinitions import *
import logging
from ModelingFunctions import get_busyness
from gymnasium.spaces.utils import flatten

class HostWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self,immutable_config,mutable_config=None):

        super(HostWorldEnv, self).__init__()
        self.immutable_config = immutable_config
        self.mutable_config = mutable_config
        logging.basicConfig(filename="logs/Model_testing.txt",
                                filemode='w',
                                format='%(asctime)s %(levelname)s-%(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S',
                                level=logging.INFO)
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
        self.tables = self.immutable_config['tables']

        # Define action space
        num_assign_actions = 4*len(self.tables)
        quote_wait_times = 0
        for time in range(immutable_config['wait_quote_min'], immutable_config['wait_quote_max'],
                          immutable_config['wait_quote_step']):
            quote_wait_times += 1
        num_quote_wait_actions = 4
        num_deny_actions = 4
        num_default_actions = 1

        unique_combos = []
        for table in self.tables:
            for other_table in table.combinable_with:
                combo = [table,other_table]
                combo = sorted(combo)
                if combo not in unique_combos:
                    unique_combos.append(combo)
        num_combine_actions = len(unique_combos)
        self.action_space = spaces.Discrete((num_assign_actions * 2) +
                                            (quote_wait_times * num_quote_wait_actions) +
                                            num_deny_actions +
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
                    'rez':True,
                    'party_pool':pools,
                    'table_index':table_index})
                cnt += 1
        for pools in range(4):
            for table_index in range(len(self.tables)):
                self.action_handlers[cnt] = (self.assign_party_to_table,{
                    'rez':False,
                    'party_pool':pools,
                    'table_index':table_index})
                cnt += 1
        for pools in range(4):
            for time in range (immutable_config['wait_quote_min'], immutable_config['wait_quote_max'],immutable_config['wait_quote_step']):
                self.action_handlers[cnt] = (self.quote_wait_time_to_pary,{
                    'time': time,
                    'party_pool':pools})
                cnt += 1
        for pools in range(4):
            self.action_handlers[cnt] = (self.deny_party, {
                'party_pool': pools})
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
        print(f"cnt is {cnt}\n")

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

        self.observation_space = self.create_observation_space(len(self.tables), self.immutable_config['max_party_size']
                                                               , self.immutable_config['max_time'],
                                                               self.immutable_config['max_wait_list'],
                                                               self.immutable_config['max_res_list'])
        self.state = None

        """
        We want to make sure that every episode has a set amount of steps in order for our sequence matching algorithm to work
        """
        self.n_steps = self.immutable_config['n_steps']
        self.reset()

        # Setup Pygame
        pygame.init()
        self.screen = pygame.display.set_mode(self.immutable_config['window_size'])
        pygame.display.set_caption("Restaurant Environment")
        self.font = pygame.font.SysFont(None, 24)

    def set_mutable_config(self,config):
        self.mutable_config = config

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
        self.flattened_mapping = create_flattened_mapping(observation_space)
        print(self.flattened_mapping)
        return observation_space

    def step(self,action):
        reward = 0
        done = False
        self.n_steps -= 1
        # Default action means action will be None

        # Call the handler function for the action with parameters
        if self.mutable_config['log_dir'] != '':
            logging.info(f"Taking action {self.action_handlers[action]}")
        handler, params = self.action_handlers[action]
        reward, done = handler(**params)
        if (reward == -1):
            for i in range(1):
                self.advance_time()
            reward = 0
        # For debugging etc purposes
        info = {}
        reward += self.update_tables()
        self.update_arrivals()
        reward += self.update_parties()
        # We are trying just to finish after a number of steps
        """
        if self.universal_clock.current_time >= self.end_time:
            # Game Over
            # Need to make sure we have reached the total actions
            if len(self.waitlist) == 0:
                for table in self.tables:
                    if table.status != TableStatus.READY and table.status != TableStatus.COMBINED:
                        done = False
                        break
                    done = True
            pass
        """
        #Override for RUDDER PRACTICE- >>>>
        #reward = 0

        if self.n_steps == 1:
            total_num_served = 0
            for party in self.served:
                total_num_served += party.num_people
            #reward = total_num_served
        if self.n_steps ==0:
            done = True

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
        self.end_time = self.mutable_config['end_time']
        self.window_size = self.immutable_config['window_size'] # The size of the Pygame Window
        self.GRID_SIZE = self.immutable_config['grid_size']  # Size of the grid cells
        self.ROWS = self.window_size[1] // self.GRID_SIZE
        self.COLS = self.window_size[0] // self.GRID_SIZE

        self.rez_party_pool_manager = PartyPoolManager(4,[2,4,6,8])
        self.walkin_party_pool_manager = PartyPoolManager(4,[2,4,6,8])
        self.clean_time = self.mutable_config['clean_time']

        self.TABLE_SECTION = pygame.Rect(self.window_size[0] // 4, 0, self.window_size[0], self.window_size[1])
        self.PARTY_SECTION = pygame.Rect(0, 0, self.window_size[0] // 4, self.window_size[1])

        # Reset tables that were combined
        for table in self.tables:
            table.reset()

        #Server sections
        self.num_servers = self.mutable_config['num_servers']
        self.server_sections = self.mutable_config['server_sections']
        self.server_busyness = dict()
        for server in range(self.num_servers):
            self.server_busyness[server] = 0


        # Misc variables for GUI sizes
        self.PARTY_RECT_SIZE_Y = self.window_size[1] // 10
        self.PARTY_PADDING_X = self.PARTY_RECT_SIZE_Y // 4


        self.show_reservations = False
        self.waitlist = []
        # Parties are added to this list once they have finished eating and left
        self.served = []
        self.n_steps = self.immutable_config['n_steps']

        # Beginning of game we read in the reservations and walk-ins for the evening
        reservations = self.read_reservations(self.mutable_config['reservations_path'])
        self.reservations = sorted(reservations, key=lambda  x: x.reservation_time)

        self.walk_ins = self.read_walk_ins(self.mutable_config['walk_ins_path'])

        # Convert reservations to have their own party object tied to the reservation
        for reservation in reservations:
            party = Party(reservation.party_name,reservation.num_people, reservation, None,None,
                          PartyStatus.NONE, reservation.reservation_time,None, None,
                          10,reservation.dine_time,reservation.meal_split)
            self.walk_ins.append(party)

        self.universal_clock = UniversalClock(self.start_time,1)

        return self._get_observation(),{}

    def read_reservations(self,csv_file):
        reservations = []
        with open(csv_file, mode='r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                reservation = Reservation(row['name'], int(row['num_people']), int(row['reservation_time']), "",
                                          "", ReservationStatus[row['status']],int(row['dine_time']),row['meal_split'])
                reservations.append(reservation)
        return reservations

    # Function to read walk-ins from CSV
    def read_walk_ins(self,csv_file):
        walk_ins = []
        with open(csv_file, mode='r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                party = Party(row['name'],int(row['num_people']), None, None, None, PartyStatus.NONE,
                              row['arrival_time'], None, None, 10, int(row['dine_time']),row['meal_split'])

                walk_ins.append(party)
        return walk_ins

    def default_action(self):

        self.advance_time()
        reward = 0
        #Advancing the clock when there is an open table is a negative reward
        for party in self.waitlist:
            for table in self.tables:
                if table.status == TableStatus.READY:
                    if party.num_people <= table.get_combined_size([]):
                        reward = -1
                        return reward, False

        #Advancing the clock when there is no possible way to seat people is positive task
        reward = 0

        return reward, False

    def deny_party(self,party_pool):
        party = self.walkin_party_pool_manager.pools[party_pool].get_party()
        self.waitlist.remove(party)
        return -party.num_people / 2 ,False

    def quote_wait_time_to_pary(self,party_pool,time):
        """
        Gives a walk_in party a wait time and takes them off the waitlist. Party will return at quoted time +- tolerance
        The time that we quote is always our guessed wait_time for that size party which the model keeps track of
        :param party_pool: The pool to which this party belongs
        :param time: The quoted time for the party to return in, ranging from 10min - 90min
        :return: void
        """
        party = self.walkin_party_pool_manager.pools[party_pool].get_party()
        #Take our party out of the waitlist
        self.waitlist.remove(party)
        #Now we create a new reservation that is essentially the time_now + quoted time
        new_time = int(self.universal_clock.current_time) + int(time)
        new_rez = Reservation(party.name,party.num_people,new_time,"","",ReservationStatus.WALK_IN,party.dine_time,party.meal_split)

        #Configure our party by updating arrival time and tie our reservation to the party object
        party.reservation = new_rez
        party.arrival_time = new_time

        self.reservations.append(new_rez)
        self.reservations = sorted(self.reservations, key=lambda  x: x.reservation_time)

        self.walk_ins.append(party)

        #Need to add better reward here probably
        return 0, False

    def assign_party_to_table(self,rez,party_pool,table_index):
        if rez:
            try:
                party = self.rez_party_pool_manager.pools[party_pool].get_party()
            except:
                return 0, False
        else:
            try:
                party = self.walkin_party_pool_manager.pools[party_pool].get_party()
            except:
                return 0, False
        table = self.tables[table_index]
        table.assign_party(party)
        self.waitlist.remove(party)
        party.sat_time = self.universal_clock.current_time

        #Get the server for that table and make them a little busyier
        server_num = self.server_sections[str(table.number)]
        self.server_busyness[server_num] += party.num_people

        reward = party.num_people
        happiness_modifier = (party.happiness / 10)
        if self.mutable_config['log_dir'] != "":
            logging.info(f"Party {party.name} of size {party.num_people} has been seated at t={self.universal_clock.current_time}\n")
        # Needs to return a reward and a done
        return reward * happiness_modifier, False

    def get_action_mask(self):

        # Generate our action mask
        action_mask = [1] * (self.action_space.n )

        cnt = 0
        val = 1
        for pools in range(4):
            pool = self.rez_party_pool_manager.pools[pools]
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

        for pools in range(4):
            pool = self.walkin_party_pool_manager.pools[pools]
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

        for pools in range(4):
            for time in range (self.immutable_config['wait_quote_min'], self.immutable_config['wait_quote_max'],self.immutable_config['wait_quote_step']):
                pool = self.walkin_party_pool_manager.pools[pools]
                val = 1
                if len(pool) == 0:
                    val = 0
                if int(self.universal_clock.current_time) + time >= self.mutable_config['end_time']:
                    val = 0
                action_mask[cnt] = val
                cnt += 1
        for pools in range(4):
            pool = self.walkin_party_pool_manager.pools[pools]
            val = 1
            if len(pool) == 0:
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
    def advance_time(self):
        #Update some happinesses of people waiting
        for party in self.waitlist:
            #Only start losing happiness after wait_tolerance minutes of waiting
            if (int(self.universal_clock.current_time) > (int(party.arrival_time) + int(self.mutable_config['wait_tolerance']))):
                continue
            else:
                if party.reservation:
                    # Parties with reservations will lose happiness more
                    party.happiness -= 2.5
                else:
                    party.happiness -= 0.5

        #Make our servers a little less busy
        for server in self.server_busyness.keys():
            if self.server_busyness[server] > 0:
                self.server_busyness[server] -= 1

        self.universal_clock.update()

    def update_parties(self):
        """
        add to the waiting time of all parties currently on wait list
        """
        for party in self.waitlist:
            if party.happiness <= 0:
                self.waitlist.remove(party)
                if party.reservation:
                    self.rez_party_pool_manager.find_pool_for_size(party.num_people).remove(party)
                    if self.mutable_config['log_dir'] != "":
                        logging.info(
                            f"Party {party.name} of size {party.num_people} Left at {self.universal_clock.current_time}\n")
                    return -party.num_people * 4
                else:
                    self.walkin_party_pool_manager.find_pool_for_size(party.num_people).remove(party)
                    if self.mutable_config['log_dir'] != "":
                        logging.info(
                            f"Party {party.name} of size {party.num_people} Left at {self.universal_clock.current_time}\n")
                    return -party.num_people
        return 0

    def update_arrivals(self):

        for party in self.walk_ins:

            # Remove parties that have a canceled reservation
            if party.reservation:
                if party.reservation.status == ReservationStatus.CANCELED:
                    self.walk_ins.remove(party)
                    continue

            clock_min = self.universal_clock.current_time
            arrival_min = int(party.arrival_time)
            if arrival_min <= clock_min:
                self.waitlist.append(party)

                # We search for the pool to add them to
                if party.reservation:
                    for pool in self.rez_party_pool_manager.pools:
                        if pool.party_size >= party.num_people:
                            pool.add(party)
                            break
                else:
                    for pool in self.walkin_party_pool_manager.pools:
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
                else:
                    server_scaler = translate(self.server_busyness[self.server_sections[str(table.number)]],
                                              0,20,1,1.5)
                    _ = table.party.update_seated_status(time_seated,get_busyness(self),server_scaler)

            elif table.status == TableStatus.DIRTY:
                if table.clean_progress >= self.clean_time[table.get_combined_size([])]:
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
        while len(observation['waitlist']) < self.immutable_config['max_wait_list']:
            observation['waitlist'].append(dummy_party)
        for reservation in self.reservations:
            observation['reservation_list'].append({
                'time_of_reservation': int(reservation.reservation_time),
                'reservation_status': reservation.status.value
            })
        while len(observation['reservation_list']) < self.immutable_config['max_res_list']:
            observation['reservation_list'].append(dummy_reservation)
        return observation


    def get_state_shape(self):
        return [flatten(self.observation_space,self.observation_space.sample()).size]
    def get_n_actions(self):
        return [self.action_space.n]
def mask_fn(env: HostWorldEnv) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.get_action_mask()
