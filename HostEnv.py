import csv
from utils.helperFunctions import *
import gymnasium as gym
from gymnasium import spaces
from ClassDefinitions import *
import logging
from ModelingFunctions import get_busyness
from gymnasium.spaces.utils import flatten
hard_code_rew = [ 1.0499e+01,  5.6138e+00,  5.8108e+00,  5.8805e+00,  8.5557e-01,
         4.4992e+00, -5.6754e+00,  1.9406e+00,  1.7596e+00, -1.6481e-01,
         5.9192e-02,  7.2112e-02, -1.0781e+00,  5.0928e-01, -5.4420e-01,
        -4.3247e-01,  6.0848e-01, -2.2424e-01,  3.0479e-01, -2.1502e-01,
        -1.3480e-02, -2.9192e-02,  3.7632e-01, -2.5408e-01,  2.0873e-01,
        -1.6766e-01,  4.4328e-01, -1.5010e-01,  2.8058e-01, -6.9451e-02,
         4.9992e-01, -8.9021e-02, -1.3998e+00,  1.2097e-01, -5.8286e-01,
         1.0435e+00, -4.1428e-01,  1.1636e-01, -6.1096e-01,  2.4278e-01,
        -3.6461e-01,  7.3043e-01,  1.3292e-01, -2.2069e-01,  2.0709e-01,
        -2.0533e-01,  3.2526e-01, -1.5541e-01,  1.6565e-01, -1.6276e-01,
         1.2570e-01, -6.4521e-02, -2.7650e-01,  2.1421e-01,  8.6858e-02,
        -1.0430e-01,  5.7223e-02, -1.7776e-01,  1.5120e-01, -1.4219e-01,
         1.2851e-03, -3.4630e-02,  8.1086e-02, -4.0774e-02,  4.8585e-02,
        -3.6287e-03,  9.7382e-02, -7.8659e-02,  6.8550e-02, -1.7913e-01,
         4.0960e-02,  3.1409e-02, -4.8864e-02,  5.5387e-02,  1.5659e-02,
         3.9082e-02,  3.4313e-02, -2.0388e-01,  1.2652e-01, -5.9516e-02,
         3.6662e-02, -4.3304e-02, -6.3896e-04,  1.0293e-01,  2.9039e-03,
        -4.0541e-02,  9.3007e-02, -1.3080e-02, -1.2519e-01,  2.0061e-01,
        -3.7539e-02,  8.1422e-02, -4.3015e-02,  4.4906e-02,  2.5008e-02,
         3.1915e-02, -8.7311e-02,  3.2592e-03, -7.3719e-03]
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
                'table_combined_size': spaces.Discrete(25)
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
        self.score += reward
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
                self.draw_background_table(table, table_offset)
            for table in self.tables:
                self.draw_table(table, table_offset)
            self.screen.set_clip(None)


            # Draw universal clock
            self.draw_universal_clock(self.universal_clock)
            self.draw_score()
            pygame.display.flip()


    def reset(self,seed=None, options=None):
        super().reset(seed=seed)
        self.colors = {

            'WHITE' : (255, 255, 255),
            'RED' : (255, 0, 0),
            'ORANGE' : (255, 165, 0),
            'LIGHT_ORANGE' : (255, 205, 50),
            'PURPLE' : (255, 0, 255),
            'BLUE' : (0, 0, 255),
            'LIGHT_BLUE' : (50, 50, 255),
            'BLACK' : (0, 0, 0),
            'GREEN' : (0, 255, 0),
            'LIGHT_GREEN' : (0, 205, 0),
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

        self.score = 0
        self.selected_button = None
        self.buttons = []
        self.selected_table = None
        self.selected_wait = None
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
    def handle_events(self,events,function_dict):
        """
        We use this function in the human part of the game inorder to let the user "play"
        :param events: pygame events
        :function_dict: maps our actions in string form into the actual values that the model uses to learn
        :return: returns -1 if a non env changing action was taken, otherwise returns the action value
        of the action that was taken
        """


        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                #Clicked on a table inside the table section
                if self.TABLE_SECTION.collidepoint(pos):
                    for table in self.tables:
                        x1, y1 = table.footprint[0]
                        x2, y2 = table.footprint[1]
                        table_rect = pygame.Rect(x1 * self.GRID_SIZE + self.TABLE_SECTION.x, y1 * self.GRID_SIZE + self.TABLE_SECTION.y,
                                                 (x2 - x1) * self.GRID_SIZE, (y2 - y1) * self.GRID_SIZE)
                        if table_rect.collidepoint(pos):
                            if self.selected_table == None:
                                if self.selected_button is not None and table.status == TableStatus.READY and table.get_combined_size([]) >= int(self.selected_button[0][2:]):
                                    if self.selected_button[0][0] == '1':
                                        #Rez
                                        if len(self.rez_party_pool_manager.find_pool_for_size(int(self.selected_button[0][2:]))) > 0:
                                            ret = function_dict[f"assign_res_{self.selected_button[0][2:]}_table:{table.number}"]
                                            self.selected_button = None
                                            return ret
                                    elif self.selected_button[0][0] == '0':
                                        #Walk-In
                                        if len(self.walkin_party_pool_manager.find_pool_for_size(int(self.selected_button[0][2:]))) > 0:
                                            ret = function_dict[f"assign_walk-in_{self.selected_button[0][2:]}_table:{table.number}"]
                                            self.selected_button = None
                                            return ret
                            if self.selected_button == None:
                                #We already selected a table so we are doing a combine action
                                if table == self.selected_table:
                                    self.selected_table = None
                                elif self.selected_table is not None:
                                    #Combine action
                                    if table.can_combine_with(self.selected_table):
                                        #There is only one action with a specific table index 0 and 1 so we try the first combo then if error try the other
                                        try:
                                            action = function_dict[f"combine_{table.number}_with_{self.selected_table.number}"]
                                        except:
                                            action = function_dict[f"combine_{self.selected_table.number}_with_{table.number}"]
                                        self.selected_table = None
                                        return action
                                    elif table.can_uncombine_with(self.selected_table):
                                        try:
                                            action = function_dict[
                                                f"uncombine_{table.number}_with_{self.selected_table.number}"]
                                        except:
                                            action = function_dict[
                                                f"uncombine_{self.selected_table.number}_with_{table.number}"]
                                        self.selected_table = None
                                        return action
                                else:
                                    self.selected_table = table
                            self.selected_button = None

                #Clicked on a party inside the party section
                elif self.PARTY_SECTION.collidepoint(pos):
                    for id, button in self.buttons:
                        if button.collidepoint(pos) and id[0] == 'w':
                            self.selected_wait = (id, button)
                            break
                    self.select_button(pos)
                    for id, button in self.buttons:
                        if button.collidepoint(pos) and (id[0] == 'q'):
                            if self.selected_button and self.selected_button[0][0] == '0':
                                # We are selected a walk in party
                                # Check to see if there is a party in that pool
                                if len(self.walkin_party_pool_manager.find_pool_for_size(
                                        int(self.selected_button[0][2:]))) > 0:
                                    # Pool is not empty so lets quote a wait to that party
                                    if self.selected_wait:
                                        wait = self.selected_wait[0][2:]
                                        action = function_dict[f"quote_wait_{wait}_party_size:{self.selected_button[0][2:]}"]
                                        return action
                    self.selected_table = None
                    return -1



            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    print(f"Took action Advance Time")
                    return function_dict["advance_time"]

        return -1
    def select_button(self,pos):
        #Find which button to select
        for id, button in self.buttons:
            if button.collidepoint(pos) and (id[0] == "1" or id[0] == "0"):
                if self.selected_button == (id,button):
                    self.selected_button = None
                else:
                    self.selected_button = (id,button)
                return

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
            if (int(self.universal_clock.current_time) < (int(party.arrival_time) + int(self.mutable_config['wait_tolerance']))):
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
        reward = 0
        remove_list = []
        for party in self.waitlist:
            if party.happiness <= 0:
                remove_list.append(party)
                if party.reservation:
                    self.rez_party_pool_manager.find_pool_for_size(party.num_people).remove(party)
                    if self.mutable_config['log_dir'] != "":
                        logging.info(
                            f"Party {party.name} of size {party.num_people} Left at {self.universal_clock.current_time}\n")
                    reward += -party.num_people * 4
                else:
                    self.walkin_party_pool_manager.find_pool_for_size(party.num_people).remove(party)
                    if self.mutable_config['log_dir'] != "":
                        logging.info(
                            f"Party {party.name} of size {party.num_people} Left at {self.universal_clock.current_time}\n")
                    reward += -party.num_people
        for party in remove_list:
            self.waitlist.remove(party)
        return reward

    def update_arrivals(self):

        remove_list = []
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
                    self.rez_party_pool_manager.find_pool_for_size(party.num_people).add(party)
                else:
                    self.walkin_party_pool_manager.find_pool_for_size(party.num_people).add(party)
                party.status = PartyStatus.ARRIVED
                remove_list.append(party)
        for party in remove_list:
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
        # Button settings
        section_width, section_height = self.PARTY_SECTION.width, self.PARTY_SECTION.height
        subsection_height = section_height / 2
        sub_label_height = subsection_height / 8
        quote_height = subsection_height / 6
        slider_height = subsection_height / 6
        deny_height = subsection_height / 8
        assign_height = subsection_height / 3

        res_banner = draw_button(self.screen, "Reservations", offset[0],offset[1]+subsection_height,
                    section_width, sub_label_height, self.colors['GRAY'], self.colors['WHITE'])
        if ("_",res_banner) not in self.buttons:
            self.buttons.append(("_",res_banner))
        walk_in_banner = draw_button(self.screen, "Walk-Ins", offset[0],offset[1],
                    section_width, sub_label_height, self.colors['GRAY'], self.colors['WHITE'])
        if ("_",walk_in_banner) not in self.buttons:
            self.buttons.append(("_",walk_in_banner))
        quote_wait_button = draw_button(self.screen, "Quote Wait", offset[0],offset[1]+sub_label_height+assign_height,
                    section_width, quote_height, self.colors['GRAY'], self.colors['WHITE'])
        if ("quote",quote_wait_button) not in self.buttons:
            self.buttons.append(("quote",quote_wait_button))
        #Draw the buttons for wait times that we will need
        wait_times = []
        for time in range(self.immutable_config['wait_quote_min'], self.immutable_config['wait_quote_max'],
                          self.immutable_config['wait_quote_step']):
            wait_times.append(str(time))
        num_rows = 2
        button_width = section_width / len(wait_times) * num_rows
        for index, time in enumerate(wait_times):
            if index < (len(wait_times)/num_rows):
                start_y = 0
                second_half = 0
            else:
                start_y = slider_height/2
                second_half = 1
            if self.selected_wait and self.selected_wait[0][2:] == time:
                color = self.colors["LIGHT_GREEN"]
            else:
                color = self.colors["GREEN"]
            time_button = draw_button(self.screen, time, offset[0]+(button_width*index)-(section_width*second_half),
                                      offset[1] + sub_label_height + assign_height + quote_height+start_y,
                                      button_width, slider_height/2, color, self.colors['WHITE'])
            if (f"w_{time}", time_button) not in self.buttons:
                self.buttons.append((f"w_{time}", time_button))


        deny_button = draw_button(self.screen, "Deny Party", offset[0],offset[1]+sub_label_height+assign_height+quote_height+slider_height,
                    section_width, deny_height, self.colors['ORANGE'], self.colors['WHITE'])
        if ("deny",deny_button) not in self.buttons:
            self.buttons.append(("deny",deny_button))
        # Buttons for party sizes (Waitlist Section)
        party_sizes = ['2', '4', '6', '8']
        for i, size in enumerate(party_sizes):
            for j in range(2):
                button_rect = pygame.Rect(offset[0] + ((i % 2) * section_width / 2), offset[1] +( (i // 2) * assign_height / 2 )+sub_label_height+(j*subsection_height) , section_width/2, assign_height/2)
                party_pool_button = draw_button(self.screen, size,button_rect.x,button_rect.y,button_rect.width,button_rect.height, self.colors['BLUE'],self.colors['WHITE'])
                if (f"{j}-{size}",party_pool_button) not in self.buttons:
                    self.buttons.append((f"{j}-{size}",party_pool_button))
                if self.selected_button is not None and self.selected_button[1] == party_pool_button:
                    _ = draw_button(self.screen, size, button_rect.x,button_rect.y,button_rect.width,button_rect.height, self.colors['LIGHT_BLUE'],self.colors['WHITE'])
                amt_in_pool = len(self.walkin_party_pool_manager.pools[i]) if not j else len(self.rez_party_pool_manager.pools[i])
                try:
                    happiness = self.walkin_party_pool_manager.pools[i].inspect_party().happiness if not j else self.rez_party_pool_manager.pools[i].inspect_party().happiness
                except:
                    happiness = "_"
                text_surface = self.font.render(f"{amt_in_pool}", True, self.colors['RED'])
                happiness_surface = self.font.render(f"{happiness}", True, self.colors['ORANGE'])
                text_rect = text_surface.get_rect(center=button_rect.center)
                self.screen.blit(text_surface, (button_rect.x+button_rect.width*2/3, button_rect.y+button_rect.height*2/3))
                self.screen.blit(happiness_surface, (button_rect.x+button_rect.width*1/4, button_rect.y+button_rect.height*1/4))

    def draw_score(self):
        score_surface = pygame.font.SysFont(None, 60).render(f"{self.score}", True, self.colors['BLUE'])
        self.screen.blit(score_surface, (self.PARTY_SECTION.width / 2, self.PARTY_SECTION.height * 6 / 8))

    def draw_universal_clock(self,clock):
        clock_text = clock.get_time_str()
        clock_surface = pygame.font.SysFont(None,60).render(clock_text, True, self.colors['GREEN'])
        self.screen.blit(clock_surface, (self.PARTY_SECTION.width/2,self.PARTY_SECTION.height*7/8))
    def draw_background_table(self,table,offset):
        x1, y1 = table.footprint[0]
        x2, y2 = table.footprint[1]
        table_rect = pygame.Rect(x1 * self.GRID_SIZE + offset[0], y1 * self.GRID_SIZE + offset[1], (x2 - x1) * self.GRID_SIZE, (y2 - y1) * self.GRID_SIZE)
        if table.status == TableStatus.COMBINED:
            for combined_table in table.combined_with:
                x1, y1 = combined_table.footprint[0]
                x2, y2 = combined_table.footprint[1]
                combined_table_rect = pygame.Rect(x1 * self.GRID_SIZE + offset[0], y1 * self.GRID_SIZE + offset[1],
                                                  (x2 - x1) * self.GRID_SIZE, (y2 - y1) * self.GRID_SIZE)
                pygame.draw.line(self.screen, self.colors['RED'], table_rect.center, combined_table_rect.center,width=20)

    def draw_table(self,table,offset):
        x1, y1 = table.footprint[0]
        x2, y2 = table.footprint[1]
        table_rect = pygame.Rect(x1 * self.GRID_SIZE + offset[0], y1 * self.GRID_SIZE + offset[1], (x2 - x1) * self.GRID_SIZE, (y2 - y1) * self.GRID_SIZE)
        if table.status == TableStatus.READY :
            table_color = self.colors['GREEN']
            if self.selected_table == table:
                table_color = self.colors['LIGHT_GREEN']
            pygame.draw.rect(self.screen, table_color, table_rect)
            text = f"{table.number}"
            text_surface = self.font.render(text, True, self.colors['BLACK'])
            text_rect = text_surface.get_rect(center=table_rect.center)
            self.screen.blit(text_surface, text_rect)
            return
        elif table.status == TableStatus.COMBINED:
            table_color = self.colors['ORANGE']
            if self.selected_table == table:
                table_color = self.colors['LIGHT_ORANGE']
            pygame.draw.rect(self.screen, table_color, table_rect)
            text = f"{table.number}"
            text_surface = self.font.render(text, True, self.colors['BLACK'])
            text_rect = text_surface.get_rect(center=table_rect.center)
            self.screen.blit(text_surface, text_rect)
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
