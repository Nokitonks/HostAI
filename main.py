# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from ClassDefinitions import *
import pygame
import sys
from helperFunctions import *
from datetime import datetime, timedelta
import csv

class Screen:
    def __init__(self):
        self.next_screen = None

    def handle_events(self, events):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def draw(self, screen):
        raise NotImplementedError

class GameManager:
    def __init__(self):
        self.screens = {
            'main_screen': MainScreen(),
            'reservation_view': ReservationView(),
            'game_over': GameOverScreen()
        }
        self.current_screen = self.screens['main_screen']

    def handle_events(self, events):
        self.current_screen.handle_events(events)

    def update(self):
        next_screen = self.current_screen.next_screen
        if next_screen:
            self.current_screen = self.screens[next_screen]
            self.current_screen.next_screen = None
        self.current_screen.update()

    def draw(self, screen):
        self.current_screen.draw(screen)
# Main screen
class MainScreen(Screen):
    def __init__(self):
        super().__init__()
        self.button_rect = pygame.Rect((SCREEN_WIDTH - BUTTON_WIDTH) // 2, SCREEN_HEIGHT - BUTTON_HEIGHT - 10, BUTTON_WIDTH, BUTTON_HEIGHT)

    def handle_events(self, events):
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.button_rect.collidepoint(event.pos):
                    self.next_screen = 'reservation_view'

    def update(self):
        pass

    def draw(self, screen):
        screen.fill(BLUE)
        text = font.render("Main Screen", True, WHITE)
        screen.blit(text, (100, 100))
        pygame.draw.rect(screen, GRAY, self.button_rect)
        button_text = font.render("View Reservations", True, BLACK)
        screen.blit(button_text, (self.button_rect.x + 10, self.button_rect.y + 10))

# Reservation view
class ReservationView(Screen):
    def __init__(self):
        super().__init__()

        self.tables = [table1, table2, table3, table4]
        self.game_score = 0
        self.update_tables_flag = False
        self.selected_party = None
        self.selected_table = None
        self.options_box_loc = None
        self.show_option_box = False
        self.option_rects = []
        self.options = []
        self.show_reservations = False
        self.waitlist = [
        ]

        # Parties are added to this list once they have finished eating and left
        served = [
        ]

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


        # Initialize universal clock

        self.universal_clock = UniversalClock(datetime.now().replace(hour=18, minute=0, second=0, microsecond=0),speed_factor=15)

    def handle_events(self, events):
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()

                #Clicked on a table inside the table section
                if TABLE_SECTION.collidepoint(pos):
                    self.selected_table = None
                    for table in self.tables:
                        x1, y1 = table.footprint[0]
                        x2, y2 = table.footprint[1]
                        table_rect = pygame.Rect(x1 * GRID_SIZE + TABLE_SECTION.x, y1 * GRID_SIZE + TABLE_SECTION.y,
                                                 (x2 - x1) * GRID_SIZE, (y2 - y1) * GRID_SIZE)
                        if table_rect.collidepoint(pos):
                            if self.selected_party and table.status == TableStatus.READY and table.size_px >= self.selected_party.num_people:
                                table.assign_party(self.selected_party)
                                self.waitlist.remove(self.selected_party)
                                self.selected_party.sat_time = self.universal_clock.current_time
                                self.selected_party = None
                            elif not self.selected_party and table.party :
                                self.selected_table = table
                            else:
                                self.selected_table = None

                #Clicked on a party inside the party section
                elif PARTY_SECTION.collidepoint(pos):
                    self.select_party(pos)


            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.show_reservations = not self.show_reservations

    def update(self):

        # Update universal clock - returns true only when it is updated every second
        if self.universal_clock.update():
            if not self.update_tables_flag: self.update_tables_flag = True

        self.update_tables()
        self.update_arrivals()
        # Game logic here

        pass

    def draw(self, screen):
        screen.fill(BLACK)

        # Draw party section
        pygame.draw.rect(screen, BLUE, PARTY_SECTION, 2)
        screen.set_clip(PARTY_SECTION)
        screen.set_clip(None)
        self.draw_parties((PARTY_SECTION.x, PARTY_SECTION.y), scroll_offset)

        # Draw table section
        pygame.draw.rect(screen, BLUE, TABLE_SECTION, 2)
        table_offset = (TABLE_SECTION.x, TABLE_SECTION.y)
        screen.set_clip(TABLE_SECTION)
        self.draw_grid(table_offset)

        for table in self.tables:
            self.draw_table(table, table_offset)
        screen.set_clip(None)

        # Draw option box if needed
        if self.selected_table:
            self.draw_party_info(self.selected_table, table_offset)

        # Draw universal clock
        self.draw_universal_clock(self.universal_clock)

        # Draw score
        self.draw_score()

        # Draw reservation view
        if self.show_reservations:
            self.draw_reservation_view((0, SCREEN_WIDTH * 0.2))

    def get_max_font_size(self,text, max_width, max_height, base_font_size):
        font_size = base_font_size
        font = pygame.font.Font(None, font_size)
        text_width, text_height = font.size(text)

        while (text_width > max_width or text_height > max_height) and font_size > 1:
            font_size -= 1
            font = pygame.font.Font(None, font_size)
            text_width, text_height = font.size(text)

        return font

    def draw_grid(self,offset):
        for row in range(ROWS):
            for col in range(COLS):
                rect = pygame.Rect(col * GRID_SIZE + offset[0], row * GRID_SIZE + offset[1], GRID_SIZE, GRID_SIZE)
                pygame.draw.rect(screen, GRAY, rect, 1)

    def draw_parties(self,offset, scroll_offset):
        start_y = offset[1] - scroll_offset
        for i, party in enumerate(self.waitlist):
            party_rect = pygame.Rect(offset[0], start_y + i * PARTY_RECT_SIZE_Y, PARTY_SECTION.width, PARTY_RECT_SIZE_Y)
            if PARTY_SECTION.colliderect(party_rect):  # Only draw if within the PARTY_SECTION
                if self.selected_party is party:
                    pygame.draw.rect(screen, GREEN, party_rect)
                else:
                    if party.status is PartyStatus.SEATED:
                        pygame.draw.rect(screen, ORANGE , party_rect)
                    else:
                        pygame.draw.rect(screen, WHITE, party_rect)
                pygame.draw.rect(screen, BLACK, party_rect, 1)
                party_text = str(party)
                max_font = self.get_max_font_size(party_text, PARTY_SECTION.width - PARTY_PADDING_X, PARTY_RECT_SIZE_Y, 36)
                text_surface = max_font.render(party_text, True, BLACK)
                text_rect = text_surface.get_rect(center=party_rect.center)
                screen.blit(text_surface, text_rect.topleft)

    def draw_party_info(self,table, offset):
        x1, y1 = table.footprint[0]
        x2, y2 = table.footprint[1]
        table_rect = pygame.Rect(x1 * GRID_SIZE + offset[0], y1 * GRID_SIZE + offset[1], (x2 - x1) * GRID_SIZE, (y2 - y1) * GRID_SIZE)

        # Calculate party rectangle position (above the table)
        party_rect_x = table_rect.x
        party_rect_y = table_rect.y - PARTY_RECT_SIZE_Y
        party_rect_width = PARTY_SECTION.width - PARTY_PADDING_X
        party_rect_height = PARTY_RECT_SIZE_Y

        # Draw the party rectangle
        party_rect = pygame.Rect(party_rect_x, party_rect_y, party_rect_width, party_rect_height)

        if table.party.status is PartyStatus.SEATED:
            pygame.draw.rect(screen, ORANGE, party_rect)
        else:
            pygame.draw.rect(screen, WHITE, party_rect)
        pygame.draw.rect(screen, BLACK, party_rect, 1)
        party_text = str(table.party)
        max_font = self.get_max_font_size(party_text, PARTY_SECTION.width - PARTY_PADDING_X, PARTY_RECT_SIZE_Y, 36)
        text_surface = max_font.render(party_text, True, BLACK)
        text_rect = text_surface.get_rect(center=party_rect.center)
        screen.blit(text_surface, text_rect.topleft)

    def select_party(self,pos):
        index = (pos[1]) // PARTY_RECT_SIZE_Y
        if index < len(self.waitlist):
            if self.waitlist[index].status is PartyStatus.ARRIVED:
                self.selected_party = self.waitlist[index]
        else:
            self.selected_party = None

    def draw_table(self,table,offset):
        x1, y1 = table.footprint[0]
        x2, y2 = table.footprint[1]
        table_rect = pygame.Rect(x1 * GRID_SIZE + offset[0], y1 * GRID_SIZE + offset[1], (x2 - x1) * GRID_SIZE, (y2 - y1) * GRID_SIZE)
        if table.status == TableStatus.READY :
            table_color = GREEN
            pygame.draw.rect(screen, table_color, table_rect)
            return
        elif table.status == TableStatus.DIRTY:
            table_color = RED
            pygame.draw.rect(screen, table_color, table_rect)

            # Draw the foreground rectangle based on progress
            foreground_width = (table_rect.width * table.clean_progress) / table.clean_time
            pygame.draw.rect(screen, GREEN, (table_rect.x, table_rect.y, foreground_width, table_rect.height))

            return
        else:
            #If we are here it means that table.party is not null
            if table.party is None: raise ValueError("ERROR: Table is occupied but no Party Object ")
            table_color = GRAY


        #  table.party exists from here on out:-->

        pygame.draw.rect(screen, table_color, table_rect)
        #Find status of the party at the table

        if table.party.status == PartyStatus.SEATED:
            party_color = GREEN
        elif table.party.status == PartyStatus.APPS:
            party_color = ORANGE
        elif table.party.status == PartyStatus.MAIN_COURSE:
            party_color = RED
        elif table.party.status == PartyStatus.DESSERT:
            party_color = PURPLE
        elif table.party.status == PartyStatus.CHECK_DROPPED:
            party_color = BLUE

        party_rect = create_scaled_rect(table_rect,60)
        pygame.draw.rect(screen, party_color, party_rect)
        pygame.draw.rect(screen, party_color, party_rect)

        text = f"{(self.universal_clock.current_time - table.party.sat_time).seconds//60}"
        text_surface = font.render(text, True, BLACK)
        text_rect = text_surface.get_rect(center=party_rect.center)
        screen.blit(text_surface, text_rect)

    def draw_score(self):
        score_text = f"Score: {self.game_score}"
        score_surface = font.render(score_text, True, GREEN)
        score_rect = score_surface.get_rect(center=(SCREEN_WIDTH * 2 // 3, 30))
        screen.blit(score_surface, score_rect.topright)

    def draw_universal_clock(self,clock):
        clock_text = clock.get_time_str()
        clock_surface = font.render(clock_text, True, GREEN)
        clock_rect = clock_surface.get_rect(center=(SCREEN_WIDTH // 2, 30))
        screen.blit(clock_surface, clock_rect.topleft)

    def update_tables(self):
        global update_tables_flag
        update_tables_flag = False
        for table in self.tables:
            if table.party:
                time_seated = (self.universal_clock.current_time - table.party.sat_time).seconds//60
                if time_seated >= table.party.dine_time:

                    # Update our score
                    self.score_party(table.party)

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

    def update_arrivals(self):

        for party in self.walk_ins:
            clock_min = datetime.strptime(self.universal_clock.get_time_str(),"%H:%M:%S")
            arrival_min = datetime.strptime(party.arrival_time,"%H:%M")
            if arrival_min <= clock_min:
                self.waitlist.append(party)
                party.status = PartyStatus.ARRIVED
                self.walk_ins.remove(party)

    def score_party(self,party):
        """
        :param party: the party to score

        This is where we will implement how our scoring system works (Very important for eventual AI)
        """
        self.game_score += party.num_people

    def draw_reservation_view(self,offset):
        # Graph dimensions and position
        graph_x = 0
        graph_y = SCREEN_HEIGHT*0.2
        graph_width = SCREEN_WIDTH
        graph_height = SCREEN_HEIGHT * 0.8

        pygame.draw.rect(screen, ORANGE, (graph_x, SCREEN_HEIGHT*0.2, graph_width, graph_height), 2)
        pygame.draw.rect(screen, GRAY, (graph_x, SCREEN_HEIGHT*0.2, graph_width, graph_height) )


        # Draw reservations
        RESO_RECT_SIZE = 40
        start_y = offset[1]
        for i, reservation in enumerate(self.reservations):
            reso_rect = pygame.Rect(offset[0], start_y + i * RESO_RECT_SIZE, SCREEN_WIDTH, RESO_RECT_SIZE)
            pygame.draw.rect(screen, WHITE, reso_rect)
            pygame.draw.rect(screen, BLACK, reso_rect, 1)
            reso_text = str(reservation)
            max_font = self.get_max_font_size(reso_text, SCREEN_WIDTH, RESO_RECT_SIZE, 36)
            text_surface = max_font.render(reso_text, True, BLACK)
            text_rect = text_surface.get_rect(center=reso_rect.center)
            screen.blit(text_surface, text_rect.topleft)

    # Function to read reservations from CSV
    def read_reservations(self,csv_file):
        reservations = []
        with open(csv_file, mode='r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                reservation = Reservation(row['name'], int(row['num_people']), row['reservation_time'], row['phone'],
                                          row['notes'], ReservationStatus[row['status']])
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

# Game over screen
class GameOverScreen(Screen):
    def __init__(self):
        super().__init__()
        self.button_rect = pygame.Rect((SCREEN_WIDTH - BUTTON_WIDTH) // 2, SCREEN_HEIGHT // 2, BUTTON_WIDTH, BUTTON_HEIGHT)

    def handle_events(self, events):
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.button_rect.collidepoint(event.pos):
                    self.next_screen = 'main_screen'

    def update(self):
        pass

    def draw(self, screen):
        screen.fill(BLACK)
        text = font.render("Game Over", True, WHITE)
        screen.blit(text, (SCREEN_WIDTH // 2 - text.get_width() // 2, SCREEN_HEIGHT // 2 - text.get_height() // 2 - 50))
        pygame.draw.rect(screen, GRAY, self.button_rect)
        button_text = font.render("Restart", True, BLACK)
        screen.blit(button_text, (self.button_rect.x + 30, self.button_rect.y + 10))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pygame.init()

    # Screen dimensions
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Restaurant Simulation")
    # Button dimensions
    BUTTON_WIDTH = 200
    BUTTON_HEIGHT = 50
    # Colors
    WHITE = (255, 255, 255)
    RED= (255,0, 0)
    ORANGE= (255,165, 0)
    PURPLE = (255, 0, 255)
    BLUE = (0, 0, 255)
    BLACK = (0, 0, 0)
    GREEN = (0, 255, 0)
    GRAY = (200, 200, 200)
    LIGHT_GRAY = (220, 220, 220)

    # Grid settings
    GRID_SIZE = 40  # Size of the grid cells
    ROWS = SCREEN_HEIGHT // GRID_SIZE
    COLS = SCREEN_WIDTH // GRID_SIZE

    # Scroll settings
    scroll_offset = 0
    scroll_speed = 20

    # Font
    font = pygame.font.Font(None, 36)
    # Create example instances
    check1 = Check("18:30", "20:00", 75.50)
    table1 = Table(((4, 4), (6, 6)), 4, 8, 'regular table', [], None,TableStatus.READY)
    table2 = Table(((8, 8), (10, 10)), 4, 8, 'regular table', [], None,TableStatus.READY)
    table3 = Table(((8, 4), (10, 6)), 4, 8, 'regular table', [], None,TableStatus.READY)
    table4 = Table(((4, 8), (6, 10)), 4, 8, 'regular table', [], None,TableStatus.READY)


    # Define sections
    TABLE_SECTION = pygame.Rect(SCREEN_WIDTH // 4, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
    PARTY_SECTION = pygame.Rect(0, 0, SCREEN_WIDTH // 4, SCREEN_HEIGHT)

    PARTY_PADDING_X = 10
    PARTY_RECT_SIZE_Y = 40


    # Main game loop
    running = True
    game_manager = GameManager()

    running = True
    while running:
        events = pygame.event.get()
        game_manager.handle_events(events)
        game_manager.update()

        game_manager.draw(screen)
        # Drawing code
        pygame.display.flip()

    pygame.quit()
    sys.exit()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
