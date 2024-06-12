# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from ClassDefinitions import *
import pygame
import sys
from helperFunctions import *
from datetime import datetime, timedelta

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pygame.init()

    # Screen dimensions
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Restaurant Simulation")

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
    party1 = Party(4, None, [], PartyStatus.ARRIVED, "18:00", "18:20", "19:30", 10,90)
    party2 = Party(4, None, [], PartyStatus.ARRIVED, "18:00", "18:20", "19:30", 10,30)
    party3 = Party(4, None, [], PartyStatus.ARRIVED, "18:00", "18:20", "19:30", 10,2)
    reservation1 = Reservation("Smith", 4, "19:00", "555-1234", "Window seat", ReservationStatus.PENDING)
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
    def get_max_font_size(text, max_width, max_height, base_font_size):
        font_size = base_font_size
        font = pygame.font.Font(None, font_size)
        text_width, text_height = font.size(text)

        while (text_width > max_width or text_height > max_height) and font_size > 1:
            font_size -= 1
            font = pygame.font.Font(None, font_size)
            text_width, text_height = font.size(text)

        return font

    def draw_grid(offset):
        for row in range(ROWS):
            for col in range(COLS):
                rect = pygame.Rect(col * GRID_SIZE + offset[0], row * GRID_SIZE + offset[1], GRID_SIZE, GRID_SIZE)
                pygame.draw.rect(screen, GRAY, rect, 1)


    def draw_parties(offset, scroll_offset):
        start_y = offset[1] - scroll_offset
        for i, party in enumerate(waitlist):
            party_rect = pygame.Rect(offset[0], start_y + i * PARTY_RECT_SIZE_Y, PARTY_SECTION.width, PARTY_RECT_SIZE_Y)
            if PARTY_SECTION.colliderect(party_rect):  # Only draw if within the PARTY_SECTION
                if selected_party is party:
                    pygame.draw.rect(screen, GREEN, party_rect)
                else:
                    if party.status is PartyStatus.SEATED:
                        pygame.draw.rect(screen, ORANGE , party_rect)
                    else:
                        pygame.draw.rect(screen, WHITE, party_rect)
                pygame.draw.rect(screen, BLACK, party_rect, 1)
                party_text = str(party)
                max_font = get_max_font_size(party_text, PARTY_SECTION.width - PARTY_PADDING_X, PARTY_RECT_SIZE_Y, 36)
                text_surface = max_font.render(party_text, True, BLACK)
                text_rect = text_surface.get_rect(center=party_rect.center)
                screen.blit(text_surface, text_rect.topleft)

    def draw_party_info(table, offset):
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
        max_font = get_max_font_size(party_text, PARTY_SECTION.width - PARTY_PADDING_X, PARTY_RECT_SIZE_Y, 36)
        text_surface = max_font.render(party_text, True, BLACK)
        text_rect = text_surface.get_rect(center=party_rect.center)
        screen.blit(text_surface, text_rect.topleft)

    def select_party(pos):
        global selected_party
        index = (pos[1]) // PARTY_RECT_SIZE_Y
        if index < len(waitlist):
            if waitlist[index].status is PartyStatus.ARRIVED:
                selected_party = waitlist[index]
        else:
            selected_party = None

    def draw_table(table,offset):
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

        text = f"{(universal_clock.current_time - table.party.sat_time).seconds//60}"
        text_surface = font.render(text, True, BLACK)
        text_rect = text_surface.get_rect(center=party_rect.center)
        screen.blit(text_surface, text_rect)

    def draw_score():
        score_text = f"Score: {game_score}"
        score_surface = font.render(score_text, True, GREEN)
        score_rect = score_surface.get_rect(center=(SCREEN_WIDTH * 2 // 3, 30))
        screen.blit(score_surface, score_rect.topright)

    def draw_universal_clock(clock):
        clock_text = clock.get_time_str()
        clock_surface = font.render(clock_text, True, GREEN)
        clock_rect = clock_surface.get_rect(center=(SCREEN_WIDTH // 2, 30))
        screen.blit(clock_surface, clock_rect.topleft)

    def update_tables():
        global update_tables_flag
        update_tables_flag = False
        for table in tables:
            if table.party:
                time_seated = (universal_clock.current_time - table.party.sat_time).seconds//60
                if time_seated >= table.party.dine_time:

                    # Update our score
                    score_party(table.party)

                    # Add party to served list and set table to dirty
                    table.party.status = PartyStatus.LEFT
                    served.append(table.party)
                    table.remove_party()
                    table.status = TableStatus.DIRTY

            elif table.status == TableStatus.DIRTY:
                if table.clean_progress >= table.clean_time:
                    table.clean_progress = 0
                    table.status = TableStatus.READY
                else:
                    table.clean_progress += 1

    def score_party(party):
        global game_score
        """
        :param party: the party to score

        This is where we will implement how our scoring system works (Very important for eventual AI)
        """
        game_score += party.num_people

        pass

    # Main game loop
    running = True
    tables = [table1, table2, table3, table4]
    game_score = 0
    update_tables_flag = False
    selected_party = None
    selected_table = None
    options_box_loc = None
    show_option_box = False
    option_rects = []
    options = []
    waitlist = [
        party1,
        party2,
        party3
        # Add more parties as needed for testing
    ]

    # Parties are added to this list once they have finished eating and left
    served = [

    ]

    # Initialize universal clock
    universal_clock = UniversalClock(datetime.now().replace(hour=18, minute=0, second=0, microsecond=0),speed_factor=15)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()

                #Clicked on a table inside the table section
                if TABLE_SECTION.collidepoint(pos):
                    selected_table = None
                    for table in tables:
                        x1, y1 = table.footprint[0]
                        x2, y2 = table.footprint[1]
                        table_rect = pygame.Rect(x1 * GRID_SIZE + TABLE_SECTION.x, y1 * GRID_SIZE + TABLE_SECTION.y,
                                                 (x2 - x1) * GRID_SIZE, (y2 - y1) * GRID_SIZE)
                        if table_rect.collidepoint(pos):
                            if selected_party and table.status == TableStatus.READY :
                                table.assign_party(selected_party)
                                waitlist.remove(selected_party)
                                selected_party.sat_time = universal_clock.current_time
                                selected_party = None
                            elif not selected_party and table.party :
                                selected_table = table
                            else:
                                selected_table = None

                #Clicked on a party inside the party section
                elif PARTY_SECTION.collidepoint(pos):
                    select_party(pos)
                    pass
        # Update universal clock - returns true only when it is updated every second
        if universal_clock.update():
            if not update_tables_flag: update_tables_flag = True

        update_tables()
        # Game logic here

        screen.fill(BLACK)

        # Draw party section
        pygame.draw.rect(screen, BLUE, PARTY_SECTION, 2)
        screen.set_clip(PARTY_SECTION)
        screen.set_clip(None)
        draw_parties((PARTY_SECTION.x, PARTY_SECTION.y), scroll_offset)

        # Draw table section
        pygame.draw.rect(screen, BLUE, TABLE_SECTION, 2)
        table_offset = (TABLE_SECTION.x,TABLE_SECTION.y)
        screen.set_clip(TABLE_SECTION)
        draw_grid(table_offset)

        for table in tables:
            draw_table(table,table_offset)
        screen.set_clip(None)

        # Draw option box if needed
        if selected_table:
            draw_party_info(selected_table,table_offset)

        # Draw universal clock
        draw_universal_clock(universal_clock)

        # Draw score
        draw_score()
        # Drawing code
        pygame.display.flip()

    pygame.quit()
    sys.exit()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
