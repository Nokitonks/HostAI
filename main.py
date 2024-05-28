# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from ClassDefinitions import *
import pygame
import sys
from helperFunctions import *


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

    # Font
    font = pygame.font.Font(None, 36)
    # Create example instances
    party1 = Party(4, None, [], PartyStatus.MAIN_COURSE, "18:00", "18:20", "19:30", 10)
    reservation1 = Reservation("Smith", 4, "19:00", "555-1234", "Window seat", ReservationStatus.PENDING)
    check1 = Check("18:30", "20:00", 75.50)
    table1 = Table(((4, 4), (6, 6)), 4, 8, 'regular table', [], party1,TableStatus.OCCUPIED)
    table2 = Table(((8, 8), (10, 10)), 4, 8, 'regular table', [], None,TableStatus.READY)
    table3 = Table(((8, 4), (10, 6)), 4, 8, 'regular table', [], None,TableStatus.READY)
    table4 = Table(((4, 8), (6, 10)), 4, 8, 'regular table', [], None,TableStatus.READY)

    # Define sections
    TABLE_SECTION = pygame.Rect(SCREEN_WIDTH // 4, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
    PARTY_SECTION = pygame.Rect(0, 0, SCREEN_WIDTH // 4, SCREEN_HEIGHT)

    def draw_grid(offset):
        for row in range(ROWS):
            for col in range(COLS):
                rect = pygame.Rect(col * GRID_SIZE + offset[0], row * GRID_SIZE + offset[1], GRID_SIZE, GRID_SIZE)
                pygame.draw.rect(screen, GRAY, rect, 1)



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
            return
        else:
            #If we are here it means that table.party is not null
            if table.party is None: raise ValueError("ERROR: Table is occupied but no Party Object ")
            table_color = GRAY

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



    # Main game loop
    running = True
    tables = [table1, table2, table3, table4]
    selected_party = None
    selected_table = None
    options_box_loc = None
    show_option_box = False
    option_rects = []
    options = []

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()


                if TABLE_SECTION.collidepoint(pos):
                    for table in tables:
                        x1, y1 = table.footprint[0]
                        x2, y2 = table.footprint[1]
                        table_rect = pygame.Rect(x1 * GRID_SIZE + TABLE_SECTION.x, y1 * GRID_SIZE + TABLE_SECTION.y,
                                                 (x2 - x1) * GRID_SIZE, (y2 - y1) * GRID_SIZE)
                        if table_rect.collidepoint(pos):
                            break

                elif PARTY_SECTION.collidepoint(pos):
                   # select_party(pos)
                    pass


        # Game logic here

        screen.fill(BLACK)

        # Draw party section
        pygame.draw.rect(screen, BLUE, PARTY_SECTION, 2)
        screen.set_clip(PARTY_SECTION)
        screen.set_clip(None)

        # Draw table section
        pygame.draw.rect(screen, BLUE, TABLE_SECTION, 2)
        table_offset = (TABLE_SECTION.x,TABLE_SECTION.y)
        screen.set_clip(TABLE_SECTION)
        draw_grid(table_offset)

        for table in tables:
            draw_table(table,table_offset)
        screen.set_clip(None)

        # Draw option box if needed
        if show_option_box:
            option_rects, options = draw_option_box(selected_table, options_box_loc)

        # Drawing code
        pygame.display.flip()

    pygame.quit()
    sys.exit()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
