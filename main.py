# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from ClassDefinitions import *
import pygame



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
    BLACK = (0, 0, 0)
    GREEN = (0, 255, 0)
    GRAY = (200, 200, 200)

    # Grid settings
    GRID_SIZE = 40  # Size of the grid cells
    ROWS = SCREEN_HEIGHT // GRID_SIZE
    COLS = SCREEN_WIDTH // GRID_SIZE

    # Font
    font = pygame.font.Font(None, 36)
    # Create example instances
    party1 = Party(4, None, [], PartyStatus.ARRIVED, "18:00", "18:20", "19:30", 10)
    reservation1 = Reservation("Smith", 4, "19:00", "555-1234", "Window seat", ReservationStatus.PENDING)
    check1 = Check("18:30", "20:00", 75.50)
    table1 = Table(((4, 4), (6, 6)), 4, 8, 'regular table', [], party1)
    table2 = Table(((8, 8), (10, 10)), 4, 8, 'regular table', [], None)
    table3 = Table(((8, 4), (10, 6)), 4, 8, 'regular table', [], None)
    table4 = Table(((4, 8), (6, 10)), 4, 8, 'regular table', [], None)



    def draw_grid():
        for row in range(ROWS):
            for col in range(COLS):
                rect = pygame.Rect(col * GRID_SIZE, row * GRID_SIZE, GRID_SIZE, GRID_SIZE)
                pygame.draw.rect(screen, GRAY, rect, 1)


    def draw_table(table):
        x1, y1 = table.footprint[0]
        x2, y2 = table.footprint[1]
        table_rect = pygame.Rect(x1 * GRID_SIZE, y1 * GRID_SIZE, (x2 - x1) * GRID_SIZE, (y2 - y1) * GRID_SIZE)
        pygame.draw.rect(screen, GREEN, table_rect)
    # Main game loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Game logic here

        # Drawing code
        screen.fill(WHITE)
        draw_grid()
        draw_table(table1)
        draw_table(table2)
        draw_table(table3)
        draw_table(table4)
        pygame.display.flip()

    pygame.quit()
    sys.exit()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
