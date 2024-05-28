import pygame


def create_scaled_rect(original_rect, percentage):
    """
    Creates a new rectangle that is centered inside the original rectangle
    with a size that is a given percentage of the original rectangle's size.

    :param original_rect: The original pygame.Rect.
    :param percentage: The percentage of the original rectangle's size (e.g., 50 for 50%).
    :return: A new pygame.Rect that is scaled and centered within the original rectangle.
    """
    # Ensure the percentage is in the range [0, 100]
    if percentage < 0 or percentage > 100:
        raise ValueError("Percentage must be between 0 and 100")

    # Calculate the new width and height based on the percentage
    new_width = int(original_rect.width * (percentage / 100))
    new_height = int(original_rect.height * (percentage / 100))

    # Calculate the new position to center the new rectangle
    new_x = original_rect.x + (original_rect.width - new_width) // 2
    new_y = original_rect.y + (original_rect.height - new_height) // 2

    # Create the new rectangle
    new_rect = pygame.Rect(new_x, new_y, new_width, new_height)

    return new_rect