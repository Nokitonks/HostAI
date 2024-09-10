import pygame
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Dict, Tuple, Box, Discrete


def function_into_action_number(tables,unique_combos,immutable_config) -> dict:
    cnt = 0
    name = {}
    size_dict = {
        0:2,
        1:4,
        2:6,
        3:8
    }
    for pools in range(4):
        for table in tables:
            name[f"assign_res_{size_dict[pools]}_table:{table.number}"] = cnt
            cnt += 1
    for pools in range(4):
        for table in tables:
            name[f"assign_walk-in_{size_dict[pools]}_table:{table.number}"] = cnt
            cnt += 1
    for pools in range(4):
        for time in range(immutable_config['wait_quote_min'], immutable_config['wait_quote_max'],
                          immutable_config['wait_quote_step']):
            name[f"quote_wait_{time}_party_size:{size_dict[pools]}"] = cnt
            cnt += 1
    for pools in range(4):
        name[f"deny_party_size:{size_dict[pools]}"] = cnt
        cnt += 1
    for combo in unique_combos:
        name[f"combine_{combo[0].number}_with_{combo[1].number}"] = cnt
        cnt += 1
        name[f"uncombine_{combo[0].number}_with_{combo[1].number}"] = cnt
        cnt += 1
    name[f"advance_time"] = cnt
    return name
def action_number_into_function(tables,unique_combos,immutable_config) -> dict:
    cnt = 0
    name = {}
    size_dict = {
        0:2,
        1:4,
        2:6,
        3:8
    }
    for pools in range(4):
        for table in tables:
            name[cnt] = f"Assign Reservation Party of size {size_dict[pools]} to Table:{table.number}"
            cnt += 1
    for pools in range(4):
        for table in tables:
            name[cnt] = f"Assign Walk-In Party of size {size_dict[pools]} to Table:{table.number}"
            cnt += 1
    for pools in range(4):
        for time in range(immutable_config['wait_quote_min'], immutable_config['wait_quote_max'],
                          immutable_config['wait_quote_step']):
            name[cnt] = f"Quoted wait of {time}minutes to party of size {size_dict[pools]} "
            cnt += 1
    for pools in range(4):
        name[cnt] = f"Denied party of size {size_dict[pools]}"
        cnt += 1
    for combo in unique_combos:
        name[cnt] = f"Combine Table{combo[0].number} with Table:{combo[1].number}"
        cnt += 1
        name[cnt] = f"Uncombine Table{combo[0].number} with Table:{combo[1].number}"
        cnt += 1
    name[cnt] = "Default Action (Advance Time)"
    print(name)
    return name


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

def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

def get_max_font_size(text, max_width, max_height, base_font_size):
    font_size = base_font_size
    font = pygame.font.Font(None, font_size)
    text_width, text_height = font.size(text)

    while (text_width > max_width or text_height > max_height) and font_size > 1:
        font_size -= 1
        font = pygame.font.Font(None, font_size)
        text_width, text_height = font.size(text)

    return font
# Function to draw a button
def draw_button(screen, text, x, y, w, h, color,text_color):
    small_font = pygame.font.SysFont(None, 32)
    pygame.draw.rect(screen, color, (x, y, w, h))
    label = small_font.render(text, True, text_color)
    screen.blit(label, (x + (w // 2 - label.get_width() // 2), y + (h // 2 - label.get_height() // 2)))
    return pygame.Rect(x, y, w, h)

def flat_obs_into_variable(obs):

    return 42

def flatten_space(space, parent_key='', sep='_'):
    """
        Recursively flatten a Gymnasium observation space (Dict, Tuple) and handle Discrete spaces
        by accounting for their one-hot encoding.

        Parameters:
        space (spaces.Space): The Gymnasium observation space.
        parent_key (str): The base key to prepend to keys in the observation.
        sep (str): The separator between parent and child keys.

        Returns:
        list: A list of (key, size) tuples, where key is the flattened key and size is the number of elements.
        """
    items = []
    if isinstance(space, Dict):
        for k, subspace in space.spaces.items():
            new_key = parent_key + sep + k if parent_key else k
            items.extend(flatten_space(subspace, new_key, sep=sep))
    elif isinstance(space, Tuple):
        for idx, subspace in enumerate(space.spaces):
            new_key = f"{parent_key}{sep}{idx}"
            items.extend(flatten_space(subspace, new_key, sep=sep))
    elif isinstance(space, Discrete):
        # Discrete space is represented as one-hot, so the size is the number of discrete choices
        items.append((parent_key, int(space.n)))  # Use n as the size for one-hot encoding
    else:
        # Base case: for Box or other spaces
        size = int(np.prod(space.shape)) if hasattr(space, 'shape') else 1
        items.append((parent_key, size))
    return items

def create_flattened_mapping(space):
    """
        Create a mapping from the original space to the flattened index, including handling one-hot
        encoding for Discrete spaces.

        Parameters:
        space (spaces.Space): The original Gymnasium observation space.

        Returns:
        dict: A mapping from feature names to index ranges in the flattened array.
        """
    flattened_space = flatten_space(space)
    mapping = {}
    current_idx = 0
    for key, size in flattened_space:
        mapping[key] = (current_idx, current_idx + size)
        current_idx += size
    return mapping

def one_hot_to_int(one_hot_vector):
    """
    Convert a one-hot encoded vector (either list or NumPy array) to its corresponding integer value.

    Parameters:
    one_hot_vector (list or np.array): A one-hot encoded vector (e.g., [0, 1, 0]).

    Returns:
    int: The index of the 1 in the one-hot vector, or -1 if the vector is invalid.
    """
    if np.sum(one_hot_vector) != 1:
        # If the sum of elements is not 1, it's not a valid one-hot vector
        return -1
    return np.argmax(one_hot_vector)  # Returns the index of the max value, which should be 1


def select_features_from_flattened(flattened_obs, flattened_mapping, desired_features):
    """
        Select specific features from the flattened observation based on the mapping, including
        one-hot encoded Discrete spaces.

        Parameters:
        flattened_obs (list or array): The flattened observation.
        flattened_mapping (dict): The mapping from feature names to index ranges.
        desired_features (list): List of feature names to extract.

        Returns:
        list: A list of values corresponding to the selected features.
        """
    selected_values = []
    for feature in desired_features:
        start_idx, end_idx = flattened_mapping[feature]
        selected_values.extend([one_hot_to_int(flattened_obs[start_idx:end_idx])])
    return selected_values