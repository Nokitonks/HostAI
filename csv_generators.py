"""
File for creating reservation lists and walk_in lists for
both learning and testing
"""

import csv
import pandas as pd
from faker import Faker # For random names and such
from scipy.stats import norm, beta, gamma
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from ClassDefinitions import Algorithm
def create_specific_reservations_list(algorithm,config):
    if algorithm == Algorithm.CL_PPO_RUDDER_PHASE_0:

        result = []
        faker = Faker(seed=1)
        # loop through all the reservations and assign them to a reservation time
        for num in (config['reservations']):
            rez = {}
            rez['num_people'] = num
            rez['reservation_time'] = 0
            rez['name'] = faker.name()
            rez['dine_time'] = 10
            rez['meal_split'] = "10:20:20:40:10"  # TODO put in normal meal_split
            rez['status'] = "CONFIRMED"
            result.append(rez)

        with open(f'reservation_files/cl_ppo_rudder/phase_0.csv', 'w', newline='') as csvfile:
            fieldnames = result[0].keys()  # Extract headers from the first dictionary
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(result)


def create_reservation_list(seed,total_time,total_covers,push_num,push_times,
                            mean=5,std=3,bins=(-np.inf,0,7.5,18,np.inf),
                            normality=0.8,size_density="normal",meal_split='normal',dine_times='normal'):
    """
    :param seed: To replicate randomness
    :param total_covers: Total amount of people in reservations total
    :param push_num: How many "pushes" for the reservations (pushes are essentially rushes of people)
    :param push_times: The corresponding times
    :param normality: 0-1 how normal we would like the distribution around each push
    :param size_density: the percentage of 2-8 tops. specified as small, medium, large, normal
    :param meal_split: What percentage of time reservations will spend on each meal section, only normal right now
    :param dine_times: how long each party will dine, specified as quick, normal, long
    :return: the reservation list as a CSV file for the evening
    """
    pass

    # Define the four specific values
    values = [2, 4, 6, 8]

    # Specify the total number of people desired
    total_people = total_covers  # Replace this with your target total number of people

    # Initialize variables to store the mapped values and the cumulative people count
    mapped_values = []
    cumulative_people = 0

    # Continue generating reservations until we reach or exceed the total number of people
    while cumulative_people < total_people:
        # Generate a normal distributed random value
        normal_value = np.random.normal(loc=mean, scale=std)

        # Discretize the normal value into one of the specific values
        discrete_value_index = np.digitize(normal_value, bins) - 1
        reservation_value = values[discrete_value_index]

        # Add the reservation to the list
        mapped_values.append(reservation_value)
        cumulative_people += reservation_value

    # Number of peaks (number of normal distributions to combine)
    num_peaks = push_num

    # Number of entries to generate
    num_entries = len(mapped_values)

    # Define the peaks (centers of the normal distributions)
    # These can be chosen randomly or set manually
    peak_positions = push_times

    # Standard deviation for each peak
    std_dev = total_time/10

    # Generate normal distributions around each peak
    times = []
    for peak in peak_positions:
        times.extend(np.random.normal(loc=peak, scale=std_dev, size=num_entries // num_peaks))
    times.extend(np.random.normal(loc=peak_positions[0], scale=std_dev, size=num_entries % num_peaks))
    # If necessary, ensure that all values are within the [0, 360] range
    times = np.clip(times, 0, total_time)

    # If you want to wrap around values that exceed 360 or are less than 0
    times = np.mod(times, total_time)

    reservation_interval = 15
    discrete_times = np.digitize(times, list(range(0,total_time+1,reservation_interval)))

    result = []
    faker = Faker(seed=seed)
    # loop through all the reservations and assign them to a reservation time
    for index, px in enumerate(mapped_values):
        rez = {}
        rez['num_people'] = px
        rez['reservation_time'] = discrete_times[index] * reservation_interval
        rez['name'] = faker.name()
        rez['dine_time'] = 20 # TODO: Put in dine time normal
        rez['meal_split'] = "10:20:20:40:10" # TODO put in normal meal_split
        rez['status'] = "CONFIRMED"
        result.append(rez)

    print(result)
    #plot_reservation(result)
    with open(f'reservation_files/reservations({seed}).csv', 'w', newline='') as csvfile:
        fieldnames = result[0].keys()  # Extract headers from the first dictionary
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(result)
    return f'reservation_files/reservations({seed}).csv'


def plot_reservation(res_list):

    time_totals = [defaultdict(int),defaultdict(int),defaultdict(int),defaultdict(int)]
    for reservation in res_list:
        if reservation['num_people'] <= 8:
            time_totals[0][reservation["reservation_time"]] += reservation["num_people"]
        if reservation['num_people'] <= 6:
            time_totals[1][reservation["reservation_time"]] += reservation["num_people"]
        if reservation['num_people'] <= 4:
            time_totals[2][reservation["reservation_time"]] += reservation["num_people"]
        if reservation['num_people'] <= 2:
            time_totals[3][reservation["reservation_time"]] += reservation["num_people"]
    times = [None,None,None,None]
    total_people = [None,None,None,None]
    for i in range(4):
        times[i] = sorted(time_totals[i].keys())
        total_people[i] = [time_totals[i][time] for time in times[i]]

    # Create the bar graph
    plt.figure(figsize=(10, 6))
    plt.bar(times[0], total_people[0], width=15, color='blue', edgecolor='black',label="8")
    plt.bar(times[1], total_people[1], width=15, color='green', edgecolor='black',label="6" )
    plt.bar(times[2], total_people[2], width=15, color='red', edgecolor='black', label="4")
    plt.bar(times[3], total_people[3], width=15, color='yellow', edgecolor='black', label="2")

    # Labeling the axes
    plt.xlabel('Time')
    plt.xlim(0,360)
    plt.ylabel('Total People')
    plt.title('Total People at Each Reservation Time')
    plt.legend()

    # Display the plot
    plt.show()
if __name__ == "__main__":
    create_reservation_list(1,360,20,2,[80,220])