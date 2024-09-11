
from enum import Enum
from datetime import datetime, timedelta
import pygame
# Game manager to handle screen switching
# Base class for screens
class LevelSettings(object):
    def __init__(self,tables,max_party_size,max_time,max_wait_list,max_res_list,combinable_tables):
        """
        :param tables: setup of the tables
        :param max_party_size: The maximum num of people in one given party
        :param max_time: the length of the level in time units
        :param max_wait_list: how long the maximum wait_list can be
        :param max_res_list:  how long the maximum reservation list can be
        :param combinable_tables: list of tuples of tables that can be combined
        """
        self.tables = tables
        self.max_party_size = max_party_size
        self.max_time = max_time
        self.max_wait_list = max_wait_list
        self.max_res_list = max_res_list
        for tuple in combinable_tables:
            tuple[0].make_combinable_with(tuple[1])



class TableStatus(Enum):
    READY = 0
    DIRTY = 1
    OCCUPIED = 2
    COMBINED = 3

class Table(object):
    def __init__(self, number,footprint, size_px , desirability, type, combinable_with, party, status,clean_time=30,clean_progress=0):
        """
        :param footprint: The footprint the table will occupy on the floor plan specified as a (x,y) to (x2,y2)
            which are the top left and bottom right coords of the table

        :param size_px: The amount of people the table can hold, specified as an Int

        :param desirability: rating specified as an int in range 0-10

        :param type: e.g. hightop, bar, couch, regular table, specified as a String

        :param combinable_with: list of tables that this table can be combined with, as a list of other Tables

        :param party: handle for a Party object that is associated with the table

        :param status: status of the table itself, can be ready or dirty etc.

        :param clean_time: float representing how long the table takes to get cleaned (in seconds)

        :param clean_progress: float representing the status of the table being cleaned (in seconds
        """
        self.number = number
        self.footprint = footprint  # (x, y) to (x2, y2)
        self.size_px = size_px  # Integer
        self.desirability = desirability  # Integer (0-10)
        self.type = type  # String
        self.combinable_with = combinable_with  # List of Table objects
        self.combined_with = []  # set of Table objects
        self.party = party  # Party object
        self.status = status  # Table status enum
        self.clean_time = clean_time
        self.clean_progress = clean_progress


    def __repr__(self):
        return (f"Table(footprint={self.footprint}, size_px={self.size_px}, "
                f" desirability={self.desirability}, "
                f"type='{self.type}', combinable_with={self.combinable_with}, "
                f"party={self.party})")

    def __lt__(self, other):
        return id(self) < id(other)

    def __gt__(self, other):
        return id(self) > id(other)

    def __eq__(self, other):
        return id(self) == id(other)

    def reset(self):
        self.status = TableStatus.READY
        self.combined_with = []

    def combined(self):
        """
        :return: returns true if combined and false otherwise
        """
        return len(self.combined_with) != 0

    def uncombine_with(self,other_table):
        """
        :param other_table: table to take apart from
        :return: reward for the action and done variable
        """
        # This method works both directions so no need to check other_table.can_combine_with(self)
        assert(self.can_uncombine_with(other_table))

        if self.status == TableStatus.READY and other_table.status == TableStatus.COMBINED:
            other_table.status = TableStatus.READY
        elif self.status == TableStatus.COMBINED and other_table.status == TableStatus.READY:
            self.status = TableStatus.READY
        else:
            """ 
            They are both combined so need to see where the head node is along the chain
            """
            if self.has_ready_node([other_table]):
                other_table.status = TableStatus.READY
            else:
                self.status = TableStatus.READY

        try:
            self.combined_with.remove(other_table)
        except:
            pass
        try:
            other_table.combined_with.remove(self)
        except:
            pass

        return -1 ,False

    def combine_with(self,other_table):
        """
        :param other_table: table to combine with
        :return: None
        """
        # This method works both directions so no need to check other_table.can_combine_with(self)
        assert(self.can_combine_with(other_table))

        # If two ready tables get combined we only make one combined
        if self.status == TableStatus.READY and other_table.status == TableStatus.READY:
            other_table.status = TableStatus.COMBINED

        elif self.status == TableStatus.READY:
            self.status = TableStatus.COMBINED
        elif other_table.status == TableStatus.READY:
            other_table.status = TableStatus.COMBINED

        # Otherwise both are combined so need to update one of the READY nodes to OCCUPIED
        else:
            self.change_ready_node(TableStatus.COMBINED,[])

        self.combined_with.append(other_table)
        other_table.combined_with.append(self)

        return -1 ,False

    def make_combinable_with(self, other_table):

        if other_table not in self.combinable_with:
            self.combinable_with.append(other_table)
        if self not in other_table.combinable_with:
            other_table.combinable_with.append(self)

    def make_uncombinable_with(self, other_table):

        if other_table in self.combinable_with:
            self.combinable_with.remove(other_table)
        if self in other_table.combinable_with:
            other_table.combinable_with.remove(self)

    def can_uncombine_with(self, other_table):
        """
        Check if this table can un-combine with another table.

        :param other_table: The other table to check.
        :return: True if combinable, False otherwise.
        """

        if other_table not in self.combined_with or other_table not in self.combined_with:
            return False

        return True

    def can_combine_with(self, other_table):
        """
        Check if this table can combine with another table.

        :param other_table: The other table to check.
        :return: True if combinable, False otherwise.
        """
        # Make sure both table structures head nodes are READY
        if not self.has_ready_node([]) or not other_table.has_ready_node([]):
            return False

        # Tables have already been combined
        if self.is_connected_to(other_table,[]) or other_table.is_connected_to(self,[]):
            return False

        return other_table in self.combinable_with and self in other_table.combinable_with

    def is_connected_to(self,other_table,seen_tables=[]):
        for table in self.combined_with:
            if table in seen_tables:
                continue
            if table == other_table:
                return True
            seen_tables.append(self)
            if table.is_connected_to(other_table,seen_tables):
                return True
        return False

    def change_ready_node(self,status,seen_tables=[]):
        if self.status == TableStatus.READY:
            return True
        for table in self.combined_with:
            if table in seen_tables:
                continue
            if table.status == TableStatus.READY:
                table.status = status
                return True
            seen_tables.append(self)
            return table.has_ready_node(seen_tables)
        return False

    def has_ready_node(self,seen_tables=[]):

        if self.status == TableStatus.READY:
            return True
        for table in self.combined_with:
            if table in seen_tables:
                continue
            if table.status == TableStatus.READY:
                return True
            seen_tables.append(self)
            return table.has_ready_node(seen_tables)
        return False

    def get_combined_size(self,seen_tables=[]):

        sum = self.size_px
        for table in self.combined_with:
            if table in seen_tables:
                continue
            new_seen = seen_tables
            new_seen.append(self)
            sum += table.get_combined_size(new_seen)
        return sum


    def is_available(self):
        """
        Check if the table is available (i.e., not associated with a party).

        :return: True if available, False otherwise.
        """
        return self.party is None

    def assign_party(self, party):
        """
        Assign a party to the table.

        :param party: The party to assign.
        """
        self.party = party
        self.update_party_status(PartyStatus.SEATED)
        self.status = TableStatus.OCCUPIED

        # Update the status of all other tables in our combined table
        for table in self.combined_with:
            table.status = TableStatus.COMBINED

    def remove_party(self):
        """
        Remove the party from the table, making it available.
        """
        self.party = None

    def update_party_status(self,status):
        """
         Update the status of the party that is at the table if availible
         :param status: The status to update to, of type PartyStatus
        """
        if self.party:
            self.party.update_status(status)


class PartyStatus(Enum):
    NONE = 0
    ARRIVED = 1
    SEATED = 2
    APPS = 3
    MAIN_COURSE = 4
    DESSERT = 5
    CHECK_DROPPED = 6
    LEFT = 7


class Party(object):
    def __init__(self,name, num_people, reservation, table_pref,pref_locked, status, arrival_time, sat_time, leave_time, happiness, dine_time,meal_split):
        """
        :param name: The name of the party specified as a string

        :param num_people: The amount of people in the party specified as an Int

        :param reservation: A reservation object that is tied to the party if it exists, if a walk-in this will be None

        :param table_pref: The number of the preferred table.

        :param table_locked: If this is true then happiness will be 0 if not sat at table_prefered

        :param status: Where the party is in their experience, whether it be main course or dessert. Specified by an enum of party_status

        :param arrival_time: Time when the party arrives, is a String

        :param sat_time: Time when party seats, is a String

        :param leave_time: Time when party leaves, is a String

        :param happiness: Rating of 1-10 based on how close the quoted wait time was to the actual time to seat

        :param dine_time: string representing how long it will take this party to eat their meal represented in minutes

        :param meal_split: string of form - "a:b:c:d:e" where a is time it takes in state 1 (SEATED) and e is state 5 (CHECK_DROPPED) in the form of percentages of total dine_time
        """
        self.name = name # String
        self.num_people = num_people  # Integer
        self.reservation = reservation  # Reservation object or None
        self.table_pref = table_pref
        self.pref_locked = pref_locked
        self.status = PartyStatus(status)  # PartyStatus enum
        self.arrival_time = arrival_time  # String
        self.sat_time = sat_time  # String
        self.leave_time = leave_time  # String
        self.happiness = happiness  # Integer (1-10)
        self.dine_time = dine_time  #  Integer (minutes)
        self.meal_split = meal_split # String (a:b:c:d:e)


    def __repr__(self):
        return (f"Party(num_people={self.num_people}, name={self.name})")

    def update_status(self, new_status):
        """
        Update the status of the party.

        :param new_status: The new status to update to, should be a value from PartyStatus.
        """
        self.status = PartyStatus(new_status)

    def update_seated_status(self,seated_time,busyness,server_busyness):
        """
        Updates the status to reflect the meal_split numbers
        :return: new status the party was changed to
        """
        # Our range is from 0 to dine_time
        time_counter = 0
        time_max = self.dine_time * busyness * server_busyness

        #This is the range of indexes in the PartyStatus enum we want to capture
        status_index = [2,3,4,5,6]

        values = self.meal_split.split(":")
        for index, val in enumerate(values):
            percentage = float(val) / 100
            time_counter += percentage * time_max
            if seated_time <= time_counter:
                new_status = PartyStatus(status_index[index])
                self.update_status(new_status)
                return new_status

        self.update_status(PartyStatus.CHECK_DROPPED)
        return PartyStatus.CHECK_DROPPED




    def calculate_wait_time(self):
        """
        Calculate the wait time from arrival to being seated.

        :return: The wait time in minutes as an integer.
        """
        if self.arrival_time and self.sat_time:
            arrival = datetime.strptime(self.arrival_time, '%H:%M')
            sat = datetime.strptime(self.sat_time, '%H:%M')
            wait_time = (sat - arrival).seconds // 60
            return wait_time
        return None

    def calculate_duration(self):
        """
        Calculate the duration of the party's stay from being seated to leaving.

        :return: The duration in minutes as an integer.
        """
        if self.sat_time and self.leave_time:
            sat = datetime.strptime(self.sat_time, '%H:%M')
            leave = datetime.strptime(self.leave_time, '%H:%M')
            duration = (leave - sat).seconds // 60
            return duration
        return None

    def update_happiness(self, quoted_wait_time):
        """
        Update the happiness rating based on the quoted wait time versus the actual wait time.

        :param quoted_wait_time: The quoted wait time in minutes as an integer.
        """
        actual_wait_time = self.calculate_wait_time()
        if actual_wait_time is not None:
            # Simple algorithm: happiness decreases by 1 point for each 10 minutes difference from quoted time
            difference = abs(quoted_wait_time - actual_wait_time)
            self.happiness = max(1, 10 - (difference // 10))

    def __str__(self):
        return f"{self.num_people} - {self.name}"


class Reservation(object):
    def __init__(self, party_name, num_people, reservation_time, contact_info, special_requests, status,dine_time,meal_split):
        """
        :param party_name: The name of the party making the reservation, specified as a String

        :param num_people: The number of people in the reservation, specified as an Int

        :param reservation_time: The time of the reservation, specified as a String in 'HH:MM' format

        :param contact_info: Contact information for the party, specified as a String

        :param special_requests: Any special requests, specified as a String

        :param status: The status of the reservation, specified as an enum of ReservationStatus
        """
        self.party_name = party_name  # String
        self.num_people = num_people  # Integer
        self.reservation_time = reservation_time  # String
        self.contact_info = contact_info  # String
        self.special_requests = special_requests  # String
        self.status = ReservationStatus(status)  # ReservationStatus enum
        self.dine_time = dine_time
        self.meal_split = meal_split

    def __repr__(self):
        return (f"Reservation(party_name='{self.party_name}', num_people={self.num_people}, "
                f"reservation_time='{self.reservation_time}', contact_info='{self.contact_info}', "
                f"special_requests='{self.special_requests}', status={self.status})")

    def update_status(self, new_status):
        """
        Update the status of the reservation.

        :param new_status: The new status to update to, should be a value from ReservationStatus.
        """
        self.status = ReservationStatus(new_status)

    def get_reservation_time_as_datetime(self):
        """
        Get the reservation time as a datetime object for comparison.

        :return: The reservation time as a datetime object.
        """
        return datetime.strptime(self.reservation_time, '%H:%M')

    def is_reservation_upcoming(self, current_time):
        """
        Check if the reservation is upcoming based on the current time.

        :param current_time: The current time as a datetime object.
        :return: True if the reservation is upcoming, False otherwise.
        """
        reservation_time = self.get_reservation_time_as_datetime()
        return reservation_time > current_time

    def calculate_time_until_reservation(self, current_time):
        """
        Calculate the time until the reservation from the current time.

        :param current_time: The current time as a datetime object.
        :return: The time until the reservation in minutes as an integer.
        """
        reservation_time = self.get_reservation_time_as_datetime()
        time_until_reservation = (reservation_time - current_time).seconds // 60
        return time_until_reservation
    def __str__(self):
        return f"{self.party_name} reserved at {self.reservation_time} for {self.num_people} people"

class ReservationStatus(Enum):
    PENDING = 0
    CONFIRMED = 1
    SEATED = 2
    CANCELED = 3
    WALK_IN = 4


class Check(object):
    def __init__(self, open_time, close_time, amount):
        """
        :param open_time: The time when the check was opened, specified as a String in 'HH:MM' format

        :param close_time: The time when the check was closed, specified as a String in 'HH:MM' format

        :param amount: The total amount of the check, specified as a float
        """
        self.open_time = open_time  # String
        self.close_time = close_time  # String
        self.amount = amount  # Float

    def __repr__(self):
        return (f"Check(open_time='{self.open_time}', close_time='{self.close_time}', amount={self.amount})")

    def get_open_time_as_datetime(self):
        """
        Get the open time as a datetime object for comparison.

        :return: The open time as a datetime object.
        """
        return datetime.strptime(self.open_time, '%H:%M')

    def get_close_time_as_datetime(self):
        """
        Get the close time as a datetime object for comparison.

        :return: The close time as a datetime object.
        """
        return datetime.strptime(self.close_time, '%H:%M')

    def calculate_duration(self):
        """
        Calculate the duration of the check from opening to closing.

        :return: The duration in minutes as an integer.
        """
        open_time = self.get_open_time_as_datetime()
        close_time = self.get_close_time_as_datetime()
        duration = (close_time - open_time).seconds // 60
        return duration

    def is_open(self, current_time):
        """
        Check if the check is currently open based on the current time.

        :param current_time: The current time as a datetime object.
        :return: True if the check is open, False otherwise.
        """
        close_time = self.get_close_time_as_datetime()
        return current_time < close_time

class UniversalClock:
    def __init__(self, start_time, speed_factor=3):
        self.current_time = start_time
        self.speed_factor = speed_factor
    def update(self):
        self.current_time += self.speed_factor
        return False
    def set_speed(self, speed_factor):
        self.speed_factor = speed_factor

    def get_time_str(self):
        return str(self.current_time)


class PartyPoolManager():
    def __init__(self,num_pools,max_amounts):
        """

        :param num_pools: amount of party pools to init to
        :param max_amounts: list containing values that are max size of each pool i.e. [2,4,6,8]
        """
        max_amounts = sorted(max_amounts)
        self.max_amounts = max_amounts
        self.pools = []
        for i in range(num_pools):
            self.pools.append(PartyPool(party_size=max_amounts[i]))

    def find_pool_for_size(self,size):
        for i, amt in enumerate(self.max_amounts):
            if size <= amt:
                return self.pools[i]


class PartyPool(set):
    def __init__(self, *args,party_size):
        super().__init__(*args)
        # Initialize additional attributes if needed
        self.party_size = party_size

    def inspect_party(self):
        """

        :return: Gives the next party that will be popped out of the pool but notable does not remove it from the pool, merely for info purposes
        """
        best = None
        for party in self:
            if self._get_most_urgent(best,party) == party:
                best = party
        return best

    def _get_most_urgent(self,party1,party2):
        """
        :param party1: party 1
        :param party2: party 2
        :return: The most urgent party based on how close they are to reservation time if they are a reservation
        Based on arrival time if they are a walk_in
        """
        if party1 == None: return party2
        if party2 == None: return party1
        if party1.reservation:
            #Shouldnt have parties with no reservation and a reservation in same pool
            if not party2.reservation: AssertionError(f"Party {party1} and Party {party2} are in the same pool but have conflicting reservations")

            if party1.reservation.reservation_time < party2.reservation.reservation_time:
                return party1
            else:
                return party2

        #If we dont have a reservation we do it on arrival time
        else:
            if party1.arrival_time < party2.arrival_time:
                return party1
            else:
                return party2


    def get_party(self):
        """
        Gets a party from the party pool WITHOUT replacement
        :return: party selected
        """
        best = None
        for party in self:
            if self._get_most_urgent(best,party) == party:
                best = party
        self.remove(best)
        return best

