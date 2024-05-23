
from enum import Enum
from datetime import datetime

class Table(object):
    def __init__(self, footprint, size_px , desirability, type, combinable_with, party):
        """
        :param footprint: The footprint the table will occupy on the floor plan specified as a (x,y) to (x2,y2)
            which are the top left and bottom right coords of the table

        :param size_px: The amount of people the table can hold, specified as an Int

        :param desirability: rating specified as an int in range 0-10

        :param type: e.g. hightop, bar, couch, regular table, specified as a String

        :param combinable_with: list of tables that this table can be combined with, as a list of other Tables

        :param party: handle for a Party object that is associated with the table
        """
        self.footprint = footprint  # (x, y) to (x2, y2)
        self.size_px = size_px  # Integer
        self.desirability = desirability  # Integer (0-10)
        self.type = type  # String
        self.combinable_with = combinable_with  # List of Table objects
        self.party = party  # Party object

    def __repr__(self):
        return (f"Table(footprint={self.footprint}, size_px={self.size_px}, "
                f"location={self.location}, desirability={self.desirability}, "
                f"type='{self.type}', combinable_with={self.combinable_with}, "
                f"party={self.party})")

    def can_combine_with(self, other_table):
        """
        Check if this table can combine with another table.

        :param other_table: The other table to check.
        :return: True if combinable, False otherwise.
        """
        return other_table in self.combinable_with

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

    def remove_party(self):
        """
        Remove the party from the table, making it available.
        """
        self.party = None


class PartyStatus(Enum):
    ARRIVED = "Arrived"
    SEATED = "Seated"
    ORDERED = "Ordered"
    MAIN_COURSE = "Main Course"
    DESSERT = "Dessert"
    LEFT = "Left"


class Party(object):
    def __init__(self, num_people, reservation, checks, status, arrival_time, sat_time, leave_time, happiness):
        """
        :param num_people: The amount of people in the party specified as an Int

        :param reservation: A reservation object that is tied to the party if it exists, if a walk-in this will be None

        :param checks: A list of checks for the current table, is a list of Check objects

        :param status: Where the party is in their experience, whether it be main course or dessert. Specified by an enum of party_status

        :param arrival_time: Time when the party arrives, is a String

        :param sat_time: Time when party seats, is a String

        :param leave_time: Time when party leaves, is a String

        :param happiness: Rating of 1-10 based on how close the quoted wait time was to the actual time to seat
        """
        self.num_people = num_people  # Integer
        self.reservation = reservation  # Reservation object or None
        self.checks = checks  # List of Check objects
        self.status = PartyStatus(status)  # PartyStatus enum
        self.arrival_time = arrival_time  # String
        self.sat_time = sat_time  # String
        self.leave_time = leave_time  # String
        self.happiness = happiness  # Integer (1-10)

    def __repr__(self):
        return (f"Party(num_people={self.num_people}, reservation={self.reservation}, checks={self.checks}, "
                f"status={self.status}, arrival_time='{self.arrival_time}', sat_time='{self.sat_time}', "
                f"leave_time='{self.leave_time}', happiness={self.happiness})")

    def update_status(self, new_status):
        """
        Update the status of the party.

        :param new_status: The new status to update to, should be a value from PartyStatus.
        """
        self.status = PartyStatus(new_status)

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




class Reservation(object):
    def __init__(self, party_name, num_people, reservation_time, contact_info, special_requests, status):
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


class ReservationStatus(Enum):
    PENDING = "Pending"
    CONFIRMED = "Confirmed"
    SEATED = "Seated"
    CANCELLED = "Cancelled"


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

