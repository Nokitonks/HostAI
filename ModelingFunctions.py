"""
Functions that calculate things like how busy is the restaurant in order to give the model more lifelike
simulations
"""
from ClassDefinitions import TableStatus, PartyStatus
import numpy as np
from utils.helperFunctions import translate
def get_busyness(env)->float:
    """
    :param env: env to evaluate
    :return: The busyness represented as a number from 0-1 where 1 is most busy and 0 is least busy
    """

    total_tables = len(env.tables)
    busy_tables = 0
    same_course = dict()

    for table in env.tables:
        if table.status != TableStatus.READY or table.status != TableStatus.DIRTY:
            if table.status == TableStatus.COMBINED:
                busy_tables += 1
            elif table.status == TableStatus.OCCUPIED:

                try:
                    same_course[table.party.status] += 1
                except KeyError:
                    same_course[table.party.status] = 0

    total_same_course = len(same_course.keys())
    max = 1 #All tables are on the same course
    var = np.var(list(same_course.values()))
    if var > 100: var = 100
    course = 1 / var

    amt_busy = translate(busy_tables, 0,total_tables,0.7,1.3)
    amt_same = translate(course, max,-20,0.1,0)
    return round(amt_busy+amt_same,3)
