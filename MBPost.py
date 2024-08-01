"""
This file contains table layouts and arguements for specific restaurants
"""
from ClassDefinitions import *

MBPPOST_TABLES = [

    #1 - 12 tables
    Table(((1, 1), (2, 2)), 2, 8, 'regular table', [], None, TableStatus.READY),
    Table(((1, 3), (2, 4)), 2, 8, 'regular table', [], None, TableStatus.READY),
    Table(((1, 5), (2, 6)), 2, 8, 'regular table', [], None, TableStatus.READY),
    Table(((1, 7), (2, 8)), 2, 8, 'regular table', [], None, TableStatus.READY),
    Table(((1, 9), (2, 10)), 2, 8, 'regular table', [], None, TableStatus.READY),
    Table(((1, 11), (2, 12)), 2, 8, 'regular table', [], None, TableStatus.READY),
    Table(((1, 13), (2, 14)), 2, 8, 'regular table', [], None, TableStatus.READY),
    Table(((1, 15), (2, 16)), 2, 8, 'regular table', [], None, TableStatus.READY),
    Table(((1, 17), (2, 18)), 2, 8, 'regular table', [], None, TableStatus.READY),
    Table(((1, 19), (2, 20)), 2, 8, 'regular table', [], None, TableStatus.READY),
    Table(((1, 21), (2, 22)), 2, 8, 'regular table', [], None, TableStatus.READY),
    Table(((1, 23), (2, 24)), 2, 8, 'regular table', [], None, TableStatus.READY),

    #20s
    Table(((4, 23), (5, 24)), 2, 8, 'regular table', [], None, TableStatus.READY),
    Table(((6, 23), (7, 24)), 2, 8, 'regular table', [], None, TableStatus.READY),
    Table(((8, 23), (9, 24)), 2, 8, 'regular table', [], None, TableStatus.READY),
    Table(((10, 23), (11, 24)), 2, 8, 'regular table', [], None, TableStatus.READY),

    Table(((4, 17), (6, 19)), 6, 8, 'regular table', [], None, TableStatus.READY),
    Table(((8, 17), (10, 19)), 6, 8, 'regular table', [], None, TableStatus.READY),

    #30s
    Table(((5, 1), (6, 2)), 2, 8, 'regular table', [], None, TableStatus.READY),
    Table(((5, 3), (6, 4)), 2, 8, 'regular table', [], None, TableStatus.READY),
    Table(((5, 5), (6, 6)), 2, 8, 'regular table', [], None, TableStatus.READY),
    Table(((5, 7), (6, 8)), 2, 8, 'regular table', [], None, TableStatus.READY),
    Table(((5, 9), (6, 10)), 2, 8, 'regular table', [], None, TableStatus.READY),
    Table(((5, 11), (6, 12)), 2, 8, 'regular table', [], None, TableStatus.READY),

    #40s
    Table(((5, 13), (7, 14)), 2, 8, 'regular table', [], None, TableStatus.READY),
    Table(((8, 13), (10, 14)), 2, 8, 'regular table', [], None, TableStatus.READY),
    Table(((15, 13), (17, 14)), 2, 8, 'regular table', [], None, TableStatus.READY),
    Table(((18, 13), (20, 14)), 2, 8, 'regular table', [], None, TableStatus.READY),

    #Atrium talbes

    Table(((8, 1), (10, 3)), 4, 8, 'regular table', [], None, TableStatus.READY),
    Table(((8, 4), (10, 6)), 4, 8, 'regular table', [], None, TableStatus.READY),
    Table(((8, 7), (10, 9)), 4, 8, 'regular table', [], None, TableStatus.READY),

    Table(((11, 4), (12, 5)), 2, 8, 'regular table', [], None, TableStatus.READY),
    Table(((11, 6), (12, 7)), 2, 8, 'regular table', [], None, TableStatus.READY),

    Table(((13, 1), (15, 3)), 4, 8, 'regular table', [], None, TableStatus.READY),
    Table(((13, 4), (15, 6)), 4, 8, 'regular table', [], None, TableStatus.READY),
    Table(((13, 7), (15, 9)), 4, 8, 'regular table', [], None, TableStatus.READY),


    #80s
    Table(((21, 13), (23, 15)), 4, 8, 'regular table', [], None, TableStatus.READY),
    Table(((21, 16), (23, 18)), 4, 8, 'regular table', [], None, TableStatus.READY),
    Table(((21, 19), (23, 21)), 4, 8, 'regular table', [], None, TableStatus.READY),

    #90s
    Table(((17, 7), (19, 9)), 6, 8, 'regular table', [], None, TableStatus.READY),

    Table(((21, 7), (23, 9)), 4, 8, 'regular table', [], None, TableStatus.READY),

    Table(((21, 19), (23, 21)), 4, 8, 'regular table', [], None, TableStatus.READY),

]

table1 = Table(((1, 1), (2, 2)), 2, 8, 'regular table', [], None, TableStatus.READY)
table2 = Table(((3, 3), (4, 4)), 2, 8, 'regular table', [], None, TableStatus.READY)
table3 = Table(((5, 5), (6, 6)), 2, 8, 'regular table', [], None, TableStatus.READY)
table4 = Table(((7, 7), (8, 8)), 2, 8, 'regular table', [], None, TableStatus.READY)

SMALL_TABLES = [
    table1,table2,table3,table4
]