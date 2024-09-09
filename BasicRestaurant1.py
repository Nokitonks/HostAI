from ClassDefinitions import *

class MBPostTables(object):
    def __init__(self):


        # 1 - 12 tables
        table1 = Table(1,((1, 1), (2, 2)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table2 = Table(2,((1, 3), (2, 4)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table3 = Table(3,((1, 5), (2, 6)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table4 = Table(4,((1, 7), (2, 8)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table5 = Table(5,((1, 9), (2, 10)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table6 = Table(6,((1, 11), (2, 12)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table7 = Table(7,((1, 13), (2, 14)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table8 = Table(8,((1, 15), (2, 16)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table9 = Table(9,((1, 17), (2, 18)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table10 = Table(10,((1, 19), (2, 20)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table11 = Table(11,((1, 21), (2, 22)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table12 = Table(12,((1, 23), (2, 24)), 2, 8, 'regular table', [], None, TableStatus.READY)

        table1.make_combinable_with(table2)
        table2.make_combinable_with(table3)
        table3.make_combinable_with(table4)
        table4.make_combinable_with(table5)
        table5.make_combinable_with(table6)
        table6.make_combinable_with(table7)
        table7.make_combinable_with(table8)
        table8.make_combinable_with(table9)
        table9.make_combinable_with(table10)
        table10.make_combinable_with(table11)
        table11.make_combinable_with(table12)
        # 20s
        table20 = Table(20,((4, 23), (5, 24)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table21 = Table(21,((6, 23), (7, 24)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table22 = Table(22,((8, 23), (9, 24)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table23 = Table(23,((10, 23), (11, 24)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table24 = Table(24,((4, 17), (6, 19)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table25 = Table(25,((8, 17), (10, 19)), 6, 8, 'regular table', [], None, TableStatus.READY)
        table26 = Table(26,((8, 17), (10, 19)), 6, 8, 'regular table', [], None, TableStatus.READY)

        table20.make_combinable_with(table21)
        table21.make_combinable_with(table22)
        table22.make_combinable_with(table23)
        table23.make_combinable_with(table24)

        # 30s
        table30 = Table(30,((5, 1), (6, 2)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table31 = Table(31,((5, 3), (6, 4)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table32 = Table(32,((5, 5), (6, 6)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table33 = Table(33,((5, 7), (6, 8)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table34 = Table(34,((5, 9), (6, 10)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table35 = Table(35,((5, 11), (6, 12)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table30.make_combinable_with(table31)
        table31.make_combinable_with(table32)
        table32.make_combinable_with(table33)
        table33.make_combinable_with(table34)
        table34.make_combinable_with(table35)

        # 40s
        table40 = Table(40,((5, 13), (7, 14)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table41 = Table(41,((8, 13), (10, 14)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table42 = Table(42,((15, 13), (17, 14)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table43 = Table(43,((18, 13), (20, 14)), 2, 8, 'regular table', [], None, TableStatus.READY)

        # Atrium talbes

        table51 = Table(51,((8, 1), (10, 3)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table52 = Table(52,((8, 4), (10, 6)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table53 = Table(53,((8, 7), (10, 9)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table54 = Table(54,((8, 7), (10, 9)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table55 = Table(55,((8, 7), (10, 9)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table56 = Table(56,((8, 7), (10, 9)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table51.make_combinable_with(table52)
        table52.make_combinable_with(table53)
        table53.make_combinable_with(table54)
        table54.make_combinable_with(table55)
        table55.make_combinable_with(table56)

        table61 = Table(61,((11, 4), (12, 5)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table62 = Table(62,((11, 6), (12, 7)), 2, 8, 'regular table', [], None, TableStatus.READY)

        table71 = Table(71,((13, 1), (15, 3)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table72 = Table(72,((13, 4), (15, 6)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table73 = Table(73,((13, 7), (15, 9)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table74 = Table(74,((13, 1), (15, 3)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table75 = Table(75,((13, 4), (15, 6)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table76 = Table(76,((13, 7), (15, 9)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table71.make_combinable_with(table72)
        table72.make_combinable_with(table73)
        table73.make_combinable_with(table74)
        table74.make_combinable_with(table75)
        table75.make_combinable_with(table76)

        # 80s
        table81 = Table(81,((21, 13), (23, 15)), 4, 8, 'regular table', [], None, TableStatus.READY)
        table82 = Table(82,((21, 16), (23, 18)), 4, 8, 'regular table', [], None, TableStatus.READY)
        table83 = Table(83,((21, 19), (23, 21)), 4, 8, 'regular table', [], None, TableStatus.READY)
        table84 = Table(84,((21, 13), (23, 15)), 4, 8, 'regular table', [], None, TableStatus.READY)
        table85 = Table(85,((21, 16), (23, 18)), 4, 8, 'regular table', [], None, TableStatus.READY)
        table86 = Table(86,((21, 19), (23, 21)), 4, 8, 'regular table', [], None, TableStatus.READY)
        table81.make_combinable_with(table82)
        table82.make_combinable_with(table83)
        table83.make_combinable_with(table84)
        table84.make_combinable_with(table85)
        table85.make_combinable_with(table86)

        # 90s
        table90 = Table(90,((17, 7), (19, 9)), 6, 8, 'regular table', [], None, TableStatus.READY)
        table91 = Table(91,((21, 7), (23, 9)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table92 = Table(92,((21, 19), (23, 21)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table93 = Table(93,((17, 7), (19, 9)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table94 = Table(94,((21, 7), (23, 9)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table95 = Table(95,((21, 19), (23, 21)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table96 = Table(96,((21, 19), (23, 21)), 2, 8, 'regular table', [], None, TableStatus.READY)

        table93.make_combinable_with(table92)
        table94.make_combinable_with(table95)
        table95.make_combinable_with(table96)
        self.tables = [table1, table2, table3, table4, table5, table6, table7, table8, table9, table10, table11, table12,
                       table21,table22,table23,table24, table25,table26,
                       table30,table31,table32,table33,table34,table35,
                       table40,table41,table42,table43,
                       table51,table52,table53,table54,table55,table56,
                       table61,table62,
                       table71,table72,table73,table74,table75,table76,
                       table81,table82,table83,table84,table85,table86,
                       table90,table91,table92,table93,table94,table95,table96]


class BasicRestaurantTables(object):
    def __init__(self):
        # 1 - 4 tables
        table1 = Table(1,((1, 1), (2, 2)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table2 = Table(2,((1, 3), (2, 4)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table3 = Table(3,((1, 5), (2, 6)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table4 = Table(4,((1, 7), (2, 8)), 2, 8, 'regular table', [], None, TableStatus.READY)

        # 5 - 8 tables
        table5 = Table(5,((8, 1), (9, 2)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table6 = Table(6,((8, 3), (9, 4)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table7 = Table(7,((8, 5), (9, 6)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table8 = Table(8,((8, 7), (9, 8)), 2, 8, 'regular table', [], None, TableStatus.READY)

        # 10 and 11 tables
        table10 = Table(10,((4, 2), (6, 4)), 4, 8, 'regular table', [], None, TableStatus.READY)
        table11 = Table(11,((4, 5), (6, 7)), 4, 8, 'regular table', [], None, TableStatus.READY)

        table1.make_combinable_with(table2)
        table2.make_combinable_with(table3)
        table3.make_combinable_with(table4)

        table5.make_combinable_with(table6)
        table6.make_combinable_with(table7)
        table7.make_combinable_with(table8)

        table10.make_combinable_with(table11)

        self.tables = [table1, table2, table3, table4, table5, table6, table7,table8,table10,table11 ]


