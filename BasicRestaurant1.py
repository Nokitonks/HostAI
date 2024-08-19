from ClassDefinitions import *


class BasicRestaurantTables(object):
    def __init__(self):
        # 1 - 4 tables
        table1 = Table(((1, 1), (2, 2)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table2 = Table(((1, 3), (2, 4)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table3 = Table(((1, 5), (2, 6)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table4 = Table(((1, 7), (2, 8)), 2, 8, 'regular table', [], None, TableStatus.READY)

        # 5 - 8 tables
        table5 = Table(((8, 1), (9, 2)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table6 = Table(((8, 3), (9, 4)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table7 = Table(((8, 5), (9, 6)), 2, 8, 'regular table', [], None, TableStatus.READY)
        table8 = Table(((8, 7), (9, 8)), 2, 8, 'regular table', [], None, TableStatus.READY)

        # 10 and 11 tables
        table10 = Table(((4, 2), (6, 4)), 4, 8, 'regular table', [], None, TableStatus.READY)
        table11 = Table(((4, 5), (6, 7)), 4, 8, 'regular table', [], None, TableStatus.READY)

        table1.make_combinable_with(table2)
        table2.make_combinable_with(table3)
        table3.make_combinable_with(table4)

        table5.make_combinable_with(table6)
        table6.make_combinable_with(table7)
        table7.make_combinable_with(table8)

        table10.make_combinable_with(table11)

        self.tables = [table1, table2, table3, table4, table5, table6, table7,table8,table10,table11 ]


