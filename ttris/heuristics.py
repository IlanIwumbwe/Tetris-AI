import numpy as np

class Heuristics:
    def __init__(self, columns, rows):
        self.width = columns
        self.height = rows
        self.field = [[0 for _ in range(self.width)] for _ in range(self.height)]
        self.valids = [(row, col) for row in range(self.height) for col in range(self.width)]

    def update_field(self, field):  # get new field with updated 1s
        self.field = field

    # height of one column
    def column_height(self, column):
        for row in range(self.height):
            if self.field[row][column] == 1:
                return self.height - row
        return 0

    # max height
    def max_height(self):
        return max([self.column_height(col) for col in range(self.width)])

    def min_height(self):
        return min([self.column_height(col) for col in range(self.width)])

    # number of holes in a column
    def column_holes(self, column):
        col_height = self.column_height(column)
        holes = 0

        for row in range(self.height - 1, -1, -1):
            if self.field[row][column] == 0 and (self.height - row) < col_height:
                holes += 1

        return holes

    # number of holes in all columns
    def total_holes(self):
        return sum([self.column_holes(col) for col in range(self.width)])

    # bumpiness of terrain
    def bumpiness(self):
        col_heights = [self.column_height(col) for col in range(self.width)]
        total = 0

        for i in range(len(col_heights) - 2):
            total += abs(col_heights[i] - col_heights[i + 1])

        return total

    # std dev of heights
    def std_heights(self):
        heights = [self.column_height(col) for col in range(self.width)]
        return np.std(heights)

    # number of pits
    def pits(self):
        pits = 0
        for col in range(self.width):
            for row in range(self.height):
                if all([self.field[row][col] == 0]):
                    pits += 1

        return pits

    # row transitions
    def row_transitions(self):
        transitions = 0
        for row in self.field:
            if 1 in row:
                for col in range(self.width-1):
                    if row[col] == 0 and row[col+1] == 1:
                        transitions += 1
                    elif row[col] == 1 and row[col+1] == 0:
                        transitions += 1
        return transitions

    # col transitions
    def col_transitions(self):
        transitions = 0

        for col in range(self.width):
            for row in range(self.height-1):
                if self.field[row][col] == 1 and self.field[row+1][col] == 0:
                    transitions += 1
                elif self.field[row][col] == 0 and self.field[row+1][col] == 1:
                    transitions += 1

        return transitions

    def total_height(self):
        return sum([self.column_height(i) for i in range(self.width)])

    # lines cleared
    def lines_cleared(self):
        lines = 0

        for row in self.field:
            if 0 not in row:
                lines += 1

        return lines

    def print_stats(self):
        print(f'a. Max H: {self.max_height()} \nb. Total Holes: {self.total_holes()} \nc. Bumpiness: {self.bumpiness()} '
              f'\nd. Heights stdDEV:{self.std_heights()} \nPits: {self.pits()}')

    def print_num_grid(self):
        print(' NEW FRAME ....................................')
        for i in self.field:
            print(i)

    def get_heuristics(self):
        # grid_data = self.empty()
        return [self.total_height(), self.total_holes(), self.bumpiness(), self.lines_cleared(), self.row_transitions(), self.std_heights(), self.pits(), self.col_transitions()]

    # number of blocks
    def total_blocks(self):
        total = 0
        for i in range(self.height):
            for col in self.field[i]:
                if col == 1:
                    total += 1

        return total
