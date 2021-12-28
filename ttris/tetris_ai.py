import random
import pygame
from pygame import mixer
from itertools import permutations
import data
import numpy as np

# game assets

I = '....' \
    'xxxx' \
    '....' \
    '....'

S = '....' \
    '.xx.' \
    'xx..' \
    '....'

O = '....' \
    '.xx.' \
    '.xx.' \
    '....'

Z = '....' \
    'xx..' \
    '.xx.' \
    '....'

T = '....' \
    '.x..' \
    'xxx.' \
    '....'

L = '....' \
    '...x' \
    '.xxx' \
    '....'

J = '....' \
    'x...' \
    'xxx.' \
    '....'

pieces = [I, S, O, Z, T, L, J]
str_pieces = {I: 'I', S: 'S', O: 'O', Z: 'Z', T: 'T', L: 'L', J: 'J'}
orig_num_pieces = {I: 1, S: 2, O: 3, Z: 4, T: 5, L: 6, J: 7}
new_num_pieces = {I: 8, S: 9, O: 10, Z: 11, T: 12, L: 13, J: 14}

centres = {I: (1, 1), S: (1, 2), O: (1, 1), Z: (1, 2), T: (1, 2), L: (2, 2), J: (1, 2)}  # some fixes  # 1,2 I , L , J

colours = {I: (51, 255, 255), S: (255, 255, 100), O: (255, 51, 255), Z: (0, 0, 255), T: (255, 128, 0), L: (51, 255, 51),
           J: (255, 0, 0)}

action_space = {pygame.K_DOWN: 'down', pygame.K_RIGHT: 'right', pygame.K_LEFT: 'left', pygame.K_UP: 'cw',
                pygame.K_w: 'ccw', pygame.K_TAB: 'hd', pygame.K_SPACE: 'hold', pygame.K_BACKSPACE: 'unhold'}

BOUNDARY = 1
ROWS = 20 + 2  # 2 extra rows for spawning
COLUMNS = 10

# dimensions
width = 800
height = 650
block_size = 20
play_w = ROWS * block_size
play_h = COLUMNS * block_size

top_left_x = (width - play_w) // 2 - 100
top_left_y = height - play_h - 400

# font and music!
f = 'freesansbold.ttf'

"""
mixer.init()
mixer.music.load('tetris-gameboy-02.ogg')
mixer.music.play(-1)
"""

def valid_space(piece, grid):  # piece is an object
    # valid positions if they're blank
    positions = [[(j, i) for j in range(COLUMNS) if grid[i][j] == (255, 255, 255)] for i in range(ROWS)]
    valid_positions = [tupl for pos in positions for tupl in pos]  # flatten
    x_pos = piece.current_position()

    for i in x_pos:
        if i not in valid_positions:
            return False
    return True


class SRS:
    def __init__(self, piece):  # piece is an object
        self.piece = piece
        self.centre = self.piece.centre
        self.rot_index = self.piece.rot_index

    def rotation(self, clockwise, piece_string):
        piece = [piece_string[i:i + 4] for i in range(0, len(piece_string), 4)]
        c_x, c_y = self.centre

        global_cords = []
        new_global_cords = []

        relative_cords = []
        new_relative_cords = []

        # get global cords
        for ind_x, row in enumerate(piece):
            for ind_y, col in enumerate(row):
                if col == 'x':
                    global_cords.append((ind_y, ind_x))

        # get relative cords to centre
        for x, y in global_cords:
            relative_cords.append((x - c_x, y - c_y))

        if clockwise:
            matrix = data.CLOCKWISE_MATRIX
        if not clockwise:
            matrix = data.ANTICLOCKWISE_MATRIX

        # calculate new relative cord to centre
        for cord in relative_cords:
            new_r_cord = np.dot(matrix, cord)
            for _ in range(self.rot_index):
                new_r_cord = np.dot(new_r_cord, matrix)
            new_relative_cords.append(new_r_cord)

        # get new global cords
        for x, y in new_relative_cords:
            new_global_cords.append((x + c_x, y + c_y))

        return new_global_cords


class Piece:
    def __init__(self, x=None, y=None, piece=None):
        self.x = x
        self.y = y
        self.piece = piece
        self.colour = colours[self.piece] if self.piece is not None else None
        self.rot_index = 0
        self.state = self.piece
        self.clockwise = None
        self.all = [(j, i) for i in range(4) for j in range(4)]
        self.centre = centres[self.piece]
        self.test_state = None
        self.str_id = str_pieces[self.piece]

    def rotate(self, dir):
        rotation_data_for_I = data.Data(self.str_id, self.rot_index).get_data()
        srs = SRS(self)
        new_piece = ['.'] * 16

        if self.str_id != 'I':
            if dir == 'cw':
                self.clockwise = True
            else:
                self.clockwise = False

            rotation_cords = srs.rotation(dir, self.piece)

            for x, y in rotation_cords:
                ind = 4 * y + x
                new_piece[ind] = 'x'

            self.state = ''.join(new_piece)  # at this stage, its a no-kicks-state
        else:
            for x, y in rotation_data_for_I:
                ind = 4 * y + x
                new_piece[ind] = 'x'

            self.state = ''.join(new_piece)

    def current_position(self):  # get grid positions of a passed piece object
        p = [self.state[i:i + 4] for i in range(0, len(self.state), 4)]
        positions = []

        for ind_x, i in enumerate(p):
            for ind_y, j in enumerate(i):
                if j == 'x':
                    positions.append((self.x + ind_y, self.y + ind_x))  # translate 'x' indexes into GRID indexes
        return positions

    def get_config(self):
        grid_positions = self.current_position()

        lowest_block = sorted(grid_positions, key=lambda x: x[1])[-1]
        l_x, l_y = lowest_block

        cords = [(x - l_x, y - l_y) for x, y in grid_positions]

        return self.rot_index, (l_x, l_y), cords, 0


class Collision:
    def __init__(self):  # piece is an object
        self.boundary = [(row, col) for col in range(1) for row in range(24)] + \
                        [(row, col) for col in range(12) for row in range(1)] + \
                        [(row, col) for col in range(11, 12) for row in range(24)] + \
                        [(row, col) for col in range(12) for row in range(23, 24)]
        self.field = [[0 for _ in range(COLUMNS + 2 * BOUNDARY)] for _ in range(ROWS + 2 * BOUNDARY)]
        self.landed = {}

    def create_field(self, landed):
        landed_piece = None

        self.field = [[0 for _ in range(COLUMNS + 2 * BOUNDARY)] for _ in range(ROWS + 2 * BOUNDARY)]
        # create landed positions
        if landed != {}:
            for (x, y), colour in landed.items():
                for piece, piece_colour in colours.items():
                    if colour == piece_colour:
                        landed_piece = piece

                self.field[y + BOUNDARY][x + BOUNDARY] = orig_num_pieces[landed_piece]
                self.landed[(x + BOUNDARY, y + BOUNDARY)] = orig_num_pieces[landed_piece]  # fix here

        # create boundary
        for x, y in self.boundary:
            try:
                self.field[x][y] = 20
            except IndexError:
                print('Boundary positions are wrong')

    def field_state(self):
        landed_and_wall = {}

        for x, y in self.boundary:
            landed_and_wall[(x, y)] = self.field[x][y]

        for (x, y), i in self.landed.items():
            landed_and_wall[(y, x)] = i

        return landed_and_wall

    def show_board(self):
        print('New Frame......................................')
        for row in self.field:
            print(f'{row} \n')


def has_collided(old_state, updated_field):
    return not all([old_state[(x, y)] == updated_field[(x, y)] if x > 0 and y > 0 else True for x, y in old_state.keys()])


class Board:
    def __init__(self, landed, lines, score):
        self.landed = landed
        self.score = score
        self.lines = lines
        self.level = 0

    def create_grid(self):
        # if you want to change colour of grid, change _____ to desired colour! (except tetromino colour)
        GRID = [[(255, 255, 255) for column in range(COLUMNS)] for row in range(ROWS)]

        for i in range(ROWS):
            for j in range(COLUMNS):
                if (j, i) in self.landed:
                    # set colour if position is landed i.e there is a piece there
                    GRID[i][j] = self.landed[(j, i)]
        return GRID

    @staticmethod
    def show_held_piece(surface, held_piece):
        pos_x = top_left_x + play_w - 100
        pos_y = top_left_y

        n_p = [held_piece.piece[i:i + 4] for i in range(0, len(held_piece.piece), 4)]

        # outer rectangle
        # pygame.draw.rect(surface, (100,100,100), (pos_x, pos_y, play_w//2+60, play_h), 3)

        # next piece
        for ind_x, row in enumerate(n_p):
            for ind_y, column in enumerate(row):
                if column == 'x':
                    pygame.draw.rect(surface, held_piece.colour, (
                        pos_x + ind_y * block_size + 20, pos_y + ind_x * block_size + 20, block_size, block_size), 0)
                    pygame.draw.rect(surface, (0, 0, 0), (
                        pos_x + ind_y * block_size + 20, pos_y + ind_x * block_size + 20, block_size, block_size), 2)

    @staticmethod
    def show_next_piece(surface, next_piece):
        pos_x = top_left_x + play_w - 220
        pos_y = top_left_y

        n_p = [next_piece.piece[i:i + 4] for i in range(0, len(next_piece.piece), 4)]

        # outer rectangle
        # pygame.draw.rect(surface, (100,100,100), (pos_x, pos_y, play_w//2+60, play_h), 3)

        # next piece
        for ind_x, row in enumerate(n_p):
            for ind_y, column in enumerate(row):
                if column == 'x':
                    pygame.draw.rect(surface, next_piece.colour, (
                        pos_x + ind_y * block_size + 20, pos_y + ind_x * block_size + 20, block_size, block_size), 0)
                    pygame.draw.rect(surface, (0, 0, 0), (
                        pos_x + ind_y * block_size + 20, pos_y + ind_x * block_size + 20, block_size, block_size), 2)

    @staticmethod
    def render_grid(surface, grid):
        # boundary
        for i in range(ROWS + BOUNDARY + 1):
            for j in range(COLUMNS + BOUNDARY + 1):
                pygame.draw.rect(surface, (100, 100, 100), (top_left_x - (BOUNDARY * block_size) + j * block_size,
                                                            top_left_y - (BOUNDARY * block_size) + i * block_size,
                                                            block_size, block_size), 0)
                pygame.draw.rect(surface, (0, 0, 0), (top_left_x - (BOUNDARY * block_size) + j * block_size,
                                                      top_left_y - (BOUNDARY * block_size) + i * block_size, block_size,
                                                      block_size), 2)

        # convert grid colours to output onto surface
        for ind_x, rows in enumerate(grid):
            for ind_y, colour in enumerate(rows):
                pygame.draw.rect(surface, colour, (
                    top_left_x + (ind_y * block_size), top_left_y + (ind_x * block_size), block_size, block_size), 0)

        # draw black boundaries
        for i in range(ROWS):
            for j in range(COLUMNS):
                if i == 0 or i == 1:  # first 2 rows are for spawing
                    pygame.draw.rect(surface, (50, 100, 255),
                                     (top_left_x + j * block_size, top_left_y + i * block_size, block_size, block_size),
                                     0)
                    pygame.draw.rect(surface, (255, 255, 255),
                                     (top_left_x + j * block_size, top_left_y + i * block_size, block_size, block_size),
                                     2)
                if grid[i][j] != (255, 255, 255):
                    pygame.draw.rect(surface, (0, 0, 0),
                                     (top_left_x + j * block_size, top_left_y + i * block_size, block_size, block_size),
                                     2)

    def clear_rows(self, grid):
        cleared_row = 0
        cleared_rows = 0

        for index in range(ROWS - 1, -1, -1):
            if (255, 255, 255) not in grid[index]:
                cleared_rows += 1
                cleared_row = index
                for column in range(COLUMNS):
                    del self.landed[(column, cleared_row)]  # deletes colours off cleared rows

        # sort all landed positions based on the rows in the grid, then reverse that as we are
        # searching grid from below

        for position in sorted(list(self.landed), key=lambda pos: pos[1], reverse=True):
            col, row = position

            if row < cleared_row:  # if row is above index of row that was cleared:
                new_pos = (col, row + cleared_rows)  # make new position that moves itm down by number of cleared rows

                self.landed[new_pos] = self.landed.pop(
                    position)  # .pop here removes colours from all rows above, and places them in their new positions

        self.lines += cleared_rows
        self.score += score_game(cleared_rows, self.level)

        return cleared_rows


class Piece_Gne:
    def __init__(self, bag):
        self.bag = bag
        self.start_ind = 0

    def generator_function(self):
        permu = list(permutations(self.bag))

        while True:
            for piece in random.choice(permu):
                yield piece

    def pop(self, buffer, generator):
        popped = buffer[self.start_ind]
        self.start_ind = (self.start_ind + 1) % len(buffer)
        buffer[self.start_ind] = next(generator)

        return popped

    def get_piece(self):
        rng = self.generator_function()

        size = 7

        buffer = [next(rng) for _ in range(size)]

        popped = self.pop(buffer, rng)

        if str_pieces[popped] in 'I,O'.split(','):
            piece_obj = Piece(4, 0, popped)
            return piece_obj
        else:
            piece_obj = Piece(4, 0, popped)
            return piece_obj


def move_works(collision, piece, move):  # collision is an object
    srs = SRS(piece)

    if action_space[move] == 'cw':
        coll_field = collision.field
        old_state = collision.field_state()

        updated_field = {}

        piece.rot_index = Mod(piece.rot_index + 1, 4)
        rotation_cords = get_rotation_cords(srs, 'cw', piece)

        # print(rotation_cords)

        for x, y in rotation_cords:
            try:
                coll_field[y + BOUNDARY][x + BOUNDARY] = new_num_pieces[piece.piece]
            except IndexError:
                print('Yikes')

        for ind_x, row in enumerate(coll_field):
            for ind_y, col in enumerate(row):
                updated_field[(ind_x, ind_y)] = col

        if has_collided(old_state, updated_field):
            return False

    elif action_space[move] == 'ccw':
        coll_field = collision.field
        old_state = collision.field_state()

        updated_field = {}

        piece.rot_index = Mod(piece.rot_index - 1, 4)
        rotation_cords = get_rotation_cords(srs, 'ccw', piece)

        for x, y in rotation_cords:
            try:
                coll_field[y + BOUNDARY][x + BOUNDARY] = new_num_pieces[piece.piece]
            except IndexError:
                print('Yikes')

        for ind_x, row in enumerate(coll_field):
            for ind_y, col in enumerate(row):
                updated_field[(ind_x, ind_y)] = col

        if has_collided(old_state, updated_field):
            return False

    elif action_space[move] == 'down':
        pos = piece.current_position()

        coll_field = collision.field
        old_state = collision.field_state()

        updated_field = {}

        for x, y in pos:
            try:
                coll_field[y + 1 + BOUNDARY][x + BOUNDARY] = new_num_pieces[piece.piece]
            except IndexError:
                print('Yikes')

        for ind_x, row in enumerate(coll_field):
            for ind_y, col in enumerate(row):
                updated_field[(ind_x, ind_y)] = col

        if has_collided(old_state, updated_field):
            return False

    elif action_space[move] == 'right':
        pos = piece.current_position()

        coll_field = collision.field
        old_state = collision.field_state()
        updated_field = {}

        for x, y in pos:
            try:
                coll_field[y + BOUNDARY][x + 1 + BOUNDARY] = new_num_pieces[piece.piece]
            except IndexError:
                print('Yikes')

        for ind_x, row in enumerate(coll_field):
            for ind_y, col in enumerate(row):
                updated_field[(ind_x, ind_y)] = col

        if has_collided(old_state, updated_field):
            return False

    elif action_space[move] == 'left':
        pos = piece.current_position()

        coll_field = collision.field
        old_state = collision.field_state()
        updated_field = {}

        for x, y in pos:
            try:
                coll_field[y + BOUNDARY][x + BOUNDARY - 1] = new_num_pieces[piece.piece]
            except IndexError:
                print('Yikes')

        for ind_x, row in enumerate(coll_field):
            for ind_y, col in enumerate(row):
                updated_field[(ind_x, ind_y)] = col

        if has_collided(old_state, updated_field):
            return False

    return True


def make_move(move, piece, collision):
    if action_space[move] == 'down':
        if move_works(collision, piece, move):
            piece.y += 1
            return 'worked'
        else:
            return 'collided'

    elif action_space[move] == 'right':
        if move_works(collision, piece, move):
            piece.x += 1

    elif action_space[move] == 'left':
        if move_works(collision, piece, move):
            piece.x -= 1

    elif action_space[move] == 'cw':
        if move_works(collision, piece, move) != False and not piece.piece == O:
            piece.rotate('cw')

    elif action_space[move] == 'ccw':
        if move_works(collision, piece, move) != False and not piece.piece == O:
            piece.rotate('ccw')

def get_rotation_cords(srs, dir, piece):
    data_for_I = data.Data('I', piece.rot_index).get_data()

    if piece.str_id == 'I':
        grid_cords = [(piece.x + y, piece.y + x) for x, y in data_for_I]
    else:
        ascii_cords = srs.rotation(True, piece.piece) if dir == 'cw' else srs.rotation(False, piece.piece)
        grid_cords = [(piece.x + y, piece.y + x) for x, y in ascii_cords]

    return grid_cords


def score_game(cleared, level):
    # more to be done
    if cleared == 1:
        return 40 + (40 * level)
    elif cleared == 2:
        return 100 + (100 * level)
    elif cleared == 3:
        return 300 + (300 * level)
    elif cleared == 4:
        return 1200 + (1200 * level)
    else:
        return 0


def get_data(piece, rot_index):
    return data.Data(piece, rot_index).get_data()


def Mod(n, d):  # NUMERATOR, DENOMINATOR
    return (n % d + d) % d


class Tetris:
    def __init__(self):
        self.win = pygame.display.set_mode((width, height))
        # piece generation setup
        self.generate = Piece_Gne(pieces)

        # get starting piece object
        self.current_piece = self.generate.get_piece()

        # control parameters
        self.run = True
        self.show_piece = True
        self.held_piece = None
        self.change_piece = False
        self.hold_piece = False
        self.unhold_piece = False

        # get next piece
        self.next_piece = self.generate.get_piece()

        # game board setup
        self.landed = {}
        self.lines = 0
        self.score = 0
        self.board = Board(self.landed, self.lines, self.score)
        self.collision = Collision()
        self.tetrises = 0
        self.grid = self.board.create_grid()

        # gravity setup
        self.fall_time = 0
        self.fall_speed = 0

        # game clock
        self.clock = pygame.time.Clock()

        self.best_move = None

    def reset(self):
        # self.win = pygame.display.set_mode((width, height))
        # piece generation setup
        self.generate = Piece_Gne(pieces)

        # get starting piece object
        self.current_piece = self.generate.get_piece()

        # control parameters
        self.run = True
        self.show_piece = True
        self.held_piece = None
        self.change_piece = False
        self.hold_piece = False
        self.unhold_piece = False

        # get next piece
        self.next_piece = self.generate.get_piece()

        # game board setup
        self.landed = {}
        self.lines = 0
        self.score = 0
        self.board = Board(self.landed, self.lines, self.score)
        self.collision = Collision()
        self.tetrises = 0
        self.grid = self.board.create_grid()

        # gravity setup
        self.fall_time = 0

        self.best_move = None

    def draw_window(self):  # pass instance of board
        pygame.font.init()

        font = pygame.font.Font(f, 15)

        pos_x = top_left_x + play_w
        pos_y = top_left_y + play_h // 2

        score = font.render(f'Score: {self.board.score}', True, (0, 0, 0))
        lines = font.render(f'Lines: {self.board.lines}', True, (0, 0, 0))
        level = font.render(f'Level: {self.board.level}', True, (0, 0, 0))
        next_text = font.render('NEXT PIECE', True, (0, 0, 0))
        hold_text = font.render('HOLD PIECE', True, (0, 0, 0))

        self.win.blit(score, (pos_x - 200, pos_y + 50))
        self.win.blit(lines, (pos_x - 200, pos_y + 80))
        self.win.blit(level, (pos_x - 200, pos_y + 110))
        self.win.blit(next_text, (pos_x - 200, pos_y - 90))
        self.win.blit(hold_text, (pos_x - 90, pos_y - 90))

        self.board.show_next_piece(self.win, self.next_piece)
        self.board.render_grid(self.win, self.grid)
        if self.held_piece is not None: self.board.show_held_piece(self.win, self.held_piece)
        pygame.display.update()

    def update_scores(self):
        with open('tetris_champs.txt', 'a') as file:
            file.write(f'\nScore: {self.score} ......  Lines: {self.lines}')

    def lost(self):
        # if piece touches top of grid, its a loss
        for pos in self.landed:
            if pos[1] < 2:
                return True
        return False

    def change_state(self):
        # lock position
        for i in self.current_piece.current_position():
            self.landed[i] = self.current_piece.colour

        # clear rows
        cleared = self.board.clear_rows(self.grid)

        if cleared == 4:
            self.tetrises += 1

        show_piece = True

        # update game level
        self.board.level = self.lines // 10

        self.collision = Collision()

        self.collision.create_field(self.landed)

        self.board.show_next_piece(self.win, self.next_piece)

        self.lines, self.score = self.board.lines, self.board.score  # maybe set self.lines = cleared ?

        self.current_piece = self.next_piece

        if [(255, 255, 255)] * 10 == self.grid[1] and [(255, 255, 255)] * 10 == self.grid[0]:
            self.next_piece = self.generate.get_piece()
            self.change_piece = False

    def game_logic(self):
        self.grid = self.board.create_grid()
        self.fall_speed = 0.14 # (board.level * 0.05)

        self.fall_time += self.clock.get_rawtime()
        self.clock.tick()

        if self.fall_time / 1000 > self.fall_speed:
            self.fall_time = 0
            self.current_piece.y += 1

            if not valid_space(self.current_piece, self.grid):  # piece has landed
                self.current_piece.y -= 1
                self.board.score += 1
                self.change_piece = True

        self.win.fill((128, 128, 128))

        pygame.display.set_caption('Tetris')

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.run = False

        piece_positions = self.current_piece.current_position()

        # render on board
        if self.show_piece:
            for x, y in piece_positions:
                if (x, y) not in self.landed:
                    try:
                        self.grid[y][x] = self.current_piece.colour
                    except IndexError:
                        # crushes through the wall
                        # print('wef')
                        self.run = False

        '''
        user wants to hold piece, store it in held piece, change current to next,
        generate new piece to replace next piece

        set hold piece back to false
        '''
        if self.hold_piece:
            self.held_piece = self.current_piece
            self.current_piece = self.next_piece
            self.next_piece = self.generate.get_piece()
            self.hold_piece = False

        '''
        user wants to unhold, piece, set hold pressed back to 0, so we reset state
        make it spawn exactly where current piece was
        make swap
        '''
        if self.unhold_piece:
            self.held_piece.x, self.held_piece.y = self.current_piece.x, self.current_piece.y
            self.current_piece = self.held_piece

            self.held_piece = None
            self.unhold_piece = False

        self.draw_window()

        if self.lost():
            self.run = False

    def reward_info(self):
        return self.best_move[3], self.score, self.run

    # testing computer to make raw key presses based on best move
    def make_ai_move(self):  # collision is an obj
        current_config = self.current_piece.get_config()  # exp : (3, (7, 21), '[(7, 20), (8, 20), (9, 20), (7, 21)]')
        target_config = self.best_move  # exp: (0, (7, 21), '[(6, 19), (7, 19), (7, 20), (7, 21)]', some reward)

        cu_x, cu_y = current_config[1]
        t_x, t_y = target_config[1]

        cu_rot_state = current_config[0]
        t_rot_state = target_config[0]

        diff = cu_rot_state - t_rot_state
        rotation_options = [pygame.K_UP, pygame.K_w]

        if abs(diff) == 2:
            choice = random.choice(rotation_options)
            make_move(choice, self.current_piece, self.collision)
        elif abs(diff) == 1:
            make_move(pygame.K_UP, self.current_piece, self.collision)
        elif abs(diff) == 3:
            make_move(pygame.K_w, self.current_piece, self.collision)

        moves = t_x - cu_x
        if t_x > cu_x:
            for _ in range(abs(moves)):
                make_move(pygame.K_RIGHT, self.current_piece, self.collision)
        else:
            for _ in range(abs(moves)):
                make_move(pygame.K_LEFT, self.current_piece, self.collision)

        # make_move(pygame.K_TAB, self.current_piece, self.collision)


