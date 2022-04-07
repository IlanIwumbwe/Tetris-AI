import random
import pygame
from pygame import mixer
from itertools import permutations
import data
import numpy as np
import time

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

CLOCKWISE_MATRIX = [[0, 1], [-1, 0]]
ANTICLOCKWISE_MATRIX = [[0, -1], [1, 0]]

pieces = {'I':I, 'S': S, 'O': O, 'Z': Z, 'T': T, 'L': L, 'J': J}
centres = {'I': (1, 1), 'S': (1, 2), 'O': (1, 1), 'Z': (1, 2), 'T': (1, 2), 'L': (2, 2), 'J': (1, 2)}

CYAN = (51, 255, 255)
BLUE = (0, 0, 255)
PURPLE = (225, 21, 132)
YELLOW = (255, 255, 100)
ORANGE = (255, 128, 0)
RED = (255, 0, 0)
GREEN = (51, 255, 51)
WHITE = (255, 255, 255)
GREY = (100, 100, 100)
BLACK = (0, 0, 0)
SPAWN_ROWS = (100, 140, 40) #(50, 100, 255)
BG = (128, 128, 128)

colours = {'I': CYAN, 'S': GREEN, 'O': YELLOW, 'Z': RED, 'T': PURPLE, 'L': ORANGE, 'J': BLUE}

action_space = {pygame.K_DOWN: 'down', pygame.K_RIGHT: 'right', pygame.K_LEFT: 'left', pygame.K_UP: 'cw',
                pygame.K_w: 'ccw', pygame.K_TAB: 'hd', pygame.K_SPACE: 'hold', pygame.K_BACKSPACE: 'unhold', pygame.K_p:'pause'}

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

"""mixer.init()
mixer.music.load('tetris-gameboy-02.ogg')
mixer.music.play(-1)"""

class SRS:
    def __init__(self, piece):  # piece is an object
        self.piece = piece
        self.centre = self.piece.centre
        self.rot_index = self.piece.rot_index
        self.clockwise = True
        self.desired_rot_index = None
        self.boundary = [(row, col) for col in range(1) for row in range(24)] + \
                        [(row, col) for col in range(12) for row in range(1)] + \
                        [(row, col) for col in range(11, 12) for row in range(24)] + \
                        [(row, col) for col in range(12) for row in range(23, 24)]
        self.field = [[0 for _ in range(COLUMNS + 2 * BOUNDARY)] for _ in range(ROWS + 2 * BOUNDARY)]

    def update_field(self, landed):
        self.field = [[0 for _ in range(COLUMNS + 2 * BOUNDARY)] for _ in range(ROWS + 2 * BOUNDARY)]
        # create landed positions
        if landed != {}:
            for (x, y) in landed.keys():
                try:
                    self.field[y+BOUNDARY][x+BOUNDARY] = 1
                except IndexError:
                    print('Error on landed piece placement')
                    pass

        # create boundary
        for x, y in self.boundary:
            try:
                self.field[x][y] = 1
            except IndexError:
                print('Boundary positions are wrong')

    def make_move(self, move):  # collision is an object
        if move == 'cw' or move == 'ccw':
            if move == 'cw':
                state_cords = self.basic_rotation(True)
            else:
                state_cords = self.basic_rotation(False)

            if all([self.field[y + BOUNDARY][x + BOUNDARY] == 0 for x, y in state_cords]):
                self.piece.state_cords = state_cords
                new_state = ['.' for _ in range(16)]

                for x, y in self.piece.state_cords:
                    ind = 4 * y + x
                    new_state[ind] = 'x'

                self.piece.state = ''.join(new_state)

                if move == 'cw':
                    self.piece.rot_index += 1
                else:
                    self.piece.rot_index -= 1

        elif move == 'down':
            pos = self.piece.current_position()

            try:
                return all([self.field[y + BOUNDARY + 1][x + BOUNDARY] == 0 for x, y in pos])
            except IndexError:
                print('WIll try again')
                pass

        elif move == 'right':
            pos = self.piece.current_position()

            try:
                return all([self.field[y + BOUNDARY][x + BOUNDARY + 1] == 0 for x, y in pos])
            except IndexError:
                print('Will try again')
                pass

        elif move == 'left':
            pos = self.piece.current_position()

            try:
                return all([self.field[y + BOUNDARY][x + BOUNDARY - 1] == 0 for x, y in pos])
            except IndexError:
                print('Will try again')
                pass

        return False

    def basic_rotation(self, clockwise):
        mat = None
        piece = [self.piece.state[i:i + 4] for i in range(0, len(self.piece.state), 4)]
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
            self.clockwise = True
            mat = CLOCKWISE_MATRIX
        if not clockwise:
            self.clockwise = False
            mat = ANTICLOCKWISE_MATRIX

        # calculate new relative cord to centre
        for cord in relative_cords:
            new_r_cord = np.dot(cord, mat)
            new_relative_cords.append(new_r_cord)

        # get new global cords
        for x, y in new_relative_cords:
            new_global_cords.append((x + c_x, y + c_y))

        return new_global_cords


class Piece:
    def __init__(self, x=None, y=None, str_piece=None):
        self.x = x
        self.y = y
        self.str_id = str_piece
        self.piece = pieces[self.str_id]
        self.rot_index = 0
        self.state = self.piece
        self.clockwise = None
        self.all = [(j, i) for i in range(4) for j in range(4)]
        self.centre = centres[self.str_id]
        self.colour = colours[self.str_id] if self.piece is not None else None
        self.state_cords =  data.Data(self.str_id, 0).get_data()
        self.srs = SRS(self)

    def make_move(self, move):
        if move == 'right':
            if self.srs.make_move('right'):
                self.x += 1
        elif move == 'left':
            if self.srs.make_move('left'):
                self.x -= 1
        elif move == 'down':
            if self.srs.make_move('down'):
                self.y += 1
            else:
                return False
        elif move == 'cw':
            if self.str_id == 'O':
                pass
            else:
                if self.str_id == 'I':
                    self.srs.make_move('ccw')
                else:
                    self.srs.make_move('cw')

        elif move == 'ccw':
            if self.str_id == 'O':
                pass
            else:
                self.srs.make_move('ccw')

    def current_position(self):  # get grid positions of a passed piece object
        return [(r_x+self.x, r_y+self.y) for r_x, r_y in self.state_cords]

    def get_config(self):
        return self.rot_index, (self.x, self.y), self.current_position(), 0


class Board:
    def __init__(self, landed, lines, score):
        self.landed = landed
        self.score = score
        self.lines = lines
        self.level = 0
        self.paused_effect = [(2,18),(2,19),(2,20),(3,18),(3,20),(4,18),(4,19),(4,20),(5,20),(6,20),(5,17),(6,17),(3,17),(4,17),(4,15),
                             (4,16),(2,16),(3,15),(5,15),(6,15),(2,14),(2,12),(3,12),(3,14),(4,12),(4,14),(5,12),(5,14),(6,13),(3,11),
                             (2,10),(2,9),(4,9),(4,10),(5,8),(6,9),(6,10),(6,11),(2,8),(2,7),(2,6),(4,8),(4,7),(4,6),(6,8),
                             (6,7),(6,6),(3,8),(2,5),(3,5),(4,5),(5,5),(6,5),(6,4),(2,4),(3,3),(4,3),(5,3)]
    def create_grid(self):
        # if you want to change colour of grid, change _____ to desired colour! (except tetromino colour)
        GRID = [[WHITE for column in range(COLUMNS)] for row in range(ROWS)]

        for i in range(ROWS):
            for j in range(COLUMNS):
                if (j, i) in self.landed:
                    # set colour if position is landed i.e there is a piece there
                    GRID[i][j] = self.landed[(j, i)]

        return GRID

    def paused_grid(self):
        GRID = [[BLACK for column in range(COLUMNS)] for row in range(ROWS)]
        random_colour = colours[random.choice(['J','L','S','T','Z','O','I'])]
        for i in range(ROWS):
            for j in range(COLUMNS):
                if (j, i) in self.paused_effect:
                    GRID[i][j] = random_colour

        return GRID

    @staticmethod
    def show_best_move(grid, best_move):
        if best_move:
            for x, y in best_move[2]:
                try:
                    grid[y][x] = GREY
                except IndexError:
                    print('Problem')

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
                    pygame.draw.rect(surface, BLACK, (
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
                    pygame.draw.rect(surface, BLACK, (
                        pos_x + ind_y * block_size + 20, pos_y + ind_x * block_size + 20, block_size, block_size), 2)

    @staticmethod
    def render_grid(surface, grid):
        # boundary
        for i in range(ROWS + BOUNDARY + 1):
            for j in range(COLUMNS + BOUNDARY + 1):
                pygame.draw.rect(surface, GREY, (top_left_x - (BOUNDARY * block_size) + j * block_size,
                                                            top_left_y - (BOUNDARY * block_size) + i * block_size,
                                                            block_size, block_size), 0)
                pygame.draw.rect(surface, BLACK, (top_left_x - (BOUNDARY * block_size) + j * block_size,
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
                    pygame.draw.rect(surface, SPAWN_ROWS,
                                     (top_left_x + j * block_size, top_left_y + i * block_size, block_size, block_size),
                                     0)
                    pygame.draw.rect(surface, WHITE,
                                     (top_left_x + j * block_size, top_left_y + i * block_size, block_size, block_size),
                                     2)
                if grid[i][j] != WHITE:
                    pygame.draw.rect(surface, BLACK,
                                     (top_left_x + j * block_size, top_left_y + i * block_size, block_size, block_size),
                                     2)

    def clear_rows(self, grid):
        cleared_row = 0
        cleared_rows = 0

        for index in range(ROWS - 1, -1, -1):
            if WHITE not in grid[index]:
                cleared_rows += 1
                cleared_row = index
                for column in range(COLUMNS):
                    try:
                        del self.landed[(column, cleared_row)]  # deletes colours off cleared rows
                    except KeyError:
                        pass

        # sort all landed positions based on the rows in the grid, then reverse that as we are
        # searching grid from below

        for position in sorted(list(self.landed), key=lambda pos: pos[1], reverse=True):
            col, row = position

            if row < cleared_row:  # if row is above index of row that was cleared:
                new_pos = (col, row + cleared_rows)  # make new position that moves itm down by number of cleared rows

                self.landed[new_pos] = self.landed.pop(
                    position)  # .pop here removes colours from all rows above, and places them in their new positions

        self.lines += cleared_rows
        self.score += self.score_game(cleared_rows)

        return cleared_rows

    def score_game(self, cleared):
        if cleared == 1:
            return 40 + (40 * self.level)
        elif cleared == 2:
            return 100 + (100 * self.level)
        elif cleared == 3:
            return 300 + (300 * self.level)
        elif cleared == 4:
            return 1200 + (1200 * self.level)
        else:
            return 0

class Piece_Gne:
    def __init__(self, bag):
        self.bag = bag
        self.start_ind = 0

    def generator_function(self):
        permu = list(permutations(self.bag))

        while True:
            for piece in random.choice(permu):
                yield piece

    def pop(self, buffer, gen):
        popped = buffer[self.start_ind]
        self.start_ind = (self.start_ind + 1) % len(buffer)
        buffer[self.start_ind] = next(gen)

        return popped

    def get_piece(self, landed):
        gen = self.generator_function()
        buffer = [next(gen) for _ in range(7)]

        popped = self.pop(buffer, gen)

        p = Piece(4, -1, popped)
        p.srs.update_field(landed)
        return p

class Tetris:
    def __init__(self):
        self.win = pygame.display.set_mode((width, height))
        # piece generation setup
        self.generate = Piece_Gne(['I', 'S', 'O', 'Z', 'T', 'L', 'J'])

        # game board setup
        self.landed = {}
        self.lines = 0
        self.score = 0
        self.board = Board(self.landed, self.lines, self.score)
        self.tetrises = 0
        self.grid = self.board.create_grid()

        # get starting piece object
        self.current_piece = self.generate.get_piece(self.landed)

        # control parameters
        self.run = True
        self.show_piece = True
        self.held_piece = None
        self.change_piece = False
        self.training = True

        # get next piece
        self.next_piece = self.generate.get_piece(self.landed)

        # gravity setup
        self.fall_time = 0
        self.fall_speed = 0.08

        # game clock
        self.clock = pygame.time.Clock()

        self.best_move = None #(2, (9, 21), [(6, 19), (7, 19), (7, 20), (7, 21)])

    def draw_window(self):  # pass instance of board
        pygame.font.init()

        font = pygame.font.Font(f, 15)

        pos_x = top_left_x + play_w
        pos_y = top_left_y + play_h // 2

        score = font.render(f'Score: {self.board.score}', True, BLACK)
        lines = font.render(f'Lines: {self.board.lines}', True, BLACK)
        level = font.render(f'Level: {self.board.level}', True, BLACK)
        next_text = font.render('NEXT PIECE', True, BLACK)
        hold_text = font.render('HOLD PIECE', True, BLACK)

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
            if pos[1] <= 2:
                return True

        return False

    def change_state(self):
        # lock position
        """for i in self.current_piece.current_position():
            self.landed[i] = self.current_piece.colour"""

        # clear rows
        cleared = self.board.clear_rows(self.grid)

        if cleared == 4:
            self.tetrises += 1

        # update game level
        self.board.level = self.lines // 10

        self.board.show_next_piece(self.win, self.next_piece)

        self.lines, self.score = self.board.lines, self.board.score

        self.next_piece.srs.update_field(self.landed)
        self.current_piece = self.next_piece

        #if [WHITE] * 10 == self.grid[1] and [WHITE] * 10 == self.grid[0] and [WHITE] * 10 == self.grid[2]:
        self.next_piece = self.generate.get_piece(self.landed)
        self.change_piece = False

    def game_logic(self):
        self.grid = self.board.create_grid()

        """self.fall_time += self.clock.get_rawtime()
        self.clock.tick()

        if self.fall_time / 1000 > self.fall_speed:
            self.fall_time = 0

            if self.current_piece.make_move('down') is False:
                self.board.score += 1
                self.change_piece = True
            else:
                pass
        """
        self.win.fill(BG)

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
                        self.run = False

        self.draw_window()

        if self.lost():
            self.run = False

    def fitness_func(self):
        return self.tetrises*50 + self.score*5

    def make_ai_move(self):
        target_config = self.best_move  # exp: (0, (7, 21), [(6, 19), (7, 19), (7, 20), (7, 21)])
        
        for i in target_config[2]:
            self.landed[i] = self.current_piece.colour
        
        self.change_piece = True


# Only for testing, game loop for this tetris version is run in train_agent_scratch.py

"""tetris_game = Tetris()

while tetris_game.run:
    tetris_game.game_logic()

    tetris_game.make_ai_move()

    if tetris_game.change_piece:
        tetris_game.change_state()"""








