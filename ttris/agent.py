import tetris_ai
import heuristics as hu
import data
import torch
import random
import pygame
from collections import deque
from model import LinearQNet, Trainer
import math

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.003

class AI_Agent:
    def __init__(self, rows, columns):  # piece is an object
        self.rows = rows
        self.columns = columns
        self.field = [[0 for _ in range(self.columns)] for _ in range(self.rows)]
        self.landed = None
        self.final_cords = []
        self.heuris = hu.Heuristics(self.columns, self.rows)  # this is a heuristics obj
        self.best_move = None
        self.move_data = {}
        self.final_positions = {}
        self.all_pieces = ['I', 'S', 'O', 'Z', 'T', 'L', 'J']
        self.all_configurations_per_piece = {}
        self.all_configurations = {}
        self.graph_per_piece = {}
        self.graph_for_current_piece = {}
        self.cord_scores = {}
        self.actions_scores = []
        self.n_games = 0 # number of games
        self.gamma = 0.8 # discount rate, between 0 and 1
        self.neural_network = LinearQNet(7, 500, 40)
        self.trainer = Trainer(self.neural_network, LR, self.gamma)
        self.memory = deque(maxlen=MAX_MEMORY)

    def get_possible_configurations(self):
        # need to change this a bit so it works for every single piece
        """
        These make up the nodes in the graph for A*
        """

        positions = [[(ind_y, ind_x) for ind_y in range(self.columns) if ind_x != 0 or ind_x != 1] for ind_x in range(self.rows)]
        all_positions = [tupl for li in positions for tupl in li]

        for piece_id in self.all_pieces:
            data_obj = data.Data(piece_id, None)
            all_configurations = {}

            for pos_x, pos_y in all_positions:
                all_configurations[(pos_x, pos_y)] = []
                for index in range(4):  # all rotation indices....
                    relative_cords = []
                    data_obj.rot_index = index
                    ascii_cords = data_obj.get_data()  # exp: [(0, 1), (0, 2), (1, 2), (1, 3)]

                    lowest_block = sorted(ascii_cords, key=lambda cord: cord[1])[-1]  # lowest hanging piece based on y

                    lo_x, lo_y = lowest_block
                    # get relative cords to lowest block
                    for x, y in ascii_cords:
                        relative_cords.append((lo_x - x, lo_y - y))

                    final_global_cords = [(x + pos_x, y + pos_y) for x, y in relative_cords]

                    # check validity of rotation states
                    if all([cord in all_positions for cord in final_global_cords]):
                        # if all([(x+pos_x, y+y_above) in all_positions for x, y in relative_cords for y_above in range(2, pos_y)]):
                        all_configurations[(pos_x, pos_y)].append((index, final_global_cords))

                # remove position from all configurations list if it yields an empty list i.e no configurations are
                # possible

                if len(all_configurations[(pos_x, pos_y)]) == 0:
                    del all_configurations[(pos_x, pos_y)]

            self.all_configurations_per_piece[piece_id] = all_configurations

    def final_y(self, column):
        for row in range(self.rows):
            if self.field[row][column] == 1:
                return row - 1

        return self.rows - 1

    def set_all_configurations(self, current_piece):
        self.all_configurations = self.all_configurations_per_piece[current_piece.str_id]

    def update_agent(self, current_piece):
        self.set_all_configurations(current_piece)
        self.update_all_configurations()
        self.update_field()

        self.heuris.update_field(self.field)

    def update_field(self):
        self.field = [[0 for _ in range(self.columns)] for _ in range(self.rows)]
        if self.landed != {}:
            for pos, val in self.landed.items():
                if val != (255, 255, 255):
                    x, y = pos
                    self.field[y][x] = 1

    def update_all_configurations(self):
        if self.landed != {}:
            for landed_cord in self.landed.keys():
                if landed_cord in self.all_configurations:
                    del self.all_configurations[landed_cord]

    def set_final_cords(self, final_cords):
        self.final_cords = final_cords

    def set_final_positions(self):
        self.final_positions = {}
        for cord in self.all_configurations:
            if cord in self.final_cords:
                self.final_positions[cord] = self.all_configurations.get(cord)
    
    def get_cord_and_rot_index(self, space_index):
        return space_index%4, math.floor(space_index/4)
        
    def evaluation_function(self, current_piece):
        """
        go through final positions, place them in num grid,
        calculate score
        """

        heuristics = self.heuris.get_heuristics()
        state0 = torch.tensor(heuristics, dtype=torch.float)
        scores = self.neural_network(state0)

        self.actions_scores = scores.tolist()
        dict_of_scores = {}

        for ind, val in enumerate(scores.tolist()):
            dict_of_scores[self.get_cord_and_rot_index(ind)] = val

        final_dict_of_scores = {}

        if len(self.final_positions) != 0:
            for x, y in self.final_positions.keys():
                for rot_index, x_cord in dict_of_scores.keys():
                    if x == x_cord:
                        final_dict_of_scores[(rot_index, x_cord)] = dict_of_scores.get((rot_index, x_cord))

            best_move = max(final_dict_of_scores, key=final_dict_of_scores.get)

            field = self.field.copy()
            for cord, positions in self.final_positions.items():
                x, y = cord

                for index, pos in positions:  # first part is the rotation index, second part are the positions
                    for x, y in pos:
                        field[y][x] = 1

                    self.heuris.update_field(field)

                    possible_reward = self.heuris.get_reward()

                    # reset field, and update heuristics field
                    for x, y in pos:
                        field[y][x] = 0

                    self.heuris.update_field(field)

                    if index == best_move[0] and x == best_move[1]:
                        self.best_move = index, cord, positions, possible_reward
        else:
            self.best_move = current_piece.get_config()

    def get_best_move(self, current_piece):
        final_cords = [(col, self.final_y(col)) for col in range(self.columns)]

        self.set_final_cords(final_cords)
        self.set_final_positions()

        self.evaluation_function(current_piece)

        return self.best_move

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples from remember
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def print_field(self):
        for i in self.field:
            print(i)

def init_agent(rows, columns):
    ai_obj = AI_Agent(rows, columns)

    ai_obj.get_possible_configurations()

    return ai_obj

def train():
    pygame.init()
    agent = init_agent(tetris_ai.ROWS, tetris_ai.COLUMNS)
    hueris = hu.Heuristics(tetris_ai.COLUMNS, tetris_ai.ROWS)

    record_score = 0

    tetris_game = tetris_ai.Tetris()

    while True:
        agent.landed = tetris_game.landed

        # update the agent with useful info to find the best move
        agent.update_agent(tetris_game.current_piece)

        tetris_game.best_move = agent.get_best_move(tetris_game.current_piece)

        tetris_game.game_logic()

        old_state = hueris.get_heuristics()

        # make the move
        tetris_game.make_ai_move()

        reward, current_score, done = tetris_game.reward_info()

        agent.landed = tetris_game.landed

        # update the agent with useful info to find the best move
        agent.update_agent(tetris_game.current_piece)

        new_state = hueris.get_heuristics()

        # train short memory
        agent.train_short_memory(old_state, agent.actions_scores, reward, new_state, done)

        # remember
        agent.remember(old_state, agent.actions_scores, reward, new_state, done)
        while tetris_game.run:
            agent.landed = tetris_game.landed

            # update the agent with useful info to find the best move
            agent.update_agent(tetris_game.current_piece)

            tetris_game.best_move = agent.get_best_move(tetris_game.current_piece)

            tetris_game.game_logic()

            old_state = hueris.get_heuristics()

            # make the move
            tetris_game.make_ai_move()

            reward, current_score, done = tetris_game.reward_info()

            agent.landed = tetris_game.landed

            # update the agent with useful info to find the best move
            agent.update_agent(tetris_game.current_piece)

            new_state = hueris.get_heuristics()

            # train short memory
            agent.train_short_memory(old_state, agent.actions_scores, reward, new_state, done)

            # remember
            agent.remember(old_state, agent.actions_scores, reward, new_state, done)

            if tetris_game.change_piece:
                tetris_game.change_state()

            if not tetris_game.run:
                agent.n_games += 1
                # train long memory
                agent.train_long_memory()

                if current_score > record_score:
                    record_score = current_score
                    # save this model, its probably good
                    agent.neural_network.save()

                tetris_game = tetris_ai.Tetris()

                print(f'GAME: {agent.n_games}\nScore: {current_score}\nRecord:{record_score}')


train()






