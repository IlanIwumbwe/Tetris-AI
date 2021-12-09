import random
import tetris_ai
import heuristics as hu
import numpy as np
import data
import neat
import pygame
import pickle
import math

class AI:
    def __init__(self, rows, columns, w_1, w_2, w_3, w_4, w_5, landed= None, neural_network = None):  # piece is an object
        self.rows = rows
        self.columns = columns
        self.field = [[0 for _ in range(self.columns)] for _ in range(self.rows)]
        self.weights = np.array([w_1, w_2, w_3, w_4, w_5])
        self.landed = landed
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
        self.neural_network = neural_network

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
                        relative_cords.append((x - lo_x, y - lo_y))

                    final_global_cords = [(x + pos_x, y + pos_y) for x, y in relative_cords]

                    # check validity of rotation states
                    if all([cord in all_positions for cord in final_global_cords]):
                        if all([(x, y-1) in all_positions for x, y in final_global_cords]):
                            all_configurations[(pos_x, pos_y)].append((index, final_global_cords))

                # remove position from all configurations list if it yields an empty list i.e no configurations are
                # possible
                if len(all_configurations[(pos_x, pos_y)]) == 0:
                    del all_configurations[(pos_x, pos_y)]

            self.all_configurations_per_piece[piece_id] = all_configurations

    def get_graph(self, graph):
        self.graph_per_piece = graph

    def final_y(self, column):
        for row in range(self.rows):
            if self.field[row][column] == 1:
                return row - 1

        return self.rows - 1

    def set_all_configurations(self, current_piece):
        self.all_configurations = self.all_configurations_per_piece[current_piece.str_id]

    def set_graph_for_current_piece(self, current_piece):
        self.graph_for_current_piece = self.graph_per_piece[current_piece.str_id]

    def update_ai(self, current_piece):
        self.set_all_configurations(current_piece)
        self.update_all_configurations()

        self.update_field()
        final_cords = [(col, self.final_y(col)) for col in range(self.columns)]

        self.set_final_cords(final_cords)
        self.set_final_positions()

        self.evaluation_function()
        self.set_best_move(current_piece)

        """
        config = self.best_move
        new = config[0], config[1], str(config[2]), False, math.inf, math.inf, None

        self.set_graph_for_current_piece(current_piece)
        self.update_graph()

        # get start node for graph
        start_node = None
        for node in self.graph_for_current_piece:
            if new[1] in node:
                start_node = node

        # get target node
        target_node = self.best_move[0], self.best_move[1], str(self.best_move[2]), False, math.inf, math.inf, None

        print(f'Start from {start_node} and end at {target_node}')

        graph = solve_AStar(start_node, target_node, self.graph_for_current_piece)
        """

    def update_graph(self):
        nodes_to_delete = []
        for node, neighbours in self.graph_for_current_piece.items():
            node_cords = node[1]

            if node_cords in self.landed.keys():
                nodes_to_delete.append(node)

                for neighbour in neighbours:
                    if neighbour in self.graph_for_current_piece:
                        nn_list = self.graph_for_current_piece.get(neighbour)

                        for index, nn in enumerate(nn_list):
                            if nn == node:
                                del nn_list[index]

        for node in nodes_to_delete:
            del self.graph_for_current_piece[node]

    def update_field(self):
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

    def evaluation_function(self):
        self.cord_scores = {}
        """
        go through final positions, place them in num grid,
        calculate score
        """
        for cord, positions in self.final_positions.items():
            states_scores = []
            self.cord_scores[cord] = []
            for index, pos in positions:  # first part is the rotation index, second part are the positions
                for x, y in pos:
                    self.field[y][x] = 1

                self.heuris.update_field(self.field)
                heuristics = self.heuris.get_heuristics()

                # reset field, and update heuristics field
                for x, y in pos:
                    self.field[y][x] = 0

                self.heuris.update_field(self.field)

                score = self.neural_network.activate(heuristics)
                states_scores.append((index, pos, score))  # add that state with its score to the state scores list

            self.cord_scores[cord] = states_scores

    def set_best_move(self, current_piece):
        best_config_per_cord = {}

        # for each coordinate, find its best configuration
        for cord, configurations in self.cord_scores.items():
            best_config = sorted(configurations, key=lambda data: data[2])[-1]  # the last config has highest score
            best_config_per_cord[cord] = best_config

        # find the coordinate with the best configuration, and this is your best move
        try:
            best_move = [(key, value[:2]) for key, value in sorted(best_config_per_cord.items(), key=lambda pair:pair[1][2])][-1]
            rotation_state = best_move[1][0]
            cord = best_move[0]
            positions = best_move[1][1]

            self.best_move = rotation_state, cord, positions  # exp: (3, (5, 8), [(2, 8), (3, 8), (4, 8), (5, 8)])
        except IndexError:
            self.best_move = current_piece.get_config()

    def get_best_move(self):
        return self.best_move

    def print_field(self):
        print('NEW STATE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        for i in self.field:
            print(f'{i} \n')


def init_ai_object(rows, columns):
    a, b, c, d, e = -0.1, -0.5, -0.2, -0.3, -0.2

    ai_obj = AI(rows, columns, a, b, c, d, e)

    ai_obj.get_possible_configurations()

    """graph_per_piece = create_initial_connections(ai_obj.all_configurations_per_piece)
    ai_obj.get_graph(graph_per_piece)"""

    return ai_obj

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        'config-feedfoward.txt')

population = neat.Population(config)

population.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
population.add_reporter(stats)
population.add_reporter(neat.Checkpointer(10))

def eval_genomes(genomes, config):
    ai = init_ai_object(tetris_ai.ROWS, tetris_ai.COLUMNS)
    hueris = hu.Heuristics(tetris_ai.COLUMNS, tetris_ai.ROWS)
    pygame.init()

    for genome_id, genome in genomes:
        tetris_game = tetris_ai.Tetris()

        # set up a neural network for this genome
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        ai.neural_network = net

        # set up heuristics landed and ai landed dictionaries
        ai.landed = tetris_game.landed
        hueris.landed = tetris_game.landed
        current_fitness = 0

        # update the ai with useful info to find the best move
        ai.update_ai(tetris_game.current_piece)
        tetris_game.best_move = ai.get_best_move()

        # make the move
        tetris_game.make_ai_move()

        # update heuristics field
        hueris.update_field(ai.field)

        # calculate genome fitness
        current_fitness += np.dot(hueris.get_heuristics(), ai.weights)

        # do this throughout the whole game
        while tetris_game.run:
            tetris_game.game_logic()

            if tetris_game.change_piece:
                tetris_game.change_state()

                ai.update_ai(tetris_game.current_piece)

                tetris_game.best_move = ai.get_best_move()

                tetris_game.make_ai_move()

                hueris.update_field(ai.field)
                current_fitness += np.dot(hueris.get_heuristics(), ai.weights)

            if not tetris_game.run:
                current_fitness += (tetris_game.tetrises*4 + tetris_game.score*2 + tetris_game.lines*2)
                genome.fitness = current_fitness

best_genome = population.run(eval_genomes)

with open('best_genome.pkl', 'wb') as output:
    pickle.dump(best_genome, output, 1)



"""# A* algorithm
def create_initial_connections(all_configurations_per_piece):  # populate neighbours lists for all nodes
    graph_per_piece = {}
    visited = False
    global_goal = math.inf
    local_goal = math.inf
    parent = None

    for piece_id, all_configurations in all_configurations_per_piece.items():
        graph = {}
        for cord, rotation_states in all_configurations.items():
            c_x, c_y = cord
            neighbour_cords = {}

            if (c_x+1, c_y) in all_configurations:
                neighbour_cords[(c_x+1, c_y)] = all_configurations.get((c_x+1, c_y))
            elif (c_x-1, c_y) in all_configurations:
                neighbour_cords[(c_x-1, c_y)] = all_configurations.get((c_x-1, c_y))
            elif (c_x, c_y+1) in all_configurations:
                neighbour_cords[(c_x, c_y+1)] = all_configurations.get((c_x, c_y+1))

            for rotation_state in rotation_states:
                configuration = rotation_state[0], cord, str(rotation_state[1]), visited, global_goal, local_goal, parent

                graph[configuration] = []

                # find neighbours that share same cords

                for n_state in rotation_states:
                    if n_state != rotation_state:
                        n_config = n_state[0], cord, str(n_state[1]), visited, global_goal, local_goal, parent

                        graph[configuration].append(n_config)

                # find neighbours from neighbouring cords
                for n_cord, states in neighbour_cords.items():
                    for state in states:
                        nn_config = state[0], n_cord, str(state[1]), visited, global_goal, local_goal, parent

                        graph[configuration].append(nn_config)

        graph_per_piece[piece_id] = graph

    return graph_per_piece


def modify_node(node, *args):
    mappings = [args[i:i+2] for i in range(0, len(args), 2)]
    # state, cord, config, visited, global_goal, local_goal, parent
    mapping = {'state': node[0], 'cord': node[1], 'config': str(node[2]), 'visited': node[3], 'global_goal': node[4],
               'local_goal': node[5], 'parent': node[6]}

    for thing_to_modify, new_value in mappings:
        if thing_to_modify == 'config':
            mapping[thing_to_modify] = str(new_value)
        else:
            mapping[thing_to_modify] = new_value

    new_node = []

    for part in mapping.values():
        new_node.append(part)

    return tuple(new_node)


def distance(node_1, node_2):
    n1_config_state, n1_config_cords = node_1[0], node_2[1]
    n2_config_state, n2_config_cords = node_2[0], node_2[1]

    if n1_config_cords == n2_config_cords:
        difference = abs(n1_config_state - n2_config_state)
        if difference == 2:
            return 2
        else:
            return difference % 2
    else:
        x_1, y_1 = n1_config_cords
        x_2, y_2 = n2_config_cords

        return math.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)


def search_heuristic(node_1, node_2):  # the search heuristic is the euclidean distance between 2 nodes
    return distance(node_1, node_2)


def solve_AStar(start_node, target_node, graph):
    # state, cord, config, visited, global_goal, local_goal, parent

    new_start = modify_node(start_node, 'local_goal', 0, 'global_goal', search_heuristic(start_node, target_node))

    print(new_start)
    graph[new_start] = graph.pop(start_node)

    not_tested = [new_start]

    while len(not_tested) != 0:
        not_tested = sorted(not_tested, key=lambda node: node[4])

        while not_tested[0][3] and len(not_tested) != 0:
            not_tested.pop(0)

        if len(not_tested) != 0:
            break

        new_test_node = modify_node(not_tested[0], 'visited', True)
        not_tested[0] = new_test_node

        current_node = not_tested[0]
        print(current_node)

        for neighbour in graph[current_node]:
            if not neighbour[3]:
                not_tested.append(neighbour)

            new_local_goal = current_node[5] + distance(current_node, neighbour)

            if new_local_goal < neighbour[5]:
                new_neighbour = modify_node(neighbour, 'parent', current_node, 'local_goal', new_local_goal, 'global_goal', search_heuristic(neighbour, target_node))

                graph[new_neighbour] = graph.pop(neighbour)

    return graph

"""










