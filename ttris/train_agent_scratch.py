import tetris_ai
import pickle
from nueralnet import Population
from visualize import Visualize
from heuristics import Heuristics


class AI_Agent:
    def __init__(self):  # piece is an object
        self.rows = tetris_ai.ROWS
        self.columns = tetris_ai.COLUMNS
        self.field = [[0 for _ in range(self.columns)] for _ in range(self.rows)]
        self.landed = None
        self.heuris = Heuristics(self.columns, self.rows)  # this is a heuristics obj
        self.best_move = None
        self.all_pieces = ['I', 'S', 'O', 'Z', 'T', 'L', 'J']
        self.all_configurations = []
        self.next_configurations = []
        self.next_state = None
        self.next_piece = None
        self.nueral_net = None
        self.ablation = None

    def get_possible_configurations(self, piece, field, is_next_piece):
        positions = [[(ind_y, ind_x) for ind_y in range(self.columns) if field[ind_x][ind_y] == 0] for ind_x in range(self.rows)]
        all_positions = [tupl for li in positions for tupl in li]

        data_obj = Data(piece.str_id, None)
        all_configurations = []

        for pos_x in range(-3, self.columns+3):
            possible = 0

            for ind in range(4):
                data_obj.rot_index = ind
                ascii_cords = data_obj.get_data()

                abb_y = 2
                if not all([(pos_x + x, abb_y + y) in all_positions for x, y in ascii_cords]):
                    possible += 0
                else:
                    possible += 1

            if possible != 0:
                for index in range(4):
                    data_obj.rot_index = index
                    ascii_cords = data_obj.get_data()
                    pos_y = 0
                    done = False

                    while not done:
                        if not all([(pos_x + x, pos_y + 1 + y) in all_positions for x, y in ascii_cords]):
                            final_global_cords = [(x + pos_x, y + pos_y) for x, y in ascii_cords]
                            # check validity of rotation states
                            if all([cord in all_positions for cord in final_global_cords]):
                                all_configurations.append(((pos_x, pos_y), index, final_global_cords))

                            done = True

                        elif not all([(pos_x + x, pos_y + y) in all_positions for x, y in ascii_cords]):
                            final_global_cords = [(x + pos_x, y + pos_y - 1) for x, y in ascii_cords]
                            # check validity of rotation states
                            if all([cord in all_positions for cord in final_global_cords]):
                                all_configurations.append(((pos_x, pos_y - 1), index, final_global_cords))

                            done = True

                        else:
                            pos_y += 1
            else:
                continue

        if not is_next_piece:
            self.all_configurations = all_configurations
        else:
            self.next_configurations = all_configurations

    def update_agent(self, current_piece, next_piece, landed):
        self.landed = landed
        self.update_field()
        self.get_possible_configurations(current_piece, self.field, False)
        self.next_piece = next_piece

    def update_field(self):
        self.field = [[0 for _ in range(self.columns)] for _ in range(self.rows)]
        if self.landed != {}:
            for pos, val in self.landed.items():
                if val != (255, 255, 255):
                    x, y = pos
                    try:
                        self.field[y][x] = 1
                    except IndexError:
                        print('random index error is random')
                        pass

    def next_piece_knowledge(self):
        score_next_moves = []

        self.get_possible_configurations(self.next_piece, self.next_state, True)

        if len(self.next_configurations) != 0:
            for next_cord, next_ind, next_positions in self.next_configurations:
                for x, y in next_positions:
                    self.next_state[y][x] = 1

                self.heuris.update_field(self.next_state)
                board_state = self.heuris.get_heuristics()
                board_state_2 = [board_state[k] for k in range(len(board_state)) if k not in self.ablation]
                next_move_score = self.nueral_net.query(board_state_2)[0]

                for x, y in next_positions:
                    self.next_state[y][x] = 0

                score_next_moves.append(next_move_score)

            return max(score_next_moves)
        else:
            return 0

    def evaluation_function(self, current_piece):
        if len(self.all_configurations) != 0:
            score_moves = []

            for cord, index, positions in self.all_configurations:
                move_score = 0
                for x, y in positions:
                    self.field[y][x] = 1

                self.next_state = self.field

                # move_score += self.next_piece_knowledge()

                self.heuris.update_field(self.field)

                board_state = self.heuris.get_heuristics()

                board_state_2 = [board_state[k] for k in range(len(board_state)) if k not in self.ablation]

                move_score += self.nueral_net.query(board_state_2)[0]

                for x, y in positions:
                    self.field[y][x] = 0

                score_moves.append((index, cord, positions, move_score))

            best_move = max(score_moves,  key= lambda x: x[-1])

            self.best_move = best_move[:-1]

        else:
            self.best_move = current_piece.get_config()

    def get_best_move(self, current_piece):
        self.evaluation_function(current_piece)

        return self.best_move

    def print_field(self):
        for i in self.field:
            print(i)
            
class Data:
    def __init__(self, piece_id, test_rot_index):
        self.str_id = piece_id  # its a string
        self.rot_index = test_rot_index

    def O(self):
        O = {0: [(1, 1), (2, 1), (1, 2), (2, 2)],
             1: [(1, 1), (2, 1), (1, 2), (2, 2)],
             2: [(1, 1), (2, 1), (1, 2), (2, 2)],
             3: [(1, 1), (2, 1), (1, 2), (2, 2)]
             }
        return O[self.rot_index]

    def T(self):
        T = {0: [(0, 2), (1, 2), (1, 1), (2, 2)],
             1: [(1, 1), (1, 2), (1, 3), (2, 2)],
             2: [(0, 2), (1, 2), (2, 2), (1, 3)],
             3: [(1, 1), (1, 2), (1, 3), (0, 2)]
             }
        return T[self.rot_index]

    def I(self):
        I = {0: [(0, 1), (1, 1), (2, 1), (3, 1)],
             1: [(1, 0), (1, 1), (1, 2), (1, 3)],
             2: [(0, 1), (1, 1), (2, 1), (3, 1)],
             3: [(1, 0), (1, 1), (1, 2), (1, 3)]
             }
        return I[self.rot_index]

    def L(self):
        L = {0: [(1, 2), (2, 2), (3, 2), (3, 1)],
             1: [(2, 1), (2, 2), (2, 3), (3, 3)],
             2: [(1, 2), (2, 2), (1, 3), (3, 2)],
             3: [(1, 1), (2, 2), (2, 1), (2, 3)]
             }
        return L[self.rot_index]

    def Z(self):
        Z = {0: [(0, 1), (1, 2), (1, 1), (2, 2)],
             1: [(2, 1), (1, 2), (2, 2), (1, 3)],
             2: [(0, 2), (1, 2), (1, 3), (2, 3)],
             3: [(0, 2), (1, 2), (1, 1), (0, 3)]
             }
        return Z[self.rot_index]

    def J(self):
        J = {0: [(0, 1), (1, 2), (0, 2), (2, 2)],
             1: [(1, 1), (1, 2), (1, 3), (2, 1)],
             2: [(0, 2), (1, 2), (2, 2), (2, 3)],
             3: [(1, 1), (1, 2), (1, 3), (0, 3)]
             }
        return J[self.rot_index]

    def S(self):
        S = {0: [(0, 2), (1, 2), (1, 1), (2, 1)],
             1: [(1, 1), (1, 2), (2, 2), (2, 3)],
             2: [(1, 1), (1, 2), (2, 2), (2, 3)],
             3: [(1, 3), (1, 2), (0, 3), (2, 2)]
             }
        return S[self.rot_index]

    def get_data(self):
        if self.str_id == 'O':
            return self.O()
        elif self.str_id == 'I':
            return self.I()
        elif self.str_id == 'Z':
            return self.Z()
        elif self.str_id == 'S':
            return self.S()
        elif self.str_id == 'T':
            return self.T()
        elif self.str_id == 'L':
            return self.L()
        elif self.str_id == 'J':
            return self.J()
        
class Trainer:
    def __init__(self):
        self.agent = AI_Agent()
        self.record = 0
        self.new_pop = None
        self.old_pop = None
        self.epochs = 10
        self.checkpoint = 2
        self.epoch_data = {}

    def eval(self, load_population, epoch_number, ablation):

        if ablation is None:
            li = []
        else:
            li = [int(j) for j in ablation.split(',')]

        if load_population:
            try:
                path1 = f"./populations/{epoch_number}population.pkl"
                with open(path1, "rb") as f:
                    weight_matrices = pickle.load(f)

                path2 = f"./populations/{epoch_number}fitness.pkl"
                with open(path2, "rb") as f:
                    fitnesses = pickle.load(f)

                self.old_pop = Population(1000, None, 9)

                for ind, model in enumerate(self.old_pop.models):
                    model.wi_ha = weight_matrices[ind][0]
                    model.wha_hb = weight_matrices[ind][1]
                    model.whb_o = weight_matrices[ind][2]

                self.old_pop.fitnesses = fitnesses

            except FileNotFoundError or FileExistsError:
                print('File not found, or it does not exist')

        for epoch in range(self.epochs):
            print('__________________________________________')
            scores = []
            self.new_pop = Population(1000, self.old_pop, 9-len(li))
            print(f'EPOCH: {epoch+1}')
            print(f'HIGHSCORE: {self.record}')
            for neural_index in range(self.new_pop.size):
                current_fitness = 0

                tetris_game = tetris_ai.Tetris()

                self.agent = AI_Agent()
                self.agent.ablation = li
                self.agent.nueral_net = self.new_pop.models[neural_index]

                while tetris_game.run:
                    tetris_game.game_logic()
                    # update the agent with useful info to find the best move
                    self.agent.update_agent(tetris_game.current_piece, tetris_game.next_piece, tetris_game.landed)

                    tetris_game.best_move = self.agent.get_best_move(tetris_game.current_piece)

                    # make the move
                    tetris_game.make_ai_move()

                    current_fitness += tetris_game.fitness_func()

                    if tetris_game.change_piece:
                        tetris_game.change_state()

                    if not tetris_game.run:
                        self.new_pop.fitnesses[neural_index] = current_fitness
                        break

                scores.append(tetris_game.score)
                if tetris_game.score > self.record:
                    self.record = tetris_game.score
                    print(f'HIGHSCORE: {self.record}')

            self.epoch_data[epoch+1] = (sum(self.new_pop.fitnesses)/1000, self.new_pop.fitnesses, sum(scores)/1000, scores)
            self.old_pop = self.new_pop

            """if (epoch+1) % self.checkpoint == 0:
                print('Saving models///////......')
                self.old_pop.save_population(epoch)
                print('Saved successfully////////////////')"""

            print(f'Best fitness: {max(self.epoch_data[epoch+1][1])}')
            print(f'Average fitness: {self.epoch_data[epoch+1][0]}')


if __name__ == '__main__':
    trainer = Trainer()

    print('_______ Ablation _______')
    print("0 -> Deepest well\n1 -> Total height\n2 -> Total holes\n3-> Bumpiness\n4 -> Lines Cleared\n5 -> Row transitions\n6 -> Standard deviation of heights\n7 -> Number of pits\n8 -> Column transitions")
    print('Input should be between 0 and 8, follow list above ¯\_(ツ)_/¯')
    i = input('Type one number to index to the heuristic you want to remove or type a series of numbers separated by commas, Enter not to ablate: ')

    if i:
        trainer.eval(False, 0, i)

        vis = Visualize(list(trainer.epoch_data.keys()) + list(trainer.epoch_data.values()))
        vis.visualize()

    else:
        print('__________________________________')
        load = input('LOAD POPULATION(L): Enter not to load ')
        if load == 'L':
            trainer.eval(True, int(input('From which epoch(2,4,6,8,10): ')), None)

            vis = Visualize(list(trainer.epoch_data.keys()) + list(trainer.epoch_data.values()))
            vis.visualize()
        else:
            trainer.eval(False, 0, None)

            vis = Visualize(list(trainer.epoch_data.keys()) + list(trainer.epoch_data.values()))
            vis.visualize()





