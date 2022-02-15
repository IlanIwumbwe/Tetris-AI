import tetris_ai
import heuristics as hu
import data
import pickle
from nueralnet import Population
import matplotlib.pyplot as plt
from matplotlib import style

class AI_Agent:
    def __init__(self):  # piece is an object
        self.rows = tetris_ai.ROWS
        self.columns = tetris_ai.COLUMNS
        self.field = [[0 for _ in range(self.columns)] for _ in range(self.rows)]
        self.landed = None
        self.final_cords = []
        self.heuris = hu.Heuristics(self.columns, self.rows)  # this is a heuristics obj
        self.best_move = None
        self.move_data = {}
        self.final_positions = {}
        self.all_pieces = ['I', 'S', 'O', 'Z', 'T', 'L', 'J']
        self.all_configurations = []
        self.cord_scores = {}
        self.actions_scores = []
        self.nueral_net = None

    def get_possible_configurations(self, current_piece):
        positions = [[(ind_y, ind_x) for ind_y in range(self.columns) if self.field[ind_x][ind_y] == 0] for ind_x in range(self.rows)]
        all_positions = [tupl for li in positions for tupl in li]

        data_obj = data.Data(current_piece.str_id, None)
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
                                all_configurations.append(((pos_x, pos_y), index, final_global_cords))

                            done = True
                        else:
                            pos_y += 1
            else:
                continue

        self.all_configurations = all_configurations

    def update_agent(self, current_piece):
        self.update_field()
        self.get_possible_configurations(current_piece)

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

    def evaluation_function(self, current_piece):
        if len(self.all_configurations) != 0:
            score_moves = []

            for cord, index, positions in self.all_configurations:
                for x, y in positions:
                    self.field[y][x] = 1

                self.heuris.update_field(self.field)

                board_state = self.heuris.get_heuristics()

                full_board_state = board_state

                move_score = self.nueral_net.query(full_board_state)[0]

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

class Trainer:
    def __init__(self):
        self.agent = AI_Agent()
        self.record = 0
        self.new_pop = None
        self.old_pop = None
        self.epochs = 25
        self.checkpoint = 2
        self.epoch_data = {}

    def eval(self, load_population):
        """ WHole population object is pickled"""
        if load_population:
            with open("population.pkl", "rb") as f:
                self.old_pop = pickle.load(f)

        for epoch in range(self.epochs):
            print('__________________________________________')
            scores = []
            self.new_pop = Population(1000, self.old_pop)
            print(f'EPOCH: {epoch+1}')
            print(f'HIGHSCORE: {self.record}')
            for neural_index in range(self.new_pop.size):
                current_fitness = 0
                tetris_game = tetris_ai.Tetris()

                self.agent = AI_Agent()
                self.agent.nueral_net = self.new_pop.models[neural_index]

                while tetris_game.run:
                    self.agent.landed = tetris_game.landed

                    # update the agent with useful info to find the best move
                    self.agent.update_agent(tetris_game.current_piece)
                    tetris_game.best_move = self.agent.get_best_move(tetris_game.current_piece)
                    tetris_game.game_logic()

                    # make the move
                    tetris_game.make_ai_move()

                    current_fitness += tetris_game.fitness_func()

                    self.agent.landed = tetris_game.landed

                    # update the agent with useful info to find the best move
                    self.agent.update_agent(tetris_game.current_piece)

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

            if (epoch+1) % self.checkpoint == 0:
                print('Saving models///////......')
                self.old_pop.save_population(epoch)
                print('Saved successfully////////////////')

            print(f'Best fitness: {max(self.epoch_data[epoch+1][1])}')
            print(f'Average fitness: {self.epoch_data[epoch+1][0]}')

        # plot graphs after epochs are done
        style.use("ggplot")

        # fitness graph
        for epoch, d in self.epoch_data.items():
            for fitness in d[1]:
                plt.scatter(epoch, fitness)

            plt.plot(epoch, d[0], color="red")

        plt.title("Fitness against epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Fitness")

        plt.show()

        # Scores graph
        for epoch, d in self.epoch_data.items():
            for score in d[3]:
                plt.scatter(epoch, score)

            plt.plot(epoch, d[2], color="red")

        plt.title("Score against epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Score")

        plt.show()

if __name__ == '__main__':
    trainer = Trainer()

    load = input('LOAD POPULATION: ')

    if load == 'Y':
        trainer.eval(True)
    else:
        trainer.eval(False)



