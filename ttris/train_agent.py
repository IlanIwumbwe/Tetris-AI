import tetris_ai
import heuristics as hu
import data
import neat
import pickle
from nueralnet import Population

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
        self.all_configurations_per_piece = {}
        self.all_configurations = []
        self.cord_scores = {}
        self.actions_scores = []
        self.vector = None

    def get_possible_configurations(self):
        positions = [[(ind_y, ind_x) for ind_y in range(self.columns) if self.field[ind_x][ind_y] == 0] for ind_x in range(self.rows)]
        all_positions = [tupl for li in positions for tupl in li]

        for piece_id in self.all_pieces:
            data_obj = data.Data(piece_id, None)
            all_configurations = []

            for pos_x in range(self.columns):
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
                            else:
                                pos_y += 1
                else:
                    continue

            self.all_configurations_per_piece[piece_id] = all_configurations

    def set_all_configurations(self, current_piece):
        self.all_configurations = self.all_configurations_per_piece[current_piece.str_id]

    def get_piece_mapping(self, current_piece):
        mapping = [0 for _ in range(7)]

        mapping[self.all_pieces.index(current_piece.str_id)] = 1

        return mapping

    def update_agent(self, current_piece):
        self.update_field()
        self.get_possible_configurations()
        self.set_all_configurations(current_piece)

        self.heuris.update_field(self.field)

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
                # fill in block positions in test field
                for x, y in positions:
                    self.field[y][x] = 1

                # pass that as the field to be used to get heuristics
                self.heuris.update_field(self.field)

                # access info from heuristics file
                possible_reward = self.heuris.get_reward()
                board_state = self.heuris.get_heuristics()

                # get piece mapping
                mapping = self.get_piece_mapping(current_piece)

                # prepare full board state
                full_board_state = board_state

                # get score from nueral net
                move_score = self.vector.activate(full_board_state)[0]

                # EMPTY THE POSITIONS!
                for x, y in positions:
                    self.field[y][x] = 0

                # check all positions above the piece, make sure they are empty, otherwise it is physically impossible for them to be there
                #if all([self.field[y_above][fin_x] == 0 for fin_x, fin_y in positions for y_above in range(fin_y) if (fin_x, y_above) not in positions]):
                score_moves.append((index, cord, positions, possible_reward, move_score))

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
        self.new_pop = None
        self.old_pop = None
        self.record = 0
        self.epochs = 10000
        self.games = 3
        self.pop_size = 100

    def eval_genomes(self, genomes, config):
        """for epoch in range(self.epochs):
            self.new_pop = Population(self.pop_size, self.old_pop)
            print('________________________________')
            print(f'Epoch: {epoch+1}')"""
        for genome_id, genome in genomes:
            #print(f'Member: {vector+1}')
            # print(f'{self.new_pop.models[vector]}')
            #print('_________________________________')
            current_fitness = 0
            tetris_game = tetris_ai.Tetris()

            self.agent = AI_Agent()
            self.agent.vector = neat.nn.FeedForwardNetwork.create(genome, config)

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
                    genome.fitness = current_fitness

                    # reset to a new tetris game, and reset the agent as well
                    break

            if current_fitness > self.record:
                self.record = current_fitness

                # print(f'Record: {self.record} *******')

        # self.old_pop = self.new_pop

    def train(self):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                             neat.DefaultStagnation, 'config-feedfoward.txt')

        p = neat.Population(config)

        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

        winner = p.run(self.eval_genomes)

        with open('winner.pkl', 'wb') as output:
            pickle.dump(winner, output, 1)


if __name__ == '__main__':
    trainer = Trainer()

    trainer.train()


