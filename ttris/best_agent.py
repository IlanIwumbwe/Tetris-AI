import pickle
import tetris_ai
import train_agent as a
import neat

def run():
    tetris_game = tetris_ai.Tetris()

    agent = a.AI_Agent(tetris_ai.ROWS, tetris_ai.COLUMNS)
    agent.get_possible_configurations()

    """
    setup best nueral network to replay
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, 'config-feedfoward.txt')

    with open('winner.pkl', "rb") as f:
        genome = pickle.load(f)

    agent.neural_network = neat.nn.FeedForwardNetwork.create(genome, config)

    """
    replay with best genome
    """
    while tetris_game.run:
        agent.landed = tetris_game.landed

        # update the agent with useful info to find the best move
        agent.update_agent(tetris_game.current_piece)

        tetris_game.best_move = agent.get_best_move(tetris_game.current_piece)

        tetris_game.game_logic()

        # make the move
        tetris_game.make_ai_move()

        agent.landed = tetris_game.landed

        # update the agent with useful info to find the best move
        agent.update_agent(tetris_game.current_piece)

        if tetris_game.change_piece:
            tetris_game.change_state()

        if not tetris_game.run:
            # reset to a new tetris game, and reset the agent as well
            break

if __name__ == '__main__':
    run()

