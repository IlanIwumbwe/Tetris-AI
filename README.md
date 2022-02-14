# Tetris
This code is for my NEA project, A level computer science. 

_____________________________________________________________________________
Optimisation using a genetic algorithm to make an AI that plays good Tetris

- train_agent.py uses NEAT-python module without next piece knowledge
- train_agent2.py uses NEAT-python with next piece knowledge, but hasn't been optimised to work fast enough in training
- train_agent_scratch.py uses a genetic algorithm written from scratch, without next piece knowledge

Best results so far is over 1million score by the genetic algorithm from scratch by generation 6. 

The population and nueral net are coded in neuralnet.py

train_agent_scratch.py now has capability to plot graphs after running all epochs, as well as saving a population every 2 epochs.
