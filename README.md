# Tetris
Optimisation using a genetic algorithm to make an AI that plays good Tetris

- train_agent.py uses NEAT-python module without next piece knowledge
- train_agent2.py uses NEAT-python with next piece knowledge, but hasn't been optimised to work fast enough in training
- train-agent_scratch.py uses a genetic algorithm written from scratch, without next piece knowledge

Best results so far is over 600k score by the genetic algorithm from scratch by generation 4. 

The population and nueral net are coded in neuralnet.py
