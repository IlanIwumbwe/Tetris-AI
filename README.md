# Tetris
This code is for my NEA project, A level computer science. 

_____________________________________________________________________________
Optimisation using a genetic algorithm to make an AI that plays good Tetris

- train_agent.py uses NEAT-python module without next piece knowledge
- train_agent2.py uses NEAT-python with next piece knowledge, but hasn't been optimised to work fast enough in training
- train_agent_scratch.py uses a genetic algorithm written from scratch, without next piece knowledge

Best results so far is over 1.9 million score by the genetic algorithm from scratch by generation 4. 

The population and nueral net are coded in neuralnet.py

train_agent_scratch.py now has capability to plot graphs after running all epochs, as well as saving a population every 2 epochs.


A few things:

The neural network and genetic algorithm coded from scratch gathered inspiration from these projects:

- Used a gameboy emulator and pytorch to build a Tetris AI from scratch: https://towardsdatascience.com/beating-the-world-record-in-tetris-gb-with-genetics-algorithm-6c0b2f5ace9b
- Explains use of a genetic algorithm without neural networks: https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/
- The neural network was built upon an the example in Tariq Rashid's book (Build your own Neural Network) which uses a neural network to classify handwritten digits from the MNIST database: https://www.goodreads.com/en/book/show/29746976-make-your-own-neural-network
- Explanation of genetic algorithms: https://lethain.com/genetic-algorithms-cool-name-damn-simple/
- Ideas for implementation from Greer Viau, who also used a genetic algorithm to build a Tetris test: https://www.youtube.com/watch?v=1yXBNKubb2o&t=482s 
