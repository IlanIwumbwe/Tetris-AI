# Tetris
This code is for my NEA project, A level computer science. 

_____________________________________________________________________________
Optimisation using a genetic algorithm to make an AI that plays good Tetris
- train_agent_scratch.py uses a genetic algorithm written from scratch, without next piece knowledge (hasn't been optimised yet)

The population and nueral net are coded in neuralnet.py

Update log
_______________________________________________________
- train_agent_scratch.py now has capability to plot graphs after running all epochs, as well as saving a population every 2 epochs
- tetris.py -> human playable version now has complex moves implemented such as T-spins
- Ablation tests are now possible in tetris_agent_scatch.py

A few things:
_______________

This project would not have been possible without help from:

- Used a gameboy emulator and pytorch to build a Tetris AI from scratch: https://towardsdatascience.com/beating-the-world-record-in-tetris-gb-with-genetics-algorithm-6c0b2f5ace9b
- Explains use of a genetic algorithm without neural networks: https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/
- The neural network was built upon an the example in Tariq Rashid's book (Build your own Neural Network) which uses a neural network to classify handwritten digits from the MNIST database: https://www.goodreads.com/en/book/show/29746976-make-your-own-neural-network
- Explanation of genetic algorithms: https://lethain.com/genetic-algorithms-cool-name-damn-simple/
- Ideas for implementation from Greer Viau, who also used a genetic algorithm to build a Tetris agent: https://www.youtube.com/watch?v=1yXBNKubb2o&t=482s 
- The original paper on NEAT-Python: http://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf
- Using NEAT-Python for Sonic the Hedgehog: https://www.youtube.com/watch?v=pClGmU1JEsM&list=PLTWFMbPFsvz3CeozHfeuJIXWAJMkPtAdS
- Explanation of wall kicks and the Super Rotation System: https://harddrop.com/wiki/SR
- Uses a local search algorithm to build a Tetris agent: https://github.com/saagar/ai-tetris/blob/master/paper/tetrais.pdf

How to train: 
_____________________________________

- Run tetris_ai_scratch.py to train the agent. It gives prompt for whether you want to load a population. If you want to, type "Y", it then asks from which epoch, type 
2, 4, 6 ,8 or 10. 
- If you don't want to load a population, just enter. 

Results
_______________________________
- Over 1 million highscore after 10 epochs of training 

Graphing:
If graphs don't work automatically, program will print out all the data with text so you know what is what, you'll have to manually pass it into
the visualization class. Make an object in that file and pass data in there, then run the visualisation file. Sorry 


![av_fitness](https://user-images.githubusercontent.com/56346800/162254605-8b0fb67c-1f60-45b1-9aea-97c2fb17eb83.png)
![av_score](https://user-images.githubusercontent.com/56346800/162254617-61dce174-5056-4c1e-b6cd-f347576ee134.png)
