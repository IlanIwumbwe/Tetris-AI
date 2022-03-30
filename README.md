# Tetris
This code is for my NEA project, A level computer science. 

_____________________________________________________________________________
Optimisation using a genetic algorithm to make an AI that plays good Tetris
- train_agent_scratch.py uses a genetic algorithm written from scratch, without next piece knowledge (hasn't been optimised yet)
- With next piece knowledge, agent gains over 400k by 2nd generation, in comparison, without, it gains over 400k at 7th generation

The population and nueral net are coded in neuralnet.py

Update log
_______________________________________________________
- train_agent_scratch.py now has capability to plot graphs after running all epochs, as well as saving a population every 2 epochs
- tetris.py -> human playable version now has complex moves implemented such as T-spins

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

How to train: 
_____________________________________

- Run tetris_ai_scratch.py to train the agent. It gives prompt for whether you want to load a population. If you want to, type "Y", it then asks from which epoch, type 
2, 4, 6 ,8 or 10. 
- If you don't want to load a population, just enter. 

Results
_____________________________________
![av_fitness](https://user-images.githubusercontent.com/56346800/160808808-9b650cb4-fae9-4637-b2c4-8f5732cb1f96.png)

![av_score](https://user-images.githubusercontent.com/56346800/160808836-69ad4046-2200-4628-90f2-6b57addbdb87.png)

* Training 10 epochs takes about 3 days, with a highscore of over 1 million. 
