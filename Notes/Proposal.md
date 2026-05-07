# Recreation of Results/Data

Verification: (4.2) XORs
Benchmark: (4.3) Pole Balancing

# Imrpovement/Modifications/Experimentation

## Main focus: Experimentation with gene representation

How can we make representations of connections more generalized? E.g. instead of a genome specifying a connection from node X to node Y, is there a way to form connections based on something more natural, such as proximity (which would require defining geometric positions for nodes)?

## Additional: Applying to memory and/or timing dependent tasks

With recurrant neural networks, neural circuits can be created to represent memory or keep track of time (so long as time is consistent in the environment). Some examples of a complex task could be:
 - Maze Solve and Memorize: Traverse a maze, then traverse back to the start on an optimal path using your memory of the maze. 
 - Firefighting: A game where fires will appear randomly in an area. You can choose to take time to clear an area of flamable debrees to prevent fires, and can use water to put out fires. An optimal solution would require periodically checking/cleaning each area at somewhat regular intervals, but randomness in timing of fires starting will require intelligent reaction/adaptation as well. 



# Actual Proposal

For my graduate project, I plan to recreate the results from "Evolving Neural Networks through Augmenting Topologies" by Kenneth O. Stanley and Risto Miikkulainen using their "NEAT" method. NEAT is a foundational but effective method of genetically evolving neural networks to optimally solve complex tasks. For verification of basic functionality, they have NEAT learn how to construct an XOR gate, and take note of the number of neurons in the final network, the number of generations, and then number of evaluations. For more robust benchmarks, they utilize two variations of the pole-balancing problem, and take note of the same aforementioned metrics. These evaluations will be the results I strive to recreate. 

From there, there are two kinds of experimentation I am interested in performing: 

Different methods of gene representation: In particular, I wish to explore the idea of generalizing construction of neural networks in such a way that connections do not need to be arbitrarily defined (i.e. I want to avoid genes that describe something like "Connect node X to node Y", where X and Y are arbitrary numbers/IDs with no significance to each other). This will almost certainly involve research into the works of other authors, but should also include personal modifications or methods. 

Applications involving memory and timing: NEAT is capable of creating recurrent neural networks, which can have structures capable of storing memory and measuring time. In theory, this means that NEAT should be able to optimally solve complex tasks such as memorizing a path through a maze (rather than re-solving it), or some kind of stochastically periodic cleanup/maintenance task. 

One additional goal (more personal, not a hard requirement) is to create some form of API for this program, allowing one to create their own tasks, genetic representations, learning/evolving algorithms, etc., without unreasonable difficulty. The program should also support somewhat vendor-independent GPU-accelerated computation, as well as CPU-only computation for easier setup and debugging. 