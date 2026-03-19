NEAT - NeuroEvolution of Augmenting Topologies

[Topology in Neural Networks](https://link.springer.com/rwe/10.1007/978-0-387-30164-8_837) - The way that neurons are connected

TWEANNs - Topology and Weight Evolving Artificial Neural Networks

### Main Technical Challenges:
**1: Genetic Representation of Topology that Allows for Crossover**
**2: Preserve Topological Innovations that Need Several Generations to Begin Performing Well**
**3: Minimizing Topologies Without Specifically Contrived Fitness Functions**

### Genetic Representation and Crossover
#### Methods
##### Graphs
Good way to represent topology, but can lead to size-limits and complexity in crossover
Subgraph-Swapping: Swap subsections of graphs, a method of crossover that helps to preserve the semantic significance of topological structures
##### Bitstrings
Straightforward crossover operations, but the semantics of that crossover might be iffy, the length of these strings can be enormous, and crossover usually requires same-length bitstrings, constraining the size of the network greatly. 
##### Nonmating
Give up on the idea of crossover entirely. Crossover of different topologies can often lead to loss of functionality, so why not give up and cry (me core). 
##### Indirect Encoding
In one example, genomes are programs written in a specialized graph transformation language. 




#### Problems
##### Competing Conventions
There are many ways to represent the "same" solution to a problem in a genome, and when two species develop two different but functionally identical structures to solve the same problem, crossover will often create damaged offspring. There have been attempts to create non-redundant genetic representations, but non have been successfully applied to TWEANNs. 
Nature's solution to this is *homology*: two genes are homologous if they are alleles of the same trait. The main insight in NEAT is that the historical origin of two genes is direct evidence of homology if the genes share the same origin. Thus, NEAT performs artificial synapsis based on historical markings, allowing it to add new structure without losing track of which gene is which over the course of a simulation. 


## NEAT
## Genetic Encoding
*Node genes* provide a list of inputs, hidden nodes, and outputs that can be connected. *Connection genes* refer to two *node genes* being connected, and specify the weight of the connection, whether or not the connection is active (with a boolean), and an *innovation number* for finding corresponding genes (uhhhhh?).
### Mutation
Connection weights: As expected
Structure/topology: 
 1) Add connection
 2) Add node

# Areas for expansion

## Convolutional components

## Multi-directional information flow
The math would *prolly get real fuckin complicated*, but what if you allow for information to flow backwards as well, i.e. you have intput (i), neurons (A,B,C), output (o), and connections (i,A), (A,B), (B,C), (B,o), *(A,C)* 

Of course, this creates a cycle, so there has to be a time-aspect, e.g. break neurons into layers, then only perform actions on one layer each timestamp. Basically throw out the entire idea of NNs being procedural. You could call it a **Non-Procedural Neural Network (NPNN)**

## Crossover and Genetic Representation

### Proximity-Based Priority for Connections
Use some sort of physics-ish simulation to calculate positions for neurons, and prioritize connections between closer neurons.

### Sequential vs. Parallel Expression of Genes
Should a genome describe the exact topology with little dependence, or should it describe how the topology is constructed? 
E.g. a genome could describe that a structure will be appended to the nearest available connection, rather than to a specific node. This might make complex insertions of nodes/connections/structures/etc easier, but might **vastly** increase complexity. 

##### Some immediate problems:
Deciding positions to start from: Do we start from a particular node? Is there a particular "we are constructing from here" position that will be constantly updated (this seems a bit too arbitrary)
Start/stop conditions: Say we want to repeat a structure across a specific array of nodes. How do we tell where that array is? Where is the beginning, where is the end? It it based purely on proximity? Based on properties of the nodes? 

### Placeholder