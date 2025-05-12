package breakout;

import java.util.Arrays;
import utils.Commons;

public class GeneticAlgorithm {
	
	private final int POPULATION_SIZE = 200;
	private final int NUM_GENERATIONS = 100;
	private final double MUTATION_RATE = 0.15;
	//private final double SELECTION_PERCENTAGE = 0.05;
	
	private final int K_TOURNAMENT = 6;
	
	private FeedForwardNeuralNetwork[] population = new FeedForwardNeuralNetwork[POPULATION_SIZE];
	
	public GeneticAlgorithm() {
		generatePopulation();
	}

	public FeedForwardNeuralNetwork search() {
		
		for (int i = 0; i < NUM_GENERATIONS; i++) {
			Arrays.sort(population);

			FeedForwardNeuralNetwork[] newGeneration = new FeedForwardNeuralNetwork[POPULATION_SIZE];

			for (int j = 0; j < POPULATION_SIZE - 1; j+=2) {
				FeedForwardNeuralNetwork parent1 = selectParent();
				FeedForwardNeuralNetwork parent2 = selectParent();
				FeedForwardNeuralNetwork[] children = crossover(parent1, parent2);
				mutate(children[0]);
				mutate(children[1]);
				newGeneration[j] = children[0];
				newGeneration[j+1] = children[1];
			}

			newGeneration[0] = population[0];         
			createNewPopulation(newGeneration);
			System.out.println("Geração: " + i + "\n" + "Melhor Fitness: " + population[0].getFitness());
		}
		return population[0];
	}

	private void createNewPopulation(FeedForwardNeuralNetwork[] newGeneration) {
		Arrays.sort(population);
		Arrays.sort(newGeneration);
		
	    for (int i = POPULATION_SIZE - 1; i > 0; i--) {
	    	newGeneration[i] = population[i];
	    }
		
	    population = newGeneration;
	}

	private void mutate(FeedForwardNeuralNetwork child) {
		double[] genes = child.getNeuralNetwork();
		for (int i = 0; i < genes.length; i++) {
			if (Math.random() < MUTATION_RATE) {
				genes[i] += (Math.random() - 0.5) * 0.1;
			}
		}
		child = new FeedForwardNeuralNetwork(Commons.INPUT_DIM,Commons.HIDDEN_DIM,Commons.OUTPUT_DIM, genes);

	}

	private FeedForwardNeuralNetwork[] crossover(FeedForwardNeuralNetwork parent1, FeedForwardNeuralNetwork parent2) {
		int crossoverPoint = (int) (Math.random() * parent1.getNeuralNetwork().length);

        double[] parent1Genes = parent1.getNeuralNetwork();
        double[] parent2Genes = parent2.getNeuralNetwork();
        double[] child1 = new double[parent1Genes.length];
        double[] child2 = new double[parent1Genes.length];
        
        for(int i = 0; i < crossoverPoint; i++) {                 
        	child1[i] = parent1Genes[i];
            child2[i] = parent2Genes[i];
        }
        
        for (int i = crossoverPoint; i < parent1.getNeuralNetwork().length; i++) {     
            child1[i] = parent2Genes[i];
            child2[i] = parent1Genes[i];
        }
        
        return new FeedForwardNeuralNetwork[] {new FeedForwardNeuralNetwork(Commons.INPUT_DIM, Commons.HIDDEN_DIM, Commons.OUTPUT_DIM, child1), 
        									   new FeedForwardNeuralNetwork(Commons.INPUT_DIM, Commons.HIDDEN_DIM, Commons.OUTPUT_DIM, child2)};
	}

	private FeedForwardNeuralNetwork selectParent() {
		FeedForwardNeuralNetwork best = population[(int) (Math.random() * POPULATION_SIZE)];
	    for(int i = 0; i < K_TOURNAMENT; i++) {
	        FeedForwardNeuralNetwork contender = population[(int) (Math.random() * POPULATION_SIZE)];
	        if (contender != null && contender.getFitness() > best.getFitness()) {
	            best = contender;
	        }
	    }
	    return best;
	}
	
	private void generatePopulation() {
		for (int i = 0; i < POPULATION_SIZE; i++) {
			population[i] = new FeedForwardNeuralNetwork(Commons.INPUT_DIM,Commons.HIDDEN_DIM,Commons.OUTPUT_DIM);    
		}
	}
}