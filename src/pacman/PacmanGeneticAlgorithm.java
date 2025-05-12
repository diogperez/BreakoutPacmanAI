package pacman;

import java.util.Arrays;
import utils.Commons;

public class PacmanGeneticAlgorithm {
	
	private final int POPULATION_SIZE = 200;
	private final int NUM_GENERATIONS = 100;
	private final double MUTATION_RATE = 0.25;
	//private final double SELECTION_PERCENTAGE = 0.05;
	
	final int K_TOURNAMENT = 6;
	

	private PacmanNeuralNetwork[] population = new PacmanNeuralNetwork[POPULATION_SIZE];
	
	public PacmanGeneticAlgorithm() {
		generatePopulation();
	}

	public PacmanNeuralNetwork search() {
		
		for (int i = 0; i < NUM_GENERATIONS; i++) {
			Arrays.sort(population);

			PacmanNeuralNetwork[] newGeneration = new PacmanNeuralNetwork[POPULATION_SIZE];

			for (int j = 0; j < POPULATION_SIZE - 1; j+=2) {
				PacmanNeuralNetwork parent1 = selectParent();
				PacmanNeuralNetwork parent2 = selectParent();
				PacmanNeuralNetwork[] children = crossover(parent1, parent2);
				mutate(children[0]);
				mutate(children[1]);
				newGeneration[j] = children[0];
				newGeneration[j+1] = children[1];
			}

			newGeneration[0] = population[0];         // Mantém o melhor individuo
			createNewPopulation(newGeneration);
			System.out.println("Geração: " + i + "\n" + "Melhor Fitness: " + population[0].getFitness());
		}
		return population[0];
	}
	
	private void createNewPopulation(PacmanNeuralNetwork[] newGeneration) {
		Arrays.sort(population);
		Arrays.sort(newGeneration);
		
	    for (int i = POPULATION_SIZE - 1; i > 0; i--) {
	    	newGeneration[i] = population[i];
	    }
		
	    population = newGeneration;
	}

	private void mutate(PacmanNeuralNetwork child) {
		double[] genes = child.getNeuralNetwork();
		for (int i = 0; i < genes.length; i++) {
			if (Math.random() < MUTATION_RATE) {
				genes[i] += (Math.random() - 0.5) * 0.1;
			}
		}
		child = new PacmanNeuralNetwork(Commons.INPUT_DIM_PAC,Commons.HIDDEN_DIM_PAC,Commons.OUTPUT_DIM_PAC, genes);

	}

	private PacmanNeuralNetwork[] crossover(PacmanNeuralNetwork parent1, PacmanNeuralNetwork parent2) {
		int crossoverPoint = (int) (Math.random() * parent1.getNeuralNetwork().length);

        double[] parent1Genes = parent1.getNeuralNetwork();
        double[] parent2Genes = parent2.getNeuralNetwork();
        double[] child1 = new double[parent1Genes.length];
        double[] child2 = new double[parent1Genes.length];
        
        for(int i = 0; i < crossoverPoint; i++) {                 //Percorre o array até ao ponto
        	child1[i] = parent1Genes[i];
            child2[i] = parent2Genes[i];
        }
        
        for (int i = crossoverPoint; i < parent1.getNeuralNetwork().length; i++) {     //Percorre a partir do ponto
            child1[i] = parent2Genes[i];
            child2[i] = parent1Genes[i];
        }
        
        return new PacmanNeuralNetwork[] {new PacmanNeuralNetwork(Commons.INPUT_DIM_PAC, Commons.HIDDEN_DIM_PAC, Commons.OUTPUT_DIM_PAC, child1), 
        									   new PacmanNeuralNetwork(Commons.INPUT_DIM_PAC, Commons.HIDDEN_DIM_PAC, Commons.OUTPUT_DIM_PAC, child2)};
	}

	private PacmanNeuralNetwork selectParent() {
		PacmanNeuralNetwork best = population[(int) (Math.random() * POPULATION_SIZE)];
	    for(int i = 0; i < K_TOURNAMENT; i++) {
	        PacmanNeuralNetwork contender = population[(int) (Math.random() * POPULATION_SIZE)];
	        if (contender != null && contender.getFitness() > best.getFitness()) {
	            best = contender;
	        }
	    }
	    return best;
	}
	
	private void generatePopulation() {
		for (int i = 0; i < POPULATION_SIZE; i++) {
			population[i] = new PacmanNeuralNetwork(Commons.INPUT_DIM_PAC,Commons.HIDDEN_DIM_PAC,Commons.OUTPUT_DIM_PAC);    
			System.out.println("POPULATION:" + population[i]);
		}
	}
	
}
