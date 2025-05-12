package pacman;

import utils.Commons;
import utils.GameController;

public class PacmanNeuralNetwork implements GameController, Comparable<PacmanNeuralNetwork> {
	
	private int inputDim;
	private int hiddenDim;
	private int outputDim;
	private double[][] hiddenWeights;
	private double[] hiddenBiases;
	private double[][] outputWeights;
	private double[] outputBiases;

	public PacmanNeuralNetwork(int inputDim, int hiddenDim, int outputDim) {
		this.inputDim = inputDim;
		this.hiddenDim = hiddenDim;
		this.outputDim = outputDim;
		initializeParameters();
	}
	
	public PacmanNeuralNetwork(int inputDim, int hiddenDim, int outputDim, double[] values) {
		this.inputDim = inputDim;
		this.hiddenDim = hiddenDim;
		this.outputDim = outputDim;
		
		int arrayLength = (inputDim * hiddenDim) + hiddenDim + (hiddenDim * outputDim) + outputDim;
		if (values.length != arrayLength) {
	        throw new IllegalArgumentException("The length of the values array must match the total number of parameters.");
	    }
		
		int index = 0;
		
		hiddenWeights = new double[inputDim][hiddenDim];
		for(int i = 0; i < inputDim; i++) {
			for(int j = 0; j < hiddenDim; j++) {
				hiddenWeights[i][j] = values[index++];
			}
		}
		
		hiddenBiases = new double[hiddenDim];
		for(int i = 0; i < hiddenDim; i++) {
			hiddenBiases[i] = values[index++];
		}
		
		outputWeights = new double[hiddenDim][outputDim];
		for(int i = 0; i < hiddenDim; i++) {
			for(int j = 0; j < outputDim; j++) {
				outputWeights[i][j] = values[index++];
			}
		}
		
		outputBiases = new double[outputDim];
		for(int i = 0; i < outputDim; i++) {
			outputBiases[i] = values[index++];
		}
	}
	
	public void initializeParameters() {
		
		double min = -0.1;
		double max = 0.1;
		
		hiddenWeights = new double[inputDim][hiddenDim];
		for(int i = 0; i < inputDim; i++) {
			for(int j = 0; j < hiddenDim; j++) {
				hiddenWeights[i][j] = min + Math.random() * (max - min);
			}
		}
		
		hiddenBiases = new double[hiddenDim];
		for(int i = 0; i < hiddenDim; i++) {
			hiddenBiases[i] = min + Math.random() * (max - min);
		}
		
		outputWeights = new double[hiddenDim][outputDim];
		for(int i = 0; i < hiddenDim; i++) {
			for(int j = 0; j < outputDim; j++) {
				outputWeights[i][j] = min + Math.random() * (max - min);
			}
		}
		
		outputBiases = new double[outputDim];
		for(int i = 0; i < outputDim; i++) {
			outputBiases[i] = min + Math.random() * (max - min);
		}
	}
	
	@Override
	public int nextMove(double[] currentState) {
		double[] output = forward(currentState);
		//System.out.println("Output array: " + Arrays.toString(output));
		int index = 0;
		double max = output[0];
		for (int i = 1; i < output.length; i++) {
            if (output[i] > max) {
                max = output[i];
                index = i;
            }
        }
		
		switch (index) {
	        case 0:
	            return 0;
	        case 1:
	            return 1;
	        case 2:
	            return 2;
	        case 3:
	            return 3;
	        case 4:
	            return 4;
	        default:
	            return 0;
		}								
	}
	
	public double[] forward(double[] inputValues) {
		//System.out.println("input: " + Arrays.toString(inputValues));
		double[] hiddenLayerOutput = new double[hiddenDim];
		for(int i = 0; i < hiddenDim; i++) {
			double sum = 0.0;
			for(int j = 0; j < inputDim; j++) {
				sum += inputValues[j] * hiddenWeights[j][i];
			}
			hiddenLayerOutput[i] = sigmoid(sum + hiddenBiases[i]);
		}
		
		double[] output = new double[outputDim];
		for(int i = 0; i < outputDim; i++) {
			double sum = 0.0;
			for(int j = 0; j < hiddenDim; j++) {
				sum += hiddenLayerOutput[j] * outputWeights[j][i];
			}
			output[i] = sigmoid(sum + outputBiases[i]);
		}
		
		return output;
	}
	
	public double[] getNeuralNetwork() {
		double[] neuralNetwork = new double[(inputDim * hiddenDim) + hiddenDim + (hiddenDim * outputDim) + outputDim];
		
		int index = 0;
		
		for(int i = 0; i < inputDim; i++) {
			for(int j = 0; j < hiddenDim; j++) {
				neuralNetwork[index++] = hiddenWeights[i][j];
			}
		}
		
		for(int i = 0; i < hiddenDim; i++) {
			neuralNetwork[index++] = hiddenBiases[i];
		}
		
		for(int i = 0; i < hiddenDim; i++) {
			for(int j = 0; j < outputDim; j++) {
				neuralNetwork[index++] = outputWeights[i][j];
			}
		}
		
		for(int i = 0; i < outputDim; i++) {
			neuralNetwork[index++] = outputBiases[i];
		}
		
		return neuralNetwork;
	}
	
	private double sigmoid(double x) {
		return 1 / (1 + Math.exp(-x));
	}
	
	@Override
	public String toString() {
		String result = "Neural Network:\n";
		String biasOutput = "Ouput biases: \n";
		for (int i = 0; i < outputDim; i++) {
			biasOutput += " bias ouput"+i+": " + outputBiases[i] + "\n";
		}
		result+= biasOutput;
		
		return result;
	}

	public int getInputDim() {
		return this.inputDim;
	}
	
	public int getHiddenDim() {
		return this.hiddenDim;
	}
	
	public int getOutputDim() {
		return this.outputDim;
	}

	@Override
	public int compareTo(PacmanNeuralNetwork o) {
		return Double.compare(o.getFitness(), this.getFitness());
	}
	
	public double getFitness() {
		PacmanBoard b = new PacmanBoard(this, false, Commons.PACMAN_SEED);
		b.runSimulation();
		return b.getFitness();
	}
	
}
