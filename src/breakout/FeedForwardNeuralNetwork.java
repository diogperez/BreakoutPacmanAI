package breakout;

import utils.Commons;
import utils.GameController;

public class FeedForwardNeuralNetwork implements GameController, Comparable<FeedForwardNeuralNetwork> {
	
	private final int inputDim;
	private final int hiddenDim;
	private final int outputDim;
	private double[][] hiddenWeights;
	private double[] hiddenBiases;
	private double[][] outputWeights;
	private double[] outputBiases;

	public FeedForwardNeuralNetwork(int inputDim, int hiddenDim, int outputDim) {
		this.inputDim = inputDim;
		this.hiddenDim = hiddenDim;
		this.outputDim = outputDim;
		initializeParameters();
	}

	public FeedForwardNeuralNetwork(int inputDim, int hiddenDim, int outputDim, double[] values) {
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

	private void initializeParameters() {
		
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
		if (output[0] > output[1]) {
			return BreakoutBoard.LEFT;
		} else {
			return BreakoutBoard.RIGHT;
		}
	}

	public double[] forward(double[] inputValues) {
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
	public int compareTo(FeedForwardNeuralNetwork o) {
		return Double.compare(o.getFitness(), this.getFitness());
	}
	
	public double getFitness() {
		BreakoutBoard breakoutBoard = new BreakoutBoard(this, false, Commons.BREAKOUT_SEED);
		breakoutBoard.runSimulation();
		return breakoutBoard.getFitness();
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
	
}
