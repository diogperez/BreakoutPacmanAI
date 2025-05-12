package pacman;

import utils.GameController;

public class RandomController implements GameController {

	@Override
	public int nextMove(double[] currentState) {
		// TODO Auto-generated method stub
		return (int) (Math.random()*5);
	}

}
