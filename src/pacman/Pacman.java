package pacman;

import java.awt.EventQueue;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;

import javax.swing.JFrame;

import utils.Commons;
import utils.GameController;

public class Pacman extends JFrame {

	public Pacman(GameController c, boolean b, int seed) {
		EventQueue.invokeLater(() -> {
			add(new PacmanBoard(c, b, seed));
			setTitle("Pacman");
			setDefaultCloseOperation(EXIT_ON_CLOSE);
			setSize(380, 420);
			setLocationRelativeTo(null);
			setVisible(true);
		});
	}
	
	public static void main(String[] args) {
		PacmanGeneticAlgorithm g = new PacmanGeneticAlgorithm();
        PacmanNeuralNetwork nn = g.search();
        new Pacman(nn, true, Commons.PACMAN_SEED);

	}
}
