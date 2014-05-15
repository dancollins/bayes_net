/**
 * Simple class for approximate inference based on the Wet Grass network.
 */
public class BayesNet {

	/**
	 * Inner class for representing a node in the network.
	 */
	private class Node {

		// The name of the node
		private String name;

		// The parent nodes
		private Node[] parents;

		// The probabilities for the CPT
		private double[] probs;

		// The current value of the node
		private boolean value;

		/**
		 * Initializes the node.
		 */
		private Node(String n, Node[] pa, double[] pr) {
			name = n;
			parents = pa;
			probs = pr;
		}

		/**
		 * Returns conditional probability of value "true" for the current node
		 * based on the values of the parent nodes.
		 * 
		 * @return The conditional probability of this node, given its parents.
		 */
		private double conditionalProbability() {

			int index = 0;
			for (int i = 0; i < parents.length; i++) {
				if (parents[i].value == false) {
					index += Math.pow(2, parents.length - i - 1);
				}
			}
			return probs[index];
		}
	}

	// The list of nodes in the Bayes net
	private Node[] nodes;

	// A collection of examples describing whether it is { raining, playing
	// }
	private static final boolean[][] PLAYING_EXAMPLES = { { true, false },
			{ true, false }, { true, true }, { true, false }, { true, false },
			{ false, true }, { false, false }, { false, true },
			{ false, true }, { false, true }, { false, false },
			{ false, true }, { false, true } };

	/**
	 * Constructor that sets up the Wet Grass network.
	 */
	public BayesNet() {

		nodes = new Node[6];

		nodes[0] = new Node("Cloudy", new Node[] {}, new double[] { 0.4 });
		nodes[1] = new Node("Drought", new Node[] {}, new double[] { 0.002 });
		nodes[2] = new Node("Sprinkler", new Node[] { nodes[0], nodes[1] },
				new double[] { 0.02, 0.1, 0.15, 0.5 });
		nodes[3] = new Node("Rain", new Node[] { nodes[0] }, new double[] {
				0.8, 0.1 });
		nodes[4] = new Node("PlayOutside", new Node[] { nodes[3] },
				calculatePlayOutsideProbabilities(PLAYING_EXAMPLES));
		nodes[5] = new Node("WetGrass", new Node[] { nodes[2], nodes[3] },
				new double[] { 0.99, 0.9, 0.9, 0.0 });
	}

	/**
	 * Prints the current state of the network to standard out.
	 */
	public void printState() {

		for (int i = 0; i < nodes.length; i++) {
			if (i > 0) {
				System.out.print(", ");
			}
			System.out.print(nodes[i].name + " = " + nodes[i].value);
		}
		System.out.println();
	}

	/**
	 * Calculates the probability that a child will play outside based on
	 * whether it is raining or not. Returns the conditional probabilities that
	 * a child will play when it is raining and when it is not raining.
	 * 
	 * @param rainingInstances
	 *            A set of training examples in the form { raining, playing }
	 *            from which to compute the probabilities.
	 * @return The probability that a child will play outside when it is {
	 *         raining, !raining }.
	 */
	public double[] calculatePlayOutsideProbabilities(
			boolean[][] rainingInstances) {
		double[] probabilities = new double[2];

		int[] playing = new int[2];
		int[] total = new int[2];

		// Count the number of times the child plays for both
		// raining and not raining.
		for (boolean[] sample : rainingInstances) {
		    if (sample[0]) {
			// raining
			// Increment the playing count if the child
			// is playing, and increment the total sample
			// count.
			playing[0] += sample[1] ? 1 : 0;
			total[0]++;
		    } else {
			// not raining
			// Increment the playing count if the child
			// is playing, and increment the total sample
			// count.
			playing[1] += sample[1] ? 1 : 0;
			total[1]++;
		    }
		}

		// P(p|r)
		probabilities[0] = (double)playing[0] / (double)total[0];
		// P(p|~r)
		probabilities[1] = (double)playing[1] / (double)total[1];

		return probabilities;
	}

	/**
	 * This method assigns new values to the nodes in the network by sampling
	 * from the joint distribution (based on PRIOR-SAMPLE method from text
	 * book/slides).
	 */
	public void priorSample() {

		// YOUR CODE HERE
	}

	/**
	 * Rejection sampling. Returns probability of query variable being true
	 * given the values of the evidence variables, estimated based on the given
	 * total number of samples (see REJECTION-SAMPLING method from text
	 * book/slides).
	 * 
	 * The nodes/variables are specified by their indices in the nodes array.
	 * The array evidenceValues has one value for each index in
	 * indicesOfEvidenceNodes. See also examples in main().
	 * 
	 * @param queryNode
	 *            The variable for which rejection sampling is calculating.
	 * @param indicesOfEvidenceNodes
	 *            The indices of the evidence nodes.
	 * @param evidenceValues
	 *            The values of the indexed evidence nodes.
	 * @param N
	 *            The number of iterations to perform rejection sampling.
	 * @return The probability that the query variable is true given the
	 *         evidence.
	 */
	public double rejectionSampling(int queryNode,
			int[] indicesOfEvidenceNodes, boolean[] evidenceValues, int N) {

		return 0; // REPLACE THIS LINE BY YOUR CODE
	}

	/**
	 * This method assigns new values to the non-evidence nodes in the network
	 * and computes a weight based on the evidence nodes (based on
	 * WEIGHTED-SAMPLE method from text book/slides).
	 * 
	 * The evidence is specified as in the case of rejectionSampling().
	 * 
	 * @param indicesOfEvidenceNodes
	 *            The indices of the evidence nodes.
	 * @param evidenceValues
	 *            The values of the indexed evidence nodes.
	 * @return The weight of the event occurring.
	 * 
	 */
	public double weightedSample(int[] indicesOfEvidenceNodes,
			boolean[] evidenceValues) {

		return 0; // REPLACE THIS LINE BY YOUR CODE
	}

	/**
	 * Likelihood weighting. Returns probability of query variable being true
	 * given the values of the evidence variables, estimated based on the given
	 * total number of samples (see LIKELIHOOD-WEIGHTING method from text
	 * book/slides).
	 * 
	 * The parameters are the same as in the case of rejectionSampling().
	 * 
	 * @param queryNode
	 *            The variable for which rejection sampling is calculating.
	 * @param indicesOfEvidenceNodes
	 *            The indices of the evidence nodes.
	 * @param evidenceValues
	 *            The values of the indexed evidence nodes.
	 * @param N
	 *            The number of iterations to perform rejection sampling.
	 * @return The probability that the query variable is true given the
	 *         evidence.
	 */
	public double likelihoodWeighting(int queryNode,
			int[] indicesOfEvidenceNodes, boolean[] evidenceValues, int N) {

		return 0; // REPLACE THIS LINE BY YOUR CODE
	}

	/**
	 * MCMC inference. Returns probability of query variable being true given
	 * the values of the evidence variables, estimated based on the given total
	 * number of samples (see MCMC-ASK method from text book/slides).
	 * 
	 * The parameters are the same as in the case of rejectionSampling().
	 * 
	 * @param queryNode
	 *            The variable for which rejection sampling is calculating.
	 * @param indicesOfEvidenceNodes
	 *            The indices of the evidence nodes.
	 * @param evidenceValues
	 *            The values of the indexed evidence nodes.
	 * @param N
	 *            The number of iterations to perform rejection sampling.
	 * @return The probability that the query variable is true given the
	 *         evidence.
	 */
	public double MCMCask(int queryNode, int[] indicesOfEvidenceNodes,
			boolean[] evidenceValues, int N) {

		return 0; // REPLACE THIS LINE BY YOUR CODE
	}

	/**
	 * The main method, with some example method calls.
	 */
	public static void main(String[] ops) {

		// Create network.
		BayesNet b = new BayesNet();

		double[] playOutsideProbabilities = b
				.calculatePlayOutsideProbabilities(PLAYING_EXAMPLES);
		System.out.println("When it is raining, I play outside "
				+ (playOutsideProbabilities[0] * 100) + "% of the time.");
		System.out.println("When it is not raining, I play outside "
				+ (playOutsideProbabilities[1] * 100) + "% of the time.");

		// Sample five states from joint distribution and print them
		for (int i = 0; i < 5; i++) {
			b.priorSample();
			b.printState();
		}

		// Print out results of some example queries based on rejection
		// sampling.
		// Same should be possible with likelihood weighting and MCMC inference.

		// Probability of rain given wet grass and NOT playing outside.
		System.out.println(b.rejectionSampling(3, new int[] { 4, 5 },
				new boolean[] { false, true }, 10000));

		// Probability of sprinklers given a drought
		System.out.println(b.rejectionSampling(2, new int[] { 1 },
				new boolean[] { true }, 100000));

		// Probability of wet grass given rain and sprinklers
		System.out.println(b.rejectionSampling(5, new int[] { 2, 3 },
				new boolean[] { true, true }, 10000000));
	}
}
