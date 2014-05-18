'''
Simple class for approximate inference based on the Wet Grass network.

Based on the Java skeleton provided with the assignment.

Dan Collins 2014
1183446
'''

import random

# A collection of examples describing whether it is {raining, playing}
PLAYING_EXAMPLES = [(True, False), (True, False), (True, True),
                    (True, False), (True, False), (False, True), 
                    (False, False), (False, True), (False, True),
                    (False, True), (False, False), (False, True), 
                    (False, True)]

# A node in the bayes network
class Node:
    def __init__(self, n, pa, pr):
        self.name = n
        self.parents = pa
        self.probs = pr
        self.value = None

    '''
    Returns conditional probability of value "true" for the current
    node based on the values of the parent node/s.
    '''
    def conditionalProbability(self):
        index = 0
        for i, p in enumerate(self.parents):
            if p.value == False:
                index += 2 ** (len(self.parents) - i - 1)

        return self.probs[index]

class BayesNet:
    # The nodes in the network
    nodes = []

    # Build the initial network
    def __init__(self):
        self.nodes.append(Node("Cloudy", [], [0.4]))
        self.nodes.append(Node("Drought", [], [0.002]))
        self.nodes.append(Node("Sprinkler",
                               [self.nodes[0], self.nodes[1]],
                               [0.02, 0.1, 0.15, 0.5]))
        self.nodes.append(Node("Rain", [self.nodes[0]], [0.8, 0.1]))
        self.nodes.append(Node("PlayOutside", [self.nodes[3]],
                          self.calculatePlayOutsideProbabilities(PLAYING_EXAMPLES)))
        self.nodes.append(Node("WetGrass",
                               [self.nodes[2], self.nodes[3]],
                               [0.99, 0.9, 0.9, 0.0]))

    # Prints the current state of the network to stdout
    def printState(self):
        strings = []
        for node in self.nodes:
            strings.append(node.name + " = " + str(node.value))

        print ", ".join(strings)

    '''
    Calculates probability that a child will play outside base on
    whether it is raining or not.  Returns the conditional
    probabilities that a child will play when it is raining and when
    it is not raining.

    rainingInstances
            A set of training examples in the form {raining, playing}
            from which to compute the probabilities.
    '''
    def calculatePlayOutsideProbabilities(self, rainingInstances):
        playing = [0, 0]
        total = [0, 0]
        prob = [0.0, 0.0]

        for sample in rainingInstances:
            if sample[0]:
                playing[0] += 1 if sample[1] else 0
                total[0] += 1
            else:
                playing[1] += 1 if sample[1] else 0
                total[1] += 1

        prob[0] = float(playing[0]) / float(total[0])
        prob[1] = float(playing[1]) / float(total[1])
        
        return prob

    '''
    This method will sample the value for a node given its
    conditional probability.
    '''
    def sampleNode(self, node):
        node.value = True if random.random() <= node.conditionalProbability() else False

    '''
    This method assigns new values to the nodes in the network by
    sampling from the joint distribution.  Based on the PRIOR-SAMPLE
    from the text book/slides
    '''
    def priorSample(self):
        for n in self.nodes:
            self.sampleNode(n)

    '''
    This method will return true if all the evidence variables in the
    network have the value specified by the evidence values.
    '''
    def testModel(self, indicesOfEvidenceNodes, evidenceValues):
        for i in range(len(indicesOfEvidenceNodes)):
            if (self.nodes[indicesOfEvidenceNodes[i]].value != evidenceValues[i]):
                return False

        return True

    '''
    Rejection Sampling
    This method returns the probability of the query variable being
    true given the values of the evidence variables, estimated based
    on the given total number of samples. Based on the
    REJECTION-SAMPLING method in the text book/slides.

    queryNode
            The variable for which rejection sampling is calculating.
    indicesOfEvidenceNodes
            The indicies of the evidence nodes.
    evidenceValues
            The values of the indexed evidence nodes.
    N
            The number of iterations to perform rejection sampling
    '''
    def rejectionSampling(self, queryNode, indicesOfEvidenceNodes,
                          evidenceValues, N):
        counts = [0, 0]

        for i in range(N):
            self.priorSample()
            if (self.testModel(indicesOfEvidenceNodes, evidenceValues)):
                if (self.nodes[queryNode].value):
                    counts[0] += 1
                else:
                    counts[1] += 1

        return float(counts[0]) / float(counts[0] + counts[1])

    '''
    This method assigns new values to the non-evidence nodes in the
    network and computes a weight based on the evidence nodes.
    Based on WEIGHTED-SAMPLE methon in the text book/slides.
    '''
    def weightedSample(self, indicesOfEvidenceNodes, evidenceValues):
        weight = 1.0

        # Calculate the weight, or set values in the network
        for i, n in enumerate(self.nodes):
            if i in indicesOfEvidenceNodes:
                n.value = evidenceValues[indicesOfEvidenceNodes.index(i)]
                prob = n.conditionalProbability()
                if n.value:
                    weight *= prob
                else:
                    weight *= (1.0 - prob)
            else:
                self.sampleNode(n)

        return weight

    '''
    Likelihood Weighting
    This method returns the probability of the query variable being
    true given the values of the evidence variables, estimated based
    on the given total number of samples.  Based on the
    LIKEILYHOOD-WEIGHTING method in the text book/slides.

    queryNode
            The variable for which likelihood weighting is calculating.
    indicesOfEvidenceNodes
            The indicies of the evidence nodes.
    evidenceValues
            The values of the indexed evidence nodes.
    N
            The number of iterations to perform rejection sampling
    '''
    def likelihoodWeighting(self, queryNode, indicesOfEvidenceNodes,
                          evidenceValues, N):
        weights = [0.0, 0.0]
        node = self.nodes[queryNode]

        for i in range(N):
            w = self.weightedSample(indicesOfEvidenceNodes, evidenceValues)
            if (node.value):
                weights[0] += w
            else:
                weights[1] += w

        return weights[0] / (weights[0] + weights[1])

    '''
    This method will return a list of a node's children

    queryNode
            The node to create the markov blanket for
    '''
    def children(self, queryNode):
        children = []
        for n in self.nodes:
            # If we don't already have it in our list, add it
            if (queryNode in n.parents
                and n not in children):
                children.append(n)

        return children
            

    '''
    Markov-Chain Monte Carlo Inference.
    This method returns the probability of the query variable being
    true given the values of the evidence variables, estimated based
    on the given total number of samples.  Based on the
    MCMC-ASK method in the text book/slides.

    queryNode
            The variable for which MCMC is calculating.
    indicesOfEvidenceNodes
            The indicies of the evidence nodes.
    evidenceValues
            The values of the indexed evidence nodes.
    N
            The number of iterations to perform rejection sampling
    '''
    def MCMCask(self, queryNode, indicesOfEvidenceNodes,
                          evidenceValues, N):
        counts = [0, 0]

        # Set the evidence nodes
        for i, value in enumerate(evidenceValues):
            self.nodes[indicesOfEvidenceNodes[i]].value = value

        # Get a reference to the query node
        qn = self.nodes[queryNode]

        # Build a list of non evidence nodes.
        nonEvidence = [self.nodes[i]
                       for i in range(len(self.nodes))
                       if i not in indicesOfEvidenceNodes]

        # Cache the children of the non-evidence nodes
        children = {}
        for n in nonEvidence:
            children[n] = self.children(n)
        
        # Randomly assign values to the non-evidence variables
        for n in nonEvidence:
            n.value = True if random.random() <= 0.5 else False

        # Conduct the trial N times
        for i in range(N):
            # Increment the count based on the current state of the
            # query node
            if qn.value:
                counts[0] += 1
            else:
                counts[1] += 1

            for n in nonEvidence:
                # Calculate the conditional probability of this node
                # given its parents
                # P(X | Parents(X))
                P_true = n.conditionalProbability()
                P_false = 1.0 - P_true

                # Calculate the conditional probabilities of the
                # child nodes given their parents
                for c in children[n]:
                    # P(c|n=true,OtherParents(c))
                    n.value = True
                    pt = c.conditionalProbability()

                    # P(c|n=false,OtherParents(c))
                    n.value = False
                    pf = c.conditionalProbability()

                    # If the node is false, we need 1-P.  This is
                    # because the conditionalProbability supplies us
                    # with P(c=true)
                    if c.value:
                        P_true *= pt
                        P_false *= pf
                    else:
                        P_true *= (1.0-pt)
                        P_false *= (1.0-pf)

                # Normalise the output to get the probability this
                # node is true
                P = P_true / (P_true + P_false)

                # With probability P, set the value of the node to be
                # true
                n.value = True if random.random() <= P else False

            
        # Normalise the count to get the probability
        return float(counts[0]) / float(counts[0] + counts[1])

if __name__ == "__main__":
    b = BayesNet()

    prob = b.calculatePlayOutsideProbabilities(PLAYING_EXAMPLES)
    print "When it is raining, I play outside %.5f%% of the time." % (prob[0]*100)
    print "When it is not raining, I play outside %.5f%% of the time." % (prob[1]*100)

    print ""

    # Sample five state from joint distribution and print them
    for i in range(5):
        b.priorSample()
        b.printState()

    # Print out the results of experiments with various inference methods
    # Rejection Sampling
    print "\nRejection Sampling"
    print "P(rain | wet grass, ~playing outside) = %.5f" % b.rejectionSampling(3, [4, 5], [False, True], 10000)
    print "P(sprinklers | drought) = %.5f" % b.rejectionSampling(2, [1], [True], 100000)
    print "P(wet grass | rain, sprinklers) = %.5f" % b.rejectionSampling(5, [2, 3], [True, True], 10000000)

    # Likelihood Weighting
    print "\nLikelihood Weighting"
    print "P(rain | wet grass, ~playing outside) = %.5f" % b.likelihoodWeighting(3, [4, 5], [False, True], 10000)
    print "P(sprinklers | drought) = %.5f" % b.likelihoodWeighting(2, [1], [True], 100000)
    print "P(wet grass | rain, sprinklers) = %.5f" % b.likelihoodWeighting(5, [2, 3], [True, True], 10000000)

    # MCMC
    print "\nMCMC"
    print "P(rain | wet grass, ~playing outside) = %.5f" % b.MCMCask(3, [4, 5], [False, True], 10000)
    print "P(sprinklers | drought) = %.5f" % b.MCMCask(2, [1], [True], 100000)
    print "P(wet grass | rain, sprinklers) = %.5f" % b.MCMCask(5, [2, 3], [True, True], 10000000)

