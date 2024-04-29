# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections
from queue import PriorityQueue

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for _ in range(self.iterations):
            # Create a new dictionary to store the values for this iteration
            new_values = self.values.copy()

            # Loop over all states in the MDP
            for state in self.mdp.getStates():
                # Check if the current state is terminal
                if self.mdp.isTerminal(state):
                    # If it's a terminal state, its value is 0
                    new_values[state] = 0
                else:
                    # Otherwise, compute the Q-values for all possible actions
                    q_values = [self.computeQValueFromValues(state, action)
                                for action in self.mdp.getPossibleActions(state)]
                    # Update the value for the state with the maximum Q-value
                    new_values[state] = max(q_values) if q_values else 0

            # Update the values with the new values computed in this iteration
            self.values = new_values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # Initialize Q-value to zero
        # q_value = 0.0
        # # Iterate over each next state from the state-action pair
        # for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
        #     # Compute the reward for this transition
        #     reward = self.mdp.getReward(state, action, next_state)
        #     # Update the Q-value
        #     q_value += prob * (reward + self.discount * self.values[next_state])
        # return q_value
    
        return sum(prob * (self.mdp.getReward(state, action, next_state) + 
                           self.discount * self.values[next_state])
                   for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action))

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None

        legal_actions = self.mdp.getPossibleActions(state)
        if not legal_actions:
            return None

        # Initialize the best action and the highest Q-value seen so far
        best_action = None
        highest_q_value = float("-inf")
        
        # Iterate through all legal actions for the current state
        for action in legal_actions:
            # Compute the Q-value for the current action
            q_value = self.computeQValueFromValues(state, action)
            # If this Q-value is the highest, update the best action and highest Q-value
            if q_value > highest_q_value:
                highest_q_value = q_value
                best_action = action
        
        return best_action



    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        numStates = len(self.mdp.getStates())

        for i in range(self.iterations):
            index = i % numStates
            currState = self.mdp.getStates()[index]

            if self.mdp.isTerminal(currState):
                    # If it's a terminal state, its value is 0
                    self.values[currState] = 0
            else:
                # Otherwise, compute the Q-values for all possible actions
                q_values = [self.computeQValueFromValues(currState, action) for action in self.mdp.getPossibleActions(currState)]
                # Update the value for the state with the maximum Q-value
                self.values[currState] = max(q_values) if q_values else 0

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = self.computePredecessors()
        pq = util.PriorityQueue()

        # initialize priority queue
        for s in self.mdp.getStates():
            if self.mdp.isTerminal(s):
                continue

            q_values = [self.computeQValueFromValues(s, action) for action in self.mdp.getPossibleActions(s)]
            diff = abs(self.values[s] - max(q_values))
            pq.push(s, -diff)

        for _ in range(self.iterations):
            if pq.isEmpty():
                return
            
            s = pq.pop()

            if not self.mdp.isTerminal(s):
                q_values = [self.computeQValueFromValues(s, action) for action in self.mdp.getPossibleActions(s)]
                self.values[s] = max(q_values)

            for p in predecessors[s]:
                q_values = [self.computeQValueFromValues(p, action) for action in self.mdp.getPossibleActions(p)]
                diff = abs(self.values[p] - max(q_values))

                if diff > self.theta:
                    pq.update(p, -diff)

   
    def computePredecessors(self):
        predecessors = collections.defaultdict(set)

        for state in self.mdp.getStates():
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
                for nextState, _ in self.mdp.getTransitionStatesAndProbs(state, action):
                    predecessors[nextState].add(state)

        return predecessors