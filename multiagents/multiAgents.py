# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        FoodDist = [manhattanDistance( newPos, FoodPos ) for FoodPos in newFood.asList()]
        GostDist = [manhattanDistance( newPos, g.getPosition() ) for g in newGhostStates]
        nfd = 0
        if len(FoodDist) > 0:
            nfd = min(FoodDist)

        if len(currentGameState.getFood().asList()) > len(FoodDist):
            nfd = 0

        ngd = min(GostDist)
        if ngd <= 1:
            return -9999
        if newScaredTimes > 0:
            score = nfd + ngd
        else:
            score = nfd
        if action == Directions.STOP:
            score += 10

        return successorGameState.getScore() - score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        return self.MaxValue(gameState, self.index, self.depth)[1]

    def MaxValue(self, gameState, agentIndex, depth):
        """
            Returns the max value and action at a node.
        """
        # Checking if the node can be evaluated.
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), 'Stop'

        # agentIndex = 0
        actions = gameState.getLegalActions(agentIndex)
        successorGameStates = [gameState.generateSuccessor(agentIndex, action)
                               for action in actions]
        scores = [self.MinValue(successorGameState, agentIndex + 1, depth)[0]
                  for successorGameState in successorGameStates]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if
                       scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)

        return bestScore, actions[chosenIndex]


    def MinValue(self, gameState, agentIndex, depth):
        """
            Returns a min value and action at a node.
        """

        # Checking if the node can be evaluated.
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), 'Stop'

        actions = gameState.getLegalActions(agentIndex)
        successorGameStates = [gameState.generateSuccessor(agentIndex, action)
                               for action in actions]

        # An uparxei allo fantasma tote kalw thn min gia to epomeno fantasma
        #alliws kalw thn max gia ton pacman .
        if agentIndex < gameState.getNumAgents() - 1:
            scores = [self.MinValue(successorGameState, agentIndex + 1, depth)[0]
                      for successorGameState in successorGameStates]
        else:
            scores = [self.MaxValue(successorGameState, 0, depth - 1)[0]
                      for successorGameState in successorGameStates]

        bestScore = min(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)

        return bestScore, actions[chosenIndex]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        a = -999999
        b = 999999
        return self.maxFunction(gameState, self.index, self.depth, a, b)[1]
    def maxFunction(self, gameState, agentIndex, depth, a, b):
        """
            Returns a max value and action at a node.
        """

        # Checking if the node can be evaluated.
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), 'Stop'

        value = -999999
        bestValue = -999999
        bestAction = 'Stop'

        # agentIndex = 0
        actions = gameState.getLegalActions(agentIndex)

        for action in actions:
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            value = max(value, self.minFunction(successorGameState, agentIndex + 1, depth, a, b)[0])

            # Pruning condition.
            if value > b:
                return value, action
            a = max(a, value)

            if value > bestValue:
                bestValue = value
                bestAction = action

        return bestValue, bestAction

    def minFunction(self, gameState, agentIndex, depth, a, b):
        """
            Returns a min value and action at a node.
        """

        #elegxw na dw an o komvos mporei na ginei evaluate.
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), 'Stop'

        value = 999999
        bestValue = 999999
        bestAction = 'Stop'

        actions = gameState.getLegalActions(agentIndex)

        for action in actions:
            successorGameState = gameState.generateSuccessor(agentIndex, action)

            # an uparxei ki allo fantsma kalw thn min gia to epomeno
            # fantasma alliws kalw thn max gia ton pacman.
            if agentIndex < gameState.getNumAgents() - 1:
                value = min(value, self.minFunction(successorGameState, agentIndex + 1, depth, a, b)[0])
            else:
                value = min(value, self.maxFunction(successorGameState, 0, depth - 1, a, b)[0])

            # sun8hkh gia to kladema.
            if value < a:
                return value, action
            b = min(b, value)

            if bestValue > value:
                bestValue = value
                bestAction = action

        return bestValue, bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.MaxValue(gameState, self.index, self.depth)[1]

    def MaxValue(self, gameState, agentIndex, depth):
        """
          Returns a max value and action at a node.
        """

        # Checking if the node can be evaluated.
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), 'Stop'

        # agentIndex = 0
        actions = gameState.getLegalActions(agentIndex)
        successorGameStates = [gameState.generateSuccessor(agentIndex, action)
                               for action in actions]

        scores = [self.Expect(successorGameState, agentIndex + 1, depth)[0]
                  for successorGameState in successorGameStates]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)

        return bestScore, actions[chosenIndex]


    def Expect(self, gameState, agentIndex, depth):
        """
          Returns the expected value and action at a node.
        """

        # Checking if the node can be evaluated.
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), 'Stop'

        actions = gameState.getLegalActions(agentIndex)
        successorGameStates = [gameState.generateSuccessor(agentIndex, action)
                               for action in actions]

        # If there is another ghost, then call the expecti function for the
        # next ghost, otherwise call the max function for pacman.
        if agentIndex < gameState.getNumAgents() - 1:
            scores = [self.Expect(successorGameState, agentIndex + 1, depth)[0]
                      for successorGameState in successorGameStates]
        else:
            scores = [self.MaxValue(successorGameState, 0, depth - 1)[0]
                      for successorGameState in successorGameStates]

        # Getting the expected value.
        bestScore = sum(scores) / len(scores)

        return bestScore, 'Stop'


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    pacPos = currentGameState.getPacmanPosition()
    foodlist = currentGameState.getFood().asList()
    foodNum = currentGameState.getNumFood()
    ghostStates = currentGameState.getGhostStates()

    evaluation = 0

    # an ta fantasmata er8oun polu konta tote diatrexoume kinduno
    #epomenws edw den afhnoume ton pacman na pe8anei
    for ghostState in ghostStates:
        if ghostState.scaredTimer == 0:
            if manhattanDistance(pacPos, ghostState.getPosition()) <= 2:
                evaluation -= 10000000

    # oso pio konta sto fai toso kalutera
    if (foodNum > 0):
        foodsDistance = [manhattanDistance(pacPos, food) for food in foodlist]
        nearestFoodDistance = min(foodsDistance)
        evaluation -= nearestFoodDistance

    evaluation -= 1000 * foodNum
    evaluation -= 10 * len(currentGameState.getCapsules())

    return evaluation

# Abbreviation
better = betterEvaluationFunction
