# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

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
    newFoodPos = newFood.asList()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    currentFood = currentGameState.getFood()
    currentFoodPos = currentFood.asList()
	
    score = 0
    if (len(newFoodPos) == 0) :
    	return 1000
    
    if (len(newFoodPos) < len(currentFoodPos)) :
    	score += 200
    foodDist = [manhattanDistance(newPos, food) for food in newFoodPos]
    closestFood = 1.0/min(foodDist)
    
    ghostPos = [ghost.getPosition() for ghost in newGhostStates]
    
    ghostDist = [manhattanDistance(newPos, g) for g in ghostPos]
    closestGhostDist = min(ghostDist)
    if (closestGhostDist < 2) :
    	return -1000000
    
    
    
    
    score += ((closestFood) + (1.0/closestGhostDist) - 1.0/len(newFoodPos))
    
    
    return score

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
  	action = gameState.getLegalActions(0)
  	v = float("-inf")
	for a in action:
		val = self.valueFunction(gameState.generateSuccessor(0, a), 1, self.depth)
		v = max(val, v)
		if (v == val):
			bestaction = a
	return bestaction
	
  
  def valueFunction(self, currGameState, agentIndex, depth):
  	if (agentIndex > currGameState.getNumAgents() - 1):
  		agentIndex = 0
  		depth -= 1
  	if (currGameState.isWin() or currGameState.isLose() or depth == 0):
  		return (self.evaluationFunction(currGameState))
  	if (agentIndex == 0):
  		return self.maxValue(currGameState, agentIndex, depth)
  	else:
  		return self.minValue(currGameState, agentIndex, depth)
  		
  def maxValue(self, currGameState, agentIndex, depth):
    v = float("-inf")
    legalActions = currGameState.getLegalActions(0)
    for action in legalActions:
    	nextGameState = currGameState.generateSuccessor(0, action)
    	v = max(v, self.valueFunction(nextGameState, agentIndex + 1, depth))
    return v
    
  def minValue(self, currGameState, agentIndex, depth):
  	v = float("inf")
  	legalActions = currGameState.getLegalActions(agentIndex)
  	for action in legalActions:
  		nextGameState = currGameState.generateSuccessor(agentIndex, action)
  		v = min(v, self.valueFunction(nextGameState, agentIndex + 1, depth))
  	return v
    	

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    action = gameState.getLegalActions(0)
    v = float("-inf")
    alpha = float("-inf")
    beta = float("inf")
    for a in action:
    	if a == "Stop":
    		continue
    	val = self.valueFunction(gameState.generateSuccessor(0, a), 1, self.depth, alpha, beta)
    	if (val > alpha):
    		alpha = val
    	if (val > v):
    		bestaction = a
    		v = val
    return bestaction
	
  
  def valueFunction(self, currGameState, agentIndex, depth, alpha, beta):
  	if (agentIndex > currGameState.getNumAgents() - 1):
  		agentIndex = 0
  		depth -= 1
  	if (currGameState.isWin() or currGameState.isLose() or depth == 0):
  		v = self.evaluationFunction(currGameState)
  		return v
  	if (agentIndex == 0):
  		v = self.maxValue(currGameState, agentIndex, depth, alpha, beta)
  		return v
  	else:
  		v = self.minValue(currGameState, agentIndex, depth, alpha, beta)
  		return v
  		
  def maxValue(self, currGameState, agentIndex, depth, alpha, beta):
  	v = float("-inf")
  	legalActions = currGameState.getLegalActions(0)
  	if (currGameState.isWin() or currGameState.isLose() or depth == 0):
  		v = self.evaluationFunction(currGameState)
  		return v
  	for action in legalActions:
  		nextGameState = currGameState.generateSuccessor(0, action)
  		v = max(v, self.valueFunction(nextGameState, agentIndex + 1, depth, alpha, beta))
  		if v >= beta:
  			return v
  		alpha = max(alpha, v)
  	return v
    
  def minValue(self, currGameState, agentIndex, depth, alpha, beta):
  	v = float("inf")
  	legalActions = currGameState.getLegalActions(agentIndex)
  	if (currGameState.isWin() or currGameState.isLose() or depth == 0):
  		v = self.evaluationFunction(currGameState)
  		return v
  	for action in legalActions:
  		nextGameState = currGameState.generateSuccessor(agentIndex, action)
  		v = min(v, self.valueFunction(nextGameState, agentIndex + 1, depth, alpha, beta))
  		if v <= alpha:
  			return v
  		beta = min(beta, v)
  	return v

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
    action = gameState.getLegalActions(0)
    v = float("-inf")
    for a in action:
    	val = self.valueFunction(gameState.generateSuccessor(0, a), 1, self.depth)
    	if (val > v):
    		v = val
    		bestaction = a
    return bestaction
	
  
  def valueFunction(self, currGameState, agentIndex, depth):
  	if (agentIndex > currGameState.getNumAgents() - 1):
  		agentIndex = 0
  		depth -= 1
  	if (currGameState.isWin() or currGameState.isLose() or depth == 0):
  		return float((self.evaluationFunction(currGameState)))
  	if (agentIndex == 0):
  		return self.maxValue(currGameState, agentIndex + 1, depth)
  	else:
  		return self.expValue(currGameState, agentIndex + 1, depth)
  		
  def maxValue(self, currGameState, agentIndex, depth):
    v = float("-inf")
    legalActions = currGameState.getLegalActions(0)
    for action in legalActions:
    	nextGameState = currGameState.generateSuccessor(0, action)
    	v = max(v, self.valueFunction(nextGameState, agentIndex, depth))
    return v
    
  def expValue(self, currGameState, agentIndex, depth):
  	v = 0
  	currAgent = agentIndex - 1
  	legalActions = currGameState.getLegalActions(currAgent)
  	for action in legalActions:
  		nextGameState = currGameState.generateSuccessor(currAgent, action)
  		p = (1.0 / len(legalActions))
  		v += p*self.valueFunction(nextGameState, agentIndex, depth)
  	return v

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    I check if the state I am evaluating is a win or a loss, 
    and return a very high value or a very low value (respectively)
    depending on the situation.
    
    Then I calculate the manhattan distance to the closest food and reciprocate the value.
    
    Next, I add that value of the closest food to the current score of the game.
  """
  "*** YOUR CODE HERE ***"
  score = 0
  if currentGameState.isWin() :
  	return 10000000
  if currentGameState.isLose() :
  	return -10000000
  
  position = currentGameState.getPacmanPosition()
  foodList = currentGameState.getFood()
  foodPosition = foodList.asList()
  foodDist = [manhattanDistance(position, food) for food in foodPosition]
  closestFood = 1.0/min(foodDist)
  
  score += (currentGameState.getScore() + closestFood)
  return score
 
# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

