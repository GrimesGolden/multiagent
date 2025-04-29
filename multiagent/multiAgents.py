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
        # Heavily penalize stopping
        if action == Directions.STOP:
            return -float('inf')
        
        # Distance to closest food
        foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        closestFoodDist = min(foodDistances) if foodDistances else 1
        foodReward = 1.0 / closestFoodDist  # Closer food gives higher reward

        # Ghosts: avoid non-scared ghosts, and move towards scared ghosts
        ghostPenalty = 0 # A non scared ghost, has a scaredTime of 0
        # Combine ghost position and scared time into blocks of (state, time) as (ghost, scaredTime)
        for ghost, scaredTime in zip(newGhostStates, newScaredTimes):
          dist = manhattanDistance(newPos, ghost.getPosition())
          if scaredTime == 0 and dist < 2:  # If ghosts are not scared, avoid them
            ghostPenalty += 10
          elif scaredTime > 0:  # If ghosts are scared, go after them
            ghostPenalty -= 10

        # Combine all aspects and return the evaluation value
        return successorGameState.getScore() + foodReward - ghostPenalty

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

    def minvalue(self, gameState, agentIndex, depth):
        #minval
        #Initialize infinity
        v = float('inf')
        totalAgents = gameState.getNumAgents()
        # Get a list of all actions (for this state and current agent)
        actions = gameState.getLegalActions(agentIndex)
        
        for action in actions:
            # Get the state this action would generate
            # Alongside the agents and depth as appropriate for this next layer
            # Check the next agent
            nextAgent = (agentIndex + 1) % totalAgents
            # Is the next agent pacman, then decrement the depth (we just cycled a branch)
            if(nextAgent == 0):
                nextDepth = depth - 1
            else:
                #Else the depth is unchanged
                nextDepth = depth
            #Check this next states value (from the possible action) in context of the next nodes calls, we are traversing across the tree, until we hit the end (full agents) and then lowering depth.
            newState = gameState.generateSuccessor(agentIndex, action) 
            v = min(v, self.value(newState, nextAgent, nextDepth))
        return v 
    
    def maxvalue(self, gameState, agentIndex, depth):
        #maxval
        #Initialize infinity
        v = float('-inf')
        totalAgents = gameState.getNumAgents()
        # Get a list of all actions for this current state
        actions = gameState.getLegalActions(agentIndex)
        
        for action in actions:
            # Get next agent and depth level
            nextAgent = (agentIndex + 1) % totalAgents
            # Is the next agent pacman, then decrement the depth (we just cycled a branch)
            if(nextAgent == 0):
                nextDepth = depth - 1
            else:
                #Else the depth is unchanged
                nextDepth = depth
            #Check the state this move would create
            newState = gameState.generateSuccessor(agentIndex, action) #Current state, changed from nextAgent
            # Based on the value of this state in context of its successors. (It calls from base up)
            v = max(v, self.value(newState, nextAgent, nextDepth))
        return v  
            
    def value(self, gameState, agentIndex, depth):
        # Terminal case
        if(gameState.isWin() or gameState.isLose() or depth == 0):
            # Return the final value after recursion hit
            return self.evaluationFunction(gameState)
        # Simple controller logic
        totalAgents = gameState.getNumAgents() 

        if(agentIndex > 0):
            #Min
            return self.minvalue(gameState, agentIndex, depth)
        if(agentIndex == 0):
            #Max
            return self.maxvalue(gameState, agentIndex, depth)
            
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
        # Check terminal case
        if(gameState.isWin() or gameState.isLose() or self.depth == 0):
            return self.evaluationFunction(gameState)
      
        # Get the total number of agents 
        totalAgents = gameState.getNumAgents() 
        # Starting out, we are pacman, index(0)
        agentIndex = self.index
        # Get depth
        depth = self.depth
        # Get a list of all actions
        actions = gameState.getLegalActions(agentIndex)
        # Starting a negative infinity to find true max...
        max_val = float('-inf')
        bestAction = actions[0] 
        # Analyse all possible actions
        # And for each action
        for action in actions:
            # Get the state this action would generate
            nextState = gameState.generateSuccessor(agentIndex, action)
            # Then locate the next agent in the module i.e 0..1..2..0..1..2(for total agents 3)
            nextAgent = (0 + 1) % totalAgents   # → ghost #1 
            # If the next agent is pacman again, then decrease our dept.
            if(nextAgent == 0):
                nextDepth = depth - 1
            else:
                nextDepth = depth 
          # Get value of the state, with it's appropriate agent and depth level. I.E what state this action would generate, in context of the value of the ghosts moves too
            v = self.value(nextState, nextAgent, nextDepth)
            if(v > max_val):
                max_val = v
                bestAction = action
        
        return bestAction
 

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def max_value(state, depth, agentIndex, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            
            v = -float('inf')
            actions = state.getLegalActions(agentIndex)
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                # Only increment depth after all agents have moved (agentIndex wraps around)
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                nextDepth = depth + 1 if nextAgent == 0 else depth
                v = max(v, value(successor, nextDepth, nextAgent, alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        def min_value(state, depth, agentIndex, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            
            v = float('inf')
            actions = state.getLegalActions(agentIndex)
            for action in actions:
                successor = state.generateSuccessor(agentIndex, action)
                # Only increment depth after all agents have moved (agentIndex wraps around)
                nextAgent = (agentIndex + 1) % state.getNumAgents()
                nextDepth = depth + 1 if nextAgent == 0 else depth
                v = min(v, value(successor, nextDepth, nextAgent, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        def value(state, depth, agentIndex, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            
            if agentIndex == 0:  # Pacman's turn - maximize
                return max_value(state, depth, agentIndex, alpha, beta)
            else:  # Ghost's turn - minimize
                return min_value(state, depth, agentIndex, alpha, beta)

        # Initial call for Pacman (agent 0) at depth 0
        best_score = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        best_action = Directions.STOP
        
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = value(successor, 0, 1, alpha, beta)  # Next agent is 1 (first ghost)
            if score > best_score:
                best_score = score
                best_action = action
            alpha = max(alpha, best_score)
            
        return best_action
    
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def expvalue(self, gameState, agentIndex, depth):
        #minval
        #Initialize zero
        v = 0
        totalAgents = gameState.getNumAgents()
        # Get a list of all actions (for this state and current agent)
        actions = gameState.getLegalActions(agentIndex)
        
        for action in actions:
            # Get the state this action would generate
            # Alongside the agents and depth as appropriate for this next layer
            # Check the next agent
            nextAgent = (agentIndex + 1) % totalAgents
            # Is the next agent pacman, then decrement the depth (we just cycled a branch)
            if(nextAgent == 0):
                nextDepth = depth - 1
            else:
                #Else the depth is unchanged
                nextDepth = depth
            #Check this next states value (from the possible action) in context of the next nodes calls, we are traversing across the tree, until we hit the end (full agents) and then lowering depth.
            newState = gameState.generateSuccessor(agentIndex, action) 
            # Recursively get the value
            p = 1 /len(actions)
            v += p*(self.value(newState, nextAgent, nextDepth))
        return v 
    
    def maxvalue(self, gameState, agentIndex, depth):
        #maxval
        #Initialize infinity
        v = float('-inf')
        totalAgents = gameState.getNumAgents()
        # Get a list of all actions for this current state
        actions = gameState.getLegalActions(agentIndex)
        
        for action in actions:
            # Get next agent and depth level
            nextAgent = (agentIndex + 1) % totalAgents
            # Is the next agent pacman, then decrement the depth (we just cycled a branch)
            if(nextAgent == 0):
                nextDepth = depth - 1
            else:
                #Else the depth is unchanged
                nextDepth = depth
            #Check the state this move would create
            newState = gameState.generateSuccessor(agentIndex, action) #Current state, changed from nextAgent
            # Based on the value of this state in context of its successors. (It calls from base up)
            v = max(v, self.value(newState, nextAgent, nextDepth))
        return v  

    def value(self, gameState, agentIndex, depth):
        # Terminal case
        if(gameState.isWin() or gameState.isLose() or depth == 0):
            # Return the final value after recursion hit
            return self.evaluationFunction(gameState)
        # Simple controller logic
        totalAgents = gameState.getNumAgents() 

        if(agentIndex > 0):
            #Return exp value
            return self.expvalue(gameState, agentIndex, depth)
        if(agentIndex == 0):
            #Max
            return self.maxvalue(gameState, agentIndex, depth)
        
    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
         # Check terminal case
        if(gameState.isWin() or gameState.isLose() or self.depth == 0):
            return self.evaluationFunction(gameState)
      
        # Get the total number of agents 
        totalAgents = gameState.getNumAgents() 
        # Starting out, we are pacman, index(0)
        agentIndex = self.index
        # Get depth
        depth = self.depth
        # Get a list of all actions
        actions = gameState.getLegalActions(agentIndex)
        # Starting a negative infinity to find true max...
        max_val = float('-inf')
        bestAction = actions[0] 
        # Analyse all possible actions
        # And for each action
        for action in actions:
            # Get the state this action would generate
            nextState = gameState.generateSuccessor(agentIndex, action)
            # Then locate the next agent in the module i.e 0..1..2..0..1..2(for total agents 3)
            nextAgent = (0 + 1) % totalAgents   # → ghost #1 
            # If the next agent is pacman again, then decrease our dept.
            if(nextAgent == 0):
                nextDepth = depth - 1
            else:
                nextDepth = depth 
          # Get value of the state, with it's appropriate agent and depth level. I.E what state this action would generate, in context of the value of the ghosts moves too
            v = self.value(nextState, nextAgent, nextDepth)
            if(v > max_val):
                max_val = v
                bestAction = action
        
        return bestAction
 
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction