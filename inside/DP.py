from MDP import build_mazeMDP, print_policy
import numpy as np

class DynamicProgramming:
	def __init__(self, MDP):
		self.R = MDP.R
		self.T = MDP.T
		self.discount = MDP.discount
		self.nStates = MDP.nStates
		self.nActions = MDP.nActions


	def valueIteration(self, initialV, nIterations=np.inf, tolerance=0.01):
		'''Value iteration procedure
		V <-- max_a R^a + gamma T^a V

		Inputs:
		initialV -- Initial value function: array of |S| entries
		nIterations -- limit on the # of iterations: scalar (default: infinity)
		tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

		Outputs:
		policy -- Policy: array of |S| entries
		V -- Value function: array of |S| entries
		iterId -- # of iterations performed: scalar
		epsilon -- ||V^n-V^n+1||_inf: scalar'''

		#compute value iteration where V <-- max_a R^a + gamma T^a V where the input is the initial value function of |S| entries and the output is the value function of |S| entries
		#iterate until convergence or tolerance is reached
		V = initialV
		for i in range(nIterations):
			V = np.max(self.R + self.discount * np.dot(self.T, V), axis=1)
			iterId = i
			#compute epsilon through ||V^n-V^n+1||_inf where V^n is value function at iteration n and V^n+1 is value function at iteration n+1

		policy = self.extractPolicy(V)
		return [policy, V, iterId, epsilon]

	def policyIteration_v1(self, initialPolicy, nIterations=np.inf, tolerance=0.01):
		'''Policy iteration procedure: alternate between policy
		evaluation (solve V^pi = R^pi + gamma T^pi V^pi) and policy
		improvement (pi <-- argmax_a R^a + gamma T^a V^pi).

		Inputs:
		initialPolicy -- Initial policy: array of |S| entries
		nIterations -- limit on # of iterations: scalar (default: inf)
		tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

		Outputs:
		policy -- Policy: array of |S| entries
		V -- Value function: array of |S| entries
		iterId -- # of iterations peformed by modified policy iteration: scalar'''
		
		#alternate between policy evaluation and policy imporvement
		policy = initialPolicy
		for i in range(nIterations):
			V = self.evaluatePolicy_SolvingSystemOfLinearEqs(policy)
			policy = self.extractPolicy(V)
			iterId = i

		return [policy, V, iterId]


	def extractPolicy(self, V):
		'''Procedure to extract a policy from a value function
		pi <-- argmax_a R^a + gamma T^a V

		Inputs:
		V -- Value function: array of |S| entries

		Output:
		policy -- Policy: array of |S| entries'''

		#extract policy from value function pi <-- argmax_a R^a + gamma T^a V where the input is a value function of |S| entries and the output is a policy of |S| entries
		policy = np.argmax(self.R + self.discount * np.dot(self.T, V), axis=1)
		return policy


	def evaluatePolicy_SolvingSystemOfLinearEqs(self, policy):
		'''Evaluate a policy by solving a system of linear equations
		V^pi = R^pi + gamma T^pi V^pi

		Input:
		policy -- Policy: array of |S| entries

		Ouput:
		V -- Value function: array of |S| entries'''

		#evaluate a policy by solvinga  system of linear equations V^pi = R^pi + gamma T^pi V^pi where the input is a policy of |S| entries and the output is the value function, which is an array of |S| entries
		V = np.linalg.solve(np.eye(self.nStates) - self.discount * np.dot(self.T, policy), self.R)
		return V

	def policyIteration_v2(self, initialPolicy, initialV, nPolicyEvalIterations=5, nIterations=np.inf, tolerance=0.01):
		'''Modified policy iteration procedure: alternate between
		partial policy evaluation (repeat a few times V^pi <-- R^pi + gamma T^pi V^pi)
		and policy improvement (pi <-- argmax_a R^a + gamma T^a V^pi)

		Inputs:
		initialPolicy -- Initial policy: array of |S| entries
		initialV -- Initial value function: array of |S| entries
		nPolicyEvalIterations -- limit on # of iterations to be performed in each partial policy evaluation: scalar (default: 5)
		nIterations -- limit on # of iterations to be performed in modified policy iteration: scalar (default: inf)
		tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

		Outputs:
		policy -- Policy: array of |S| entries
		V -- Value function: array of |S| entries
		iterId -- # of iterations peformed by modified policy iteration: scalar
		epsilon -- ||V^n-V^n+1||_inf: scalar'''

		V = initialV
		policy = initialPolicy
		for i in range(nIterations):
			V = self.evaluatePolicy_IterativeUpdate(policy, V, nPolicyEvalIterations)
			policy = self.extractPolicy(V)
			iterId = i
			epsilon = np.linalg.norm(V - initialV, np.inf)

		return [policy, V, iterId, epsilon]

	def evaluatePolicy_IterativeUpdate(self, policy, initialV, nIterations=np.inf):
		'''Partial policy evaluation:
		Repeat V^pi <-- R^pi + gamma T^pi V^pi

		Inputs:
		policy -- Policy: array of |S| entries
		initialV -- Initial value function: array of |S| entries
		nIterations -- limit on the # of iterations: scalar (default: infinity)

		Outputs:
		V -- Value function: array of |S| entries
		iterId -- # of iterations performed: scalar
		epsilon -- ||V^n-V^n+1||_inf: scalar'''

		#evaluate a policy by iterative update V^pi <-- R^pi + gamma T^pi V^pi where the input is a policy of |S| entries and the output is the value function, which is an array of |S| entries
		V = initialV
		for i in range(nIterations):
			V = self.R + self.discount * np.dot(self.T, V)
			iterId = i
			epsilon = np.linalg.norm(V - initialV, np.inf)
		return [V, iterId, epsilon]

if __name__ == '__main__':
	mdp = build_mazeMDP()
	dp = DynamicProgramming(mdp)
	# Test value iteration
	[policy, V, nIterations, epsilon] = dp.valueIteration(initialV=np.zeros(dp.nStates), nIterations=1000, tolerance=0.01)
	print_policy(policy)
	# Test policy iteration v1
	[policy, V, nIterations] = dp.policyIteration_v1(np.zeros(dp.nStates, nIterations=1000, dtype=int))
	print_policy(policy)
	# Test policy iteration v2
	[policy, V, nIterations, epsilon] = dp.policyIteration_v2(np.zeros(dp.nStates, dtype=int), np.zeros(dp.nStates), nIterations=1000, tolerance=0.01)
	print_policy(policy)