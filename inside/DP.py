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

		start = initialV
		epsilon = np.inf
		iterId = 0
		while epsilon > tolerance and iterId < nIterations:
			iterId += 1
			V = np.max(self.R + self.discount * np.einsum('ijk,k->ij', self.T, start), axis=0)
			epsilon = np.max(np.abs(V - start))
			start = V
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

		start = initialPolicy
		iterId = 0
		while iterId < nIterations:
			iterId += 1
			V = self.evaluatePolicy_SolvingSystemOfLinearEqs(start)
			policy = self.extractPolicy(V)
			iterId += 1
		return [policy, V, iterId]


	def extractPolicy(self, V):
		'''Procedure to extract a policy from a value function
		pi <-- argmax_a R^a + gamma T^a V
		Inputs:
		V -- Value function: array of |S| entries
		Output:
		policy -- Policy: array of |S| entries'''
		#T^a is just the transition matrix for action a
		#R^a is just the reward matrix for action a
		#V is the value function
		#pi is the policy
		#pi(s) is the action that maximizes the value function for state s
		#pi(s) = argmax_a R^a + gamma T^a V
		# pi = np.zeros(self.nStates)
		# for s in range(self.nStates):
		# 	pi[s] = np.argmax(self.R[s] + self.discount * np.dot(self.T[s], V))
		# return pi
		policy = np.argmax(self.R + self.discount * np.einsum('ijk,k->ij', self.T, V), axis=0)
		
		return policy


	def evaluatePolicy_SolvingSystemOfLinearEqs(self, policy):
		'''Evaluate a policy by solving a system of linear equations
		V^pi = R^pi + gamma T^pi V^pi
		Input:
		policy -- Policy: array of |S| entries
		Ouput:
		V -- Value function: array of |S| entries'''
		R_pi = np.array([self.R[policy[s]][s] for s in range(self.nStates)])
		T_pi = np.array([self.T[policy[s]][s] for s in range(self.nStates)])
		V = np.linalg.solve(np.eye(self.nStates) - self.discount * T_pi, R_pi)


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

		start = initialPolicy
		V = initialV
		iterId = 0
		epsilon = np.inf
		while iterId < nIterations and epsilon > tolerance:
			iterId += 1
			V = self.evaluatePolicy_IterativeUpdate(start, V, nPolicyEvalIterations)
			policy = self.extractPolicy(V)
			epsilon = np.max(np.abs(V - start))
			start = V

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

		R_pi  = np.array([self.R[s, policy[s]] for s in range(self.nStates)])
		T_pi = np.array([self.T[s, policy[s]] for s in range(self.nStates)])
		start = initialV

		iterId = 0
		while iterId < nIterations:
			iterId += 1
			V = R_pi + self.discount * np.dot(T_pi, start)
			epsilon = np.max(np.abs(V - start))
			start = V



		return [V, iterId, epsilon]


if __name__ == '__main__':
	mdp = build_mazeMDP()
	dp = DynamicProgramming(mdp)
	# Test value iteration
	[policy, V, nIterations, epsilon] = dp.valueIteration(initialV=np.zeros(dp.nStates), tolerance=0.01)
	print_policy(policy)
	# Test policy iteration v1
	[policy, V, nIterations] = dp.policyIteration_v1(np.zeros(dp.nStates, dtype=int))
	print_policy(policy)
	# Test policy iteration v2
	[policy, V, nIterations, epsilon] = dp.policyIteration_v2(np.zeros(dp.nStates, dtype=int), np.zeros(dp.nStates), tolerance=0.01)
	print_policy(policy)