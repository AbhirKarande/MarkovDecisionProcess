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
		same = False
		while (not same) and iterId < nIterations:
			iterId += 1
			V = self.evaluatePolicy_SolvingSystemOfLinearEqs(start)
			policy = self.extractPolicy(V)
			iterId += 1
			same = np.array_equal(start,policy)
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

	def selectPolicy(self, T, policy):
		T_pi = np.copy(T[0])
		for s in range(self.nStates):
			T_pi[s] = T[policy[s]][s]
		return T_pi
	def policyT(self, policy):
		return self.selectPolicy(self.T, policy)

	def policyR(self, policy):
		return self.selectPolicy(self.R, policy)
	def evaluatePolicy_SolvingSystemOfLinearEqs(self, policy):
		'''Evaluate a policy by solving a system of linear equations
		V^pi = R^pi + gamma T^pi V^pi
		Input:
		policy -- Policy: array of |S| entries
		Ouput:
		V -- Value function: array of |S| entries'''
		
		#V^pi is the value function for policy pi
		#R^pi is the reward matrix for policy pi
		#T^pi is the transition matrix for policy pi
		#V^pi = R^pi + gamma T^pi V^pi
		T_pi = self.policyT(policy)
		R_pi = self.policyR(policy)
		I = np.eye(self.nStates)
		V = np.linalg.solve(I - self.discount * T_pi, R_pi)



		return V
	def actionValue(self, V):
		'''Compute the action-value function Q(s,a) = R^a + gamma T^a V
		Inputs:
		V -- Value function: array of |S| entries
		Output:
		Q -- Action-value function: array of |S|x|A| entries'''
		Q = np.zeros((self.nActions, self.nStates))
		for a in range(self.nActions):
			for s in range(self.nStates):
				Q[a,s] = (self.T[a,s]).dot(self.R[a,s] + V * self.discount)
		return Q
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

		policy = initialPolicy
		V = initialV
		iterId = 0
		epsilon = np.inf
		same = False
		while iterId < nIterations and epsilon > tolerance:
			iterId += 1
			V = self.evaluatePolicy_IterativeUpdate(policy, V, nPolicyEvalIterations, tolerance=tolerance)
			policy = self.extractPolicy(V)
			Q = self.actionValue(V)
			V2 = Q.max(axis=0)
			epsilon = (np.fabs(V2-V)).max()
		V = self.evaluatePolicy_IterativeUpdate(policy, V, nPolicyEvalIterations, tolerance=tolerance)
		return [policy, V, iterId, epsilon]

	def evaluatePolicy_IterativeUpdate(self, policy, initialV, nIterations=np.inf, tolerance=0.01):
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
		print('reached evaluatePolicy_IterativeUpdate')

		start = initialV
		epsilon = np.inf
		iterId = 0
		while iterId < nIterations and epsilon > tolerance:
			iterId += 1
			Q = self.actionValue(start)
			V = self.selectPolicy(Q, policy)
			
			epsilon = (np.fabs(V - start)).max()
			start = V


		return V


if __name__ == '__main__':
	mdp = build_mazeMDP()
	dp = DynamicProgramming(mdp)
	# Test value iteration
	[policy, V, nIterations, epsilon] = dp.valueIteration(initialV=np.zeros(dp.nStates), tolerance=0.01)
	print_policy(policy)
	print('Value function: {}'.format(V))
	print('# of iterations: {}'.format(nIterations))
	print('epsilon: {}'.format(epsilon))
	print('done with value iteration')
	# Test policy iteration v1
	# [policy, V, nIterations] = dp.policyIteration_v1(np.zeros(dp.nStates, dtype=int))
	# print_policy(policy)
	# print('Value function: {}'.format(V))
	# print('# of iterations: {}'.format(nIterations))

	# Test policy iteration v2
	[policy, V, nIterations, epsilon] = dp.policyIteration_v2(np.zeros(dp.nStates, dtype=int), np.zeros(dp.nStates), tolerance=0.01)
	print_policy(policy)
	print('Value function: {}'.format(V))
	print('# of iterations: {}'.format(nIterations))
	print('epsilon: {}'.format(epsilon))
	print('done with policy iteration')

	