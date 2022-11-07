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
	def extractPolicy(self, V):
		policy = np.argmax(self.R + self.discount * np.einsum('ijk,k->ij', self.T, V), axis=0)		
		return policy
	
	
	def policyIteration_v1(self, initialPolicy, nIterations=np.inf, tolerance=0.01):
		policy = initialPolicy
		V = np.zeros(self.nStates)
		iterId = 0
		same = False
		while (not same) and iterId < nIterations:
			iterId += 1
			oldPolicy = np.copy(policy)
			V = self.evaluatePolicy_SolvingSystemOfLinearEqs(policy)
			policy = self.extractPolicy(V)
			iterId += 1
			same = np.array_equal(oldPolicy,policy)
		return [policy, V, iterId]
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
		T_pi = self.policyT(policy)
		R_pi = self.policyR(policy)
		I = np.eye(self.nStates)
		V = np.linalg.inv(I - T_pi * self.discount).dot(R_pi)
		return V
	
	
	
	def actionValue(self, V):
		Q = np.zeros((self.nActions, self.nStates))
		for a in range(self.nActions):
			for s in range(self.nStates):
				Q[a,s] = (self.T[a,s]).dot(self.R[a,s] + V * self.discount)
		return Q
	def policyIteration_v2(self, initialPolicy, initialV, nPolicyEvalIterations=10, nIterations=np.inf, tolerance=0.01):
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
		V = initialV
		epsilon = np.inf
		iterId = 0
		while (iterId < nIterations and epsilon > tolerance):
			V_start = np.copy(V)
			iterId += 1
			Q = self.actionValue(V)
			V = self.selectPolicy(Q, policy)
			
			epsilon = (np.fabs(V - V_start)).max()
		return V


if __name__ == '__main__':
	mdp = build_mazeMDP()
	dp = DynamicProgramming(mdp)
	# Test value iteration
	# [policy, V, nIterations, epsilon] = dp.valueIteration(initialV=np.zeros(dp.nStates), tolerance=0.01)
	# print_policy(policy)
	# print('Value function: {}'.format(V))
	# print('# of iterations: {}'.format(nIterations))
	# print('epsilon: {}'.format(epsilon))
	# print('done with value iteration')
	# # Test policy iteration v1
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

	