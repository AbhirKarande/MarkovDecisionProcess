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

		#leverage sampleRewardAndNextState to sample a reward for the next state in MDP
		V = initialV
		for i in range(nIterations):
			#keep the size of the value function the same as the size of initialV
			V = np.max(self.R + self.discount * np.dot(self.T, V), axis=0) #axis=0 means we are taking the max across the actions

			iterId = i
			epsilon = np.linalg.norm(V - initialV, np.inf)
			print(len(V),len(initialV)) #print the length of the value function and the initial value function to make sure they are the same size
		policy = self.extractPolicy(V)
		print(policy)
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

		#the policy is the max of Reward + discount * Transition * Value
		policy = np.argmax(self.R + self.discount * np.dot(self.T, V), axis=0) #axis=0 means we are taking the max across the actions
		return policy


	def evaluatePolicy_SolvingSystemOfLinearEqs(self, policy):
		'''Evaluate a policy by solving a system of linear equations
		V^pi = R^pi + gamma T^pi V^pi

		Input:
		policy -- Policy: array of |S| entries

		Ouput:
		V -- Value function: array of |S| entries'''

		#evaluate a policy by solving system of linear equations
		# V^pi = R^pi + gamma T^pi V^pi where the input is a policy of |S| entries and the output is the value function, which is an array of |S| entries
		# V^pi = R^pi + gamma T^pi V^pi
		# V^pi - gamma T^pi V^pi = R^pi
		# (I - gamma T^pi)V^pi = R^pi
		# V^pi = (I - gamma T^pi)^-1 R^pi

		#construct the matrix I - gamma T^pi
		I = np.identity(self.nStates) #identity matrix of size |S| x |S| where |S| is the number of states
		T_pi = self.T[policy, range(self.nStates), :] #T^pi is the transition matrix for the policy
		matrix = I - self.discount * T_pi #I - gamma T^pi
		#matrix should be of shape (4,17) not (4,17,17) so we need to reshape it
		#construct the vector R^pi
		R_pi = self.R[policy, range(self.nStates)] #R^pi is the reward matrix for the policy

		#solve the system of linear equations
		V = np.linalg.solve(matrix, R_pi) #V^pi = (I - gamma T^pi)^-1 R^pi
	
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
			V = self.R[policy, range(self.nStates)] + self.discount * np.dot(self.T[policy, range(self.nStates), :], V)
			iterId = i
			epsilon = np.linalg.norm(V - initialV, np.inf)
			initialV = V
		return [V, iterId, epsilon]

if __name__ == '__main__':
	mdp = build_mazeMDP()
	dp = DynamicProgramming(mdp)
	# Test value iteration
	[policy, V, nIterations, epsilon] = dp.valueIteration(initialV=np.zeros(dp.nStates), nIterations=1000, tolerance=0.01)
	print_policy(policy)
	# Test policy iteration v1
	[policy, V, nIterations] = dp.policyIteration_v1(np.zeros(dp.nStates, dtype=int), nIterations=1000)
	print_policy(policy)
	# Test policy iteration v2
	[policy, V, nIterations, epsilon] = dp.policyIteration_v2(np.zeros(dp.nStates, dtype=int), np.zeros(dp.nStates), nIterations=1000, tolerance=0.01)
	print_policy(policy)