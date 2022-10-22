from MDP import build_mazeMDP, print_policy
import numpy as np

class ReinforcementLearning:
	def __init__(self, mdp, sampleReward):
		"""
		Constructor for the RL class

		:param mdp: Markov decision process (T, R, discount)
		:param sampleReward: Function to sample rewards (e.g., bernoulli, Gaussian). This function takes one argument:
		the mean of the distribution and returns a sample from the distribution.
		"""

		self.mdp = mdp
		self.sampleReward = sampleReward

	def sampleRewardAndNextState(self,state,action):
		'''Procedure to sample a reward and the next state
		reward ~ Pr(r)
		nextState ~ Pr(s'|s,a)

		Inputs:
		state -- current state
		action -- action to be executed

		Outputs:
		reward -- sampled reward
		nextState -- sampled next state
		'''

		reward = self.sampleReward(self.mdp.R[action,state])
		cumProb = np.cumsum(self.mdp.T[action,state,:])
		nextState = np.where(cumProb >= np.random.rand(1))[0][0]
		return [reward,nextState]

	def OffPolicyTD(self, nEpisodes, epsilon=0.0):
		'''
		Off-policy TD (Q-learning) algorithm
		Inputs:
		nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
		epsilon -- probability with which an action is chosen at random
		Outputs:
		Q -- final Q function (|A|x|S| array)
		policy -- final policy
		'''

		# temporary values to ensure that the code compiles until this
		# function is coded
		Q = np.zeros([self.mdp.nActions,self.mdp.nStates])
		policy = np.zeros(self.mdp.nStates,int)

		return [Q,policy]

	def OffPolicyMC(self, nEpisodes, epsilon=0.0):
		'''
		Off-policy MC algorithm with epsilon-soft behavior policy
		Inputs:
		nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
		epsilon -- probability with which an action is chosen at random
		Outputs:
		Q -- final Q function (|A|x|S| array)
		policy -- final policy
		'''

		# temporary values to ensure that the code compiles until this
		# function is coded
		Q = np.zeros([self.mdp.nActions,self.mdp.nStates])
		policy = np.zeros(self.mdp.nStates,int)

		return [Q,policy]

if __name__ == '__main__':
	mdp = build_mazeMDP()
	rl = ReinforcementLearning(mdp, np.random.normal)

	# Test Q-learning
	[Q, policy] = rl.OffPolicyTD(nEpisodes=500, epsilon=0.1)
	print_policy(policy)

	# Test Off-Policy MC
	[Q, policy] = rl.OffPolicyMC(nEpisodes=500, epsilon=0.1)
	print_policy(policy)
