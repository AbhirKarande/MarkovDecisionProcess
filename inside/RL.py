from MDP import build_mazeMDP, print_policy
import numpy as np
from matplotlib import pyplot as plt

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
		#perform q learning
		Q = np.zeros((self.mdp.nActions, self.mdp.nStates))
		policy = np.zeros((self.mdp.nStates))
		for i in range(nEpisodes):
			state = np.random.randint(0, self.mdp.nStates)
			while state != self.mdp.nStates - 1:
				#choose action
				if np.random.rand() < epsilon:
					action = np.random.randint(0, self.mdp.nActions)
				else:
					action = np.argmax(Q[:, state])
				#sample reward and next state
				[reward, nextState] = self.sampleRewardAndNextState(state, action)
				#update Q
				Q[action, state] = Q[action, state] + 0.1 * (reward + self.mdp.discount * np.max(Q[:, nextState]) - Q[action, state])
				#update state
				state = nextState
		#update policy
		policy = np.argmax(Q, axis=0)
		#produce a figure where the x-axis indicates the number of episodes and the y-axis indicates the cumulative rewards per episode
		cumReward = np.cumsum(cumReward)
		#x axis is number of episodes
		#y axis is cumulative rewards per episode
		plt.plot(cumReward)
		plt.xlabel('Number of Episodes')
		plt.ylabel('Cumulative Rewards per Episode')
		plt.show()
		
		
		return [Q,policy]
	def generateEpisode(self, epsilon):
		'''
		Generate an episode using epsilon-soft behavior policy
		Inputs:
		epsilon -- probability with which an action is chosen at random
		Outputs:
		episode -- list of tuples (s,a,r) where s is the state, a is the action, and r is the reward
		'''
		episode = []
		state = np.random.randint(0, self.mdp.nStates)
		while state != self.mdp.nStates - 1:
			#choose action
			if np.random.rand() < epsilon:
				action = np.random.randint(0, self.mdp.nActions)
			else:
				action = np.argmax(Q[:, state])
			#sample reward and next state
			[reward, nextState] = self.sampleRewardAndNextState(state, action)
			#update episode
			episode.append((state, action, reward))
			#update state
			state = nextState
		return episode
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
		#perform MC
		Q = np.zeros((self.mdp.nActions, self.mdp.nStates))
		policy = np.zeros((self.mdp.nStates))
		for i in range(nEpisodes):
			#generate episode
			episode = self.generateEpisode(epsilon)
			#calculate returns
			G = 0
			for j in range(len(episode) - 1, -1, -1):
				G = self.mdp.discount * G + episode[j][2]
				#update Q
				Q[episode[j][1], episode[j][0]] = Q[episode[j][1], episode[j][0]] + 0.1 * (G - Q[episode[j][1], episode[j][0]])
		#update policy
		policy = np.argmax(Q, axis=0)
		

		return [Q,policy]

if __name__ == '__main__':
	mdp = build_mazeMDP()
	rl = ReinforcementLearning(mdp, np.random.normal)

	# Test Q-learning
	[Q, policy] = rl.OffPolicyTD(nEpisodes=500, epsilon=0.1)
	print_policy(policy)
	print(Q)
	
	

	

	
	
	
	# Test Off-Policy MC
	[Q, policy] = rl.OffPolicyMC(nEpisodes=500, epsilon=0.1)
	print_policy(policy)
	print(Q)