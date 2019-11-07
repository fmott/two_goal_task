"""This module contains the Simulator class that defines interactions between 
the environment and the agent. It also keeps track of all generated observations
and responses. To initiate it one needs to provide the environment 
class and the agent class that will be used for the experiment.
"""
import torch
zeros = torch.zeros


class Simulator(object):
    
    def __init__(self, environment, agent, runs = 1, mini_blocks = 1, trials = 10):
        #set inital elements of the world to None        
        self.env = environment
        self.agent = agent
        
        self.runs = runs #number of paralel runs of the experiment (e.g. number of subjects)
        self.nb = mini_blocks # number of mini-blocks
        self.nt = trials # number of trials in each mini-block
        
        #container for agents responses
        self.responses = zeros(self.runs, self.nb, self.nt, dtype = torch.long)-1

    def simulate_experiment(self):
        """Simulates the experiment by iterating through all the mini-blocks and trials, 
           for each run in parallel. Here we generate responses and outcomes and update 
           agent's beliefs.
        """

        for b in range(self.nb):
            for t in range(self.nt):
                #update single trial
                states = self.env.states[:,b,t]
                offers = self.env.offers[:,b,t]
                
                self.agent.update_beliefs(b, t, states, offers)
                self.agent.planning(b,t)
                
                res = self.agent.sample_responses(b,t)
                self.env.update_environment(b, t+1, res)
                
                self.responses[:, b, t] = res

            
            



            

        

        
        