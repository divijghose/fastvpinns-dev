"""
This module contains the scheduling algorithm for sub-domains.

Author: Thivin Anandh, Divij Ghose
Revision History:
- First implementation: 16/03/2024


"""
import numpy as np
from fastvpinns.domain_decomposition.domain_decomposition import *

class Scheduling():
    """
    This class will be responsible for: 
    1. Assigning the blocks to the workers.
    2. Scheduling the blocks to the workers.
    """

    def __init__(self, domain, decomposed_domain, scheduling_type="all"):

        self.domain = domain
        self.decomposed_domain = decomposed_domain
        self.scheduling_type = scheduling_type

        if self.scheduling_type not in ['left_to_right', 'bottom_to_top', 'top_to_bottom', 'right_to_left', 'all']:
            print("[ERROR] Invalid scheduling type at File: scheduling.py")
            raise ValueError("Invalid scheduling type, The scheduling type should be one of the following: 'left_to_right', 'bottom_to_top', 'top_to_bottom', 'right_to_left', 'all'")
        

        self.x_mean, self.y_mean = self.domain.calculate_subdomain_means()

        self.blocks_in_domain = self.decomposed_domain.blocks_in_domain
        self.scheduler_list = self.group_subdomains(self.x_mean, self.y_mean, scheduling_type)        
        

    def group_subdomains(self, x_mean, y_mean, scheduling_type):
        """
        This function groups the subdomains based on the x_mean values.

        Parameters:
        x_mean (np.ndarray): The mean x values of the subdomains.

        Returns:
        scheduler_list (dict): A dictionary containing the subdomains grouped based on the x_mean values.
        """
        grouped_subdomains = {}

        if(scheduling_type == 'left_to_right'):
            grouped_subdomains = {}
            for i, value in enumerate(x_mean):
                if value not in grouped_subdomains:
                    grouped_subdomains[value] = [i]
                else:
                    grouped_subdomains[value].append(i)
            scheduler_list = {i + 1: indices for i, indices in enumerate(grouped_subdomains.values())}

        elif(scheduling_type == 'bottom_to_top'):
            grouped_subdomains = {}
            for i, value in enumerate(y_mean):
                if value not in grouped_subdomains:
                    grouped_subdomains[value] = [i]
                else:
                    grouped_subdomains[value].append(i)
            #Sort the dictionary ascending order based on keys
            grouped_subdomains = dict(sorted(grouped_subdomains.items()))
            scheduler_list = {i + 1: indices for i, indices in enumerate(grouped_subdomains.values())}
            
        elif(scheduling_type == 'top_to_bottom'):
            grouped_subdomains = {}
            for i, value in enumerate(y_mean):
                if value not in grouped_subdomains:
                    grouped_subdomains[value] = [i]
                else:
                    grouped_subdomains[value].append(i)
            #Sort the dictionary descending order based on keys
            grouped_subdomains = dict(sorted(grouped_subdomains.items(), reverse=True))
            scheduler_list = {i + 1: indices for i, indices in enumerate(grouped_subdomains.values())}

        elif(scheduling_type == 'right_to_left'):
            grouped_subdomains = {}
            for i, value in enumerate(x_mean):
                if value not in grouped_subdomains:
                    grouped_subdomains[value] = [i]
                else:
                    grouped_subdomains[value].append(i)
            #Sort the dictionary descending order based on keys
            grouped_subdomains = dict(sorted(grouped_subdomains.items(), reverse=True))
            scheduler_list = {i + 1: indices for i, indices in enumerate(grouped_subdomains.values())}

        elif(scheduling_type == 'all'):
            scheduler_list = {1 : list(range(len(x_mean)))}

        else:
            raise ValueError("Invalid scheduling type")

        return scheduler_list
    
    def obtain_next(self, scheduler_num, scheduler_list, active_domains, trained_domains, fixed_domains):
        scheduler_num += 1
        active_domains = scheduler_list[scheduler_num]
        trained_domains = list(set(trained_domains + active_domains))
        fixed_domains = scheduler_list[scheduler_num-1]
        if(scheduler_num+1 <= len(scheduler_list)):
            for element in scheduler_list[scheduler_num+1]:
                if element in trained_domains:
                    fixed_domains.append(element)
        
        fixed_domains =  list(set(fixed_domains))

        print("\n Scheduler no is ", scheduler_num)
        print("\n Scheduler list is ", scheduler_list)
        print("\n Active Domains are ", active_domains)
        print("\n Fixed Domains are ", fixed_domains)
        print("\n Trained Domains are ", trained_domains)

        return scheduler_num, active_domains, trained_domains, fixed_domains
    
    def reset_scheduler(self):
        scheduler_num = 1
        active_domains = self.scheduler_list[scheduler_num]
        fixed_domains = self.scheduler_list[scheduler_num+1]
        trained_domains = active_domains
        print("\n Scheduler no is ", scheduler_num)
        print("\n Scheduler list is ", self.scheduler_list)
        print("\n Active Domains are ", active_domains)
        print("\n Fixed Domains are ", fixed_domains)
        print("\n Trained Domains are ", trained_domains)
        
        return scheduler_num, active_domains, trained_domains, fixed_domains