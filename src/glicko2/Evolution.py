'''
DO NOT NEED A MAIN MODULE HERE, THESE EVOLUTION HELPER FUNCTIONS WILL BE CALLED DURING AND AFTER TRAINING.

'''


import numpy as np





def skill_rating():
    
    
    
    
    
    return pass










#The idea is to check whether each discriminator in the last generation has at least one generator in its error map
#If all discriminators are indeed all infected --> Pathogens adaptation is complete
#If not, the mutation is not fixed --> No complete adaptation
#TODO: Take into account the silent infection (depending on the fitness value of the generator)
def pathogens_complete_adaptation(discriminators_last_generation):
    
    i = 0
    all_pathogens_adapted = True
    
    while i < len(discriminators_last_generation):
        
        if not (discriminators_last_generation[i].gen_error_map):  #Enter the loop when the discriminator's error_map is empty
            all_pathogens_adapted = False
            break
        
        i += 1
        
    return all_pathogens_adapted
        
    
    
    
#Similarly to the pathogens complete adaptation, we consider the hosts adaptation as complete when no host is infected anymore.
#Also need to consider the silent infection 
def hosts_complete_adaptation(discriminators_last_generation):
    
    i = 0
    all_hosts_adapted = True
    
    while i < len(discriminators_last_generation):
        
        if bool(discriminators_last_generation[i].gen_error_map):  #Enter the loop when the discriminator's error_map is non-empty
            all_hosts_adapted = False
            break
        
        i += 1
        
    return all_hosts_adapted

        


trace_dump_file = 'run_trace_bis.csv'    
    
#Function to extract all the discriminators present in the last generation, to be used in the above two functions
#Takes as argument the complete run trace, to parse and extract. (either run_trace from arena or run_trace_bis from post_analysis)
#Did not use the extractor methods already written because they dont give back the most recent generation's population, but the first one
#This function works for ALL methods (chain_evolve_with_fitness_reset, chain_evolve, homogenus_chain_progression, 
# chain_progression, round_robin_randomized, round_robin_deterministic) expect the brute force training (does not have final cross-match)
#TODO: adapt this to support brute force training too
def extract_all_discriminators_from_last_generation(file_name): #file_name should be the above trace_dump_file
    
    all_discriminators = []
    
    with open(file_name, 'r') as trace_file:
        reader = csv.reader(trace_file, delimiter='\t')
        for row in reader:            
            if row[0] == 'final cross-match:':  # enter master run
                all_discriminators.append(row[3])
    
    distinct_final_discriminators = np.unique(all_discriminators)
    
    return distinct_final_discriminators
        
         
        
        
#TODO: soft sweeps/ hard sweeps/ change in environment?? / plots? / adapt to the other code (train mainly) ..etc
