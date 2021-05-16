import random
import numpy as np
import os
import csv

from copy import deepcopy


evo_debug_file = 'evo_debug.csv'

def dump_evo(payload_list):
    if not os.path.isfile(evo_debug_file):
        open(evo_debug_file, 'w')

    with open(evo_debug_file, 'a') as destination:
        writer = csv.writer(destination, delimiter='\t')
        writer.writerow(payload_list)
        

#fitness updates -- works for both hosts and pathogens         
def update_fitnesses(individuals_list): #referenced many times in "arena.py"
    for individual in individuals_list:
        individual.finish_calc_skill_rating()
        individual.current_fitness = individual.skill_rating.mu
        
        
        
#Aneuploid explosion
def aneuploidization(individual): #510 of "match_and_train.py" in cross_train()
    '''
    Input: Individual -- Either a generator or discriminator instance
    Output: Returns nothing -- inplace changes
    
    Randomly performs a mutation (or not) of a small proportion in the parameter weights of the given instance
    '''
    
    
    '''
    #test
    i = 0 
    for name, params in individual.named_parameters():
        i +=1
        dump_evo([i, '  name: ', name])
        dump_evo([i, '  params: ', params])
    '''    
    
    proportions = [0.1, 0.075, 0.05, 0.025] #possible proportions of the number of weight layers changing
    proportion = random.choice(proportions) #randomly choose one, from 1/10th (0.1) to 1/100th (0.01)
    
    
    #test
    #dump_evo(['proportion: ', proportion])
    
    
    weights_layers = []

    #with torch.no_grad():

    #extract the weight layers (no bias layers)
    for name, _ in individual.named_parameters():
        if name.find("weight") != -1 :
            weights_layers.append(name)

    #randomly select weight layers for possible random change
    #Given the network structure, containing always 8 or 9 weight layers, and the proportions chosen manually above, 
    #we will always end up randomly changing either one weight layer (when proportion= 0.1 or 0.075), or nothing 
    #(when proportion= 0.05 or 0.025) thanks to the np.around() below.
    #So we have randomness on whether to change or not, if yes which layer (they have different sizes), and multiplying by 0.5 or 2
    nb_weight_layers = len(weights_layers)
    number_of_weight_layers_to_change = int(np.around(nb_weight_layers*proportion))
    weight_layers_to_change = random.sample(weights_layers, number_of_weight_layers_to_change)

    '''
    #test
    dump_evo(['weights_layers: ', weights_layers, 'nb_weight_layers: ', nb_weight_layers, \
               'number_of_weight_layers_to_change', number_of_weight_layers_to_change,\
               'weight_layers_to_change', weight_layers_to_change])
    '''
    
    new_params = {}

    for name, params in individual.named_parameters():
        new_params[name] = params.clone() #clone the parameters to be able to change later (or add torch.no_grad() ?)
        if name in weight_layers_to_change:
            new_params[name] = params * random.choice([0.5, 2]) #randomly either divide or multiply by 2
            params.data.copy_(new_params[name]) #copy back the newly muted parameters
    
    '''
    #test
    for name, params in individual.named_parameters():
        dump_evo(['name: ', name])
        dump_evo(['params: ', params])
    '''

    
    

#Implementation depends on how we will use this later
#As of now, this function will update the ".adapt" attribute of each single given pathogen to mark whether it has adapted or not

#initially it was implemented to check for a whole list of pathogens, and returns two lists one containing
#those who adapted and one those who did not
#changed due to implementation constrains in arena (now takes only one instance and update the adapt attribute)
def pathogens_adaptation_check(pathogen): #240 "arena.py" inside cross_train_iteration()
    '''
    Input: Generators list
    Output: Returns nothing -- inplace changes
    
    Updates the ".adapt" attribute of each pathogen depending on its fitness map
    '''
    
    #adapted_pathogens = []
    #non_adapted_pathogens = []
    
        
    if not bool(pathogen.fitness_map):#if fitness_map{} is empty
        pathogen.adapt = False
        #non_adapted_pathogens.append(pathogen)#Careful, we are adding the random tags, not the instances

    else:
        pathogen.adapt = True #if there is at least one host in the fitness_map{}
        #adapted_pathogens.append(pathogen)
            
    '''
    #test
    for pathogen in adapted_pathogens:
        dump_evo(['pathogen: ', pathogen.random_tag, ' adapted with fitness map: ', pathogen.fitness_map,\
                 ' and with current fitness: ', pathogen.current_fitness])
        
    for pathogen in non_adapted_pathogens:
        dump_evo(['pathogen: ', pathogen.random_tag, ' did not adapt with empty fitness map: ', pathogen.fitness_map,\
                 ' and with current fitness: ', pathogen.current_fitness])
    '''
    
    #return adapted_pathogens, non_adapted_pathogens


#Same reasoning for discriminators
def hosts_adaptation_check(host): #239 "arena.py" inside cross_train_iteration()
    '''
    Input: Hosts list
    Output: Returns nothing -- inplace changes
    
    Updates the ".adapt" attribute of each host depending on its gen error map
    '''
    
    #adapted_hosts = []
    #non_adapted_hosts = []
    
    #for host in hosts_list:
        
    if bool(host.gen_error_map):#if gen_error_map{} is non-empty (existance of a pathogen's random_tag)
        host.adapt = False
        #non_adapted_hosts.append(host)

    else:
        host.adapt = True #if there isn't a single pathogen in the gen_error_map{}
        #adapted_hosts.append(host)
    
    '''
    #test
    for host in adapted_hosts:
        dump_evo(['host: ', host.random_tag, ' adapted with empty gen_error_map: ', host.gen_error_map,\
                 ' and current fitness: ', host.current_fitness])
        
    for host in non_adapted_hosts:
        dump_evo(['host: ', host.random_tag, ' did not adapt with gen_error_map: ', host.gen_error_map,\
                 ' and current fitness: ', host.current_fitness])
    
    '''
    
    #return adapted_hosts, non_adapted_hosts



#first thoughts -- to be added after a complete method (after chain evolve, round robin ..etc) or after a proportion of it
#(for instance between the cross_train_iteration and evolve_in_population in chain evolve..) --> so that we only pass the best individuals
#to be trained and evolve further. (only select before the final cross match..)
#works for both hosts and pathogens
def select_best_individuals(individuals_list, proportion=0.5):#tested to work properly in "arena.py" 290, at the end of cross_train_iter()
    '''
    Input: List of hosts or pathogens, along with a float number to specify the proportion of individuals we want to keep
    Output: Returns a list containing the proportion of the best individuals
    
    Finds that proportion of best fit individuals and returns them
    '''
    
    best_individuals = []
    nb_of_individuals = len(individuals_list)
    nb_of_individuals_to_select = int(np.around(nb_of_individuals*proportion))
    
    individuals_sorted = sorted(individuals_list, key=lambda x: x.current_fitness, reverse=True)
    
    for i in range(nb_of_individuals_to_select):
        best_individuals.append(individuals_sorted[i])
    
    '''
    #test
    dump_evo(['individuals_list: ', [(ind.random_tag, ind.current_fitness) for ind in individuals_list]])
    dump_evo(['nb_of_individuals_to_select: ', nb_of_individuals_to_select])
    dump_evo(['individuals_sorted: ', [(ind.random_tag, ind.current_fitness) for ind in individuals_sorted]])
    dump_evo(['best_individuals: ', [(ind.random_tag, ind.current_fitness) for ind in best_individuals]])
    '''
    
    return best_individuals



#With a very small probability (1% as of now) of this happening, randomly kill 2/3 of the current generation (huge -natural- catastrophy)
#with some other more important probability (4% as of now), randomly kill 1/3 of the current generation (could be due to a pandemic
#or random destruction or food shortage or again natural catastrophy ..etc)
#The surviving instances will produce enough children to go back to the initial population size
#for instance with 3 hosts and 3 pathogens coadapting, we'll have 1 host and 1 pathogen left, and each of them
#will generate 3 children (copies of parent instance + new random_tag) to give back a fresh new generation of 3 vs 3 (this is for 1% case)
def bottleneck_effect(hosts_list, pathogens_list):#tested to work properly in "arena.py" 928, inside chain_evolve
                                                   #when entered:   a generation (and population) change is going to happen
    '''
    Input: List of hosts, list of pathogens
    Output: Returns a new list of hosts, new list of pathogens
    
    Randomly does nothing , or kill a proportion of the population and let the best surviving instance reproduce more
    '''
    
    kill_one_third_prop = 1/3
    kill_two_thirds_prop = 2/3
    
    possibilities = ['kill_two_thirds', 'kill_one_third', 'kill_no_one']
    weights = [0.01, 0.04, 0.95]
    choice = random.choices(possibilities, weights)
    
    if choice == 'kill_no_one':
        
        return hosts_list, pathogens_list #nothing happens
    
    elif choice == 'kill_one_third':
        
        new_hosts = bottleneck_process(hosts_list, kill_one_third_prop)
        new_pathogens = bottleneck_process(pathogens_list, kill_one_third_prop)
        return new_hosts, new_pathogens
    
    else: #choice == 'kill_two_thirds'
        
        new_hosts = bottleneck_process(hosts_list, kill_two_thirds_prop)
        new_pathogens = bottleneck_process(pathogens_list, kill_two_thirds_prop)
        
    return new_hosts, new_pathogens


#helper function where we factored out code for above bottleneck_effect()
def bottleneck_process(instances, kill_proportion):
    
    new_instances = []
    nb_of_instances = len(instances)
    
    nb_survivors = int(np.around(nb_of_instances*(1- kill_proportion)))
    survivors = random.sample(instances, nb_survivors)
    best_survivor = select_best_individuals(survivors, 1/nb_survivors)[0] #extract the best instance alive to reproduce more
    space_to_fill = nb_of_instances - nb_survivors #number of excess children needed to get back to the intial population size
    
    #best_survivor --> we could have also chosen randomly from the survivors which one to reproduce, not the best necessarily
    
    '''
    #test
    dump_evo(['nb_survivors: ', nb_survivors, '  survivors: ', [survivor.random_tag for survivor in survivors],\
              '  best_survivor ', best_survivor.random_tag, '  space_to_fill: ', space_to_fill])
    '''
    
    selected_instance = deepcopy(best_survivor) #used next for multiple "births"

    #progeny of instances that survived (normal generation change)
    for i in range(nb_survivors):
        #dump_evo(['survivors[i]: ', survivors[i].random_tag, '  with tag trace: ', survivors[i].tag_trace])
        survivors[i].bump_random_tag()
        #dump_evo(['survivors[i]: ', survivors[i].random_tag, '  with tag trace: ', survivors[i].tag_trace])
        new_instances.append(survivors[i])
        #dump_evo(['new instances: ', [instance.random_tag for instance in new_instances]])

    #excess generations of best alive instance to fill the gap in the population size
    for i in range(space_to_fill):
        #dump_evo(['selected_instance: ', selected_instance.random_tag, '  with tag trace: ', selected_instance.tag_trace])
        actual_copy = deepcopy(selected_instance)
        #dump_evo(['actual_copy: ', actual_copy.random_tag, '  with tag trace: ', actual_copy.tag_trace])
        actual_copy.bump_random_tag()
        #dump_evo(['actual_copy: ', actual_copy.random_tag, '  with tag trace: ', actual_copy.tag_trace])
        new_instances.append(actual_copy)
        #dump_evo(['new instances: ', [instance.random_tag for instance in new_instances]])

    return new_instances


#Important question in 411 "match_and_train" in match()


