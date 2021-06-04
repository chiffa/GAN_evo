import pickle
from pprint import pprint
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors as mcol
from scipy.stats import ttest_ind
import csv
from datetime import datetime
from collections import defaultdict
from itertools import combinations, product
import random
from matplotlib.lines import Line2D
from os import listdir
from os.path import isfile, join
from configs import current_dataset as _dataset
import os


trace_dump_file = 'run_trace_bis.csv'
backflow_log = 'backflow.csv'

bruteforce_gen2disc = {}
bruteforce_disc2addchars = {}

# below the mapping convention is
# after_update_tag : [pre_update_tag, post_update_partner_tag]
gen_tag_trace = {}  # Maps the random tag to its previous state and disc that changed it
disc_tag_trace = {}  # Maps the random tag to its previous state and gen that changed it.

encounter_record = {}

#EVO
master_gen_fit_map = {}

master_disc_fit_map = {}

#soft_sweeps = {} #EVO - SWEEPS_BOXPLOTS
#hard_sweeps = {} #EVO - SWEEPS_BOXPLOTS


master_fid_map = {}

master_is_map = {}

disc_phases = {}


#EVOOOO
#sweeps = {}


def stitch_run_traces(run_trace_diretory, filter):

    
    files_to_stitch = [os.path.abspath(join(run_trace_diretory, f)) for f in listdir(
                       run_trace_diretory)
                       if (isfile(join(run_trace_diretory, f))
                           and f[:9] == 'run_trace'
                           and f != trace_dump_file)]
    
    print("stitching dataset %s. Found to stitch: %s" % (_dataset, files_to_stitch))
    print("stitching datasets into %s" % os.path.abspath(trace_dump_file))

    with open(trace_dump_file, 'w') as outfile:
        #print('FILE NAME: ',os.path.abspath(trace_dump_file))
        for fname in files_to_stitch:
            if filter not in fname:
                continue
            with open(fname) as infile:
                print('dumping %s' % fname)
                lines = infile.read().splitlines()
                last_line = lines[-2]
                if last_line[0] == '<':
                    print("integrated %s" % fname)
                    for line in lines:
                        line += '\n'
                        outfile.write(line)



def parse_past_runs(trace_dump_locations):

    master_table = []

    # debug_counter = -1
    # print(os.path.abspath(trace_dump_locations))
    # print(os.path.getsize(trace_dump_locations))
    # print('debug: opening %s as trace dump' % trace_dump_locations)
    with open(trace_dump_locations, 'r') as trace_dump_file:
        reader = csv.reader(trace_dump_file, delimiter='\t')
        for row in reader:
            # debug_counter -= 1
            # print(row)
            # if row[-1] == '2020-04-01T00:55:58.583021':
                # debug_counter = 5
            # if debug_counter == 0:
            #     raise Exception('debug')
            if row[0] == '>':  # enter master run
                master_table.append([row, ])
                # print('lvl1 enter')
                continue
            if row[0] == '>>':  # enter run sub-section
                master_table[-1].append([row, ])  # in the latest master run add a function
                # print('lvl2 enter')
                continue
            if row[0] == '>>>':  # enter run sub-sub-section
                master_table[-1][-1].append([row, ])  # and a sub-function
                # print('lvl3 enter')
                print(master_table[-1][-1][-1])
                continue
            if row[0] == '<<<':  # exit run sub-sub-section
                master_table[-1][-1][-1].append(row)
                # print('lvl3 exit')
                print(master_table[-1][-1][-1])
                continue
            if row[0] == '<<':  # exist sub-section
                master_table[-1][-1].append(row)
                # print('lvl2 exit')
                continue
            if row[0] == '<':  # exit master run
                master_table[-1].append(row)
                # print('lvl1 exit')
                continue
            # print('default add')
            master_table[-1][-1][-1].append(row)
            # print(master_table[-1][-1][-1])

    return master_table


def extract_bruteforce_data(bruteforce_run):
    duration = (datetime.fromisoformat(bruteforce_run[-1][2]) -
                datetime.fromisoformat(bruteforce_run[0][4])).total_seconds()/60.
    try:
        duration = float(bruteforce_run[-1][-1])
    except:
        # print('old format timing')
        pass
    
    is_collector  = []
    fid_collector = []
    tag_collector = []
    
    for data_dump_row in bruteforce_run[1]:
        if data_dump_row[0] == 'sampled images from':
            is_collector.append(float(data_dump_row[-1]))                                     
            fid_collector.append(float(data_dump_row[-2]))
            tag_collector.append(data_dump_row[2])
        if data_dump_row[0] == 'post-cross-train and match:':
            # print('parsing bruteforce')
            bruteforce_gen2disc[data_dump_row[4]] = data_dump_row[3]
            bruteforce_disc2addchars[data_dump_row[3]] = bruteforce_run[0][2:4]

    return is_collector, fid_collector, tag_collector, duration                               


def extract_evo_data(chain_evolve_run):
    duration = (datetime.fromisoformat(chain_evolve_run[-1][2]) -
                datetime.fromisoformat(chain_evolve_run[0][4])).total_seconds()/60.
    # print('duration raw:', duration)
    try:
        duration = float(chain_evolve_run[-1][-1])
    except:
        # print('old format timing')
        pass
    # print('duration corrected:', duration)
    
    final_pathogens_list = chain_evolve_run[-2][0][3]
    final_pathogens_list = final_pathogens_list[1:-1].split(', ')
    
    
    
    gen_sweeps = {}
    disc_sweeps = {}
    
    
    #EVO
    final_hosts_list = chain_evolve_run[-2][0][2]
    final_hosts_list = final_hosts_list[1:-1].split(', ')
        
    #print('chain_evolve_run[-2][0]', chain_evolve_run[-2][0])
    
    #print('')
    #print('')
    #print('')
    #print('final_pathogens_list ', final_pathogens_list)
        
    #print('final_hosts_list ', final_hosts_list)
    #print('')
    
    
    
    
    
    
    
    is_collector  = [-1]*len(final_pathogens_list)
    fid_collector = [-1]*len(final_pathogens_list)
    gen_tag_collector = ["None"]*len(final_pathogens_list)
    
    
    disc_tag_collector = ["None"]*len(final_hosts_list)
    
    #EVO
    gen_fit_collector = [-1]*len(final_pathogens_list)
    disc_fit_collector = [-1]*len(final_hosts_list)

    
    pre_train_buffer = []
    
    #print('chain_evolve_run[-2][1:-1]: ',chain_evolve_run[-2][1:-1])
    
    for entry in chain_evolve_run[-2][1:-1]:  # here we are pulling the data from the last run alone

        
        #print('entry: ', entry)
        
        
        if entry[0] == 'sampled images from':
            # print(entry)
                 
            gen_fit_collector[int(entry[1])] = float(entry[-2]) #EVO   
            is_collector[int(entry[1])]  = float(entry[-3])
            fid_collector[int(entry[1])] = float(entry[-4])
            gen_tag_collector[int(entry[1])] = entry[2]
            
        #EVOOOO
        if entry[0] == 'against host':
            
            disc_fit_collector[int(entry[1])] = float(entry[-2]) #EVO
            
            disc_tag_collector[int(entry[1])] = entry[2]
     
    
    
    #EVO
    #print('gen_tag_collector', gen_tag_collector)
    #print('disc_tag_collector', disc_tag_collector)
            
    #print('')
    
    #EVOO        
    #print('gen_fit_collector: ', gen_fit_collector)
    #print('disc_fit_collector: ', disc_fit_collector)
    
    #print('')
    #print('')
    
    #With veeeeery small probability if disc was never chosen -- we should implement the same for the gens !
    for i in range(len(disc_fit_collector)):
        if disc_fit_collector[i] == -1:
                        
            for entry in chain_evolve_run[-2][1:-1]:
                #print('entry: ', entry)
                if entry[0] == 'current tag/fitness tables':
                    #print('entry[3]: ', entry[3])
                    #print('entry[3][i]: ', float((entry[3][1:-2]).split(', ')[i]))
                    disc_fit_collector[i] = float((entry[3][1:-1]).split(', ')[i])
                    
                                
    
    #Same thing when collecting the random tags
    for i in range(len(disc_tag_collector)):
        if disc_tag_collector[i] == 'None':
            
            for entry in chain_evolve_run[-2][1:-1]:
                if entry[0] == 'current tag/fitness tables':#(entry[1][1:-2]).split(', ')[i]
                    #print('entry ', entry)
                    
                    #print('entry[1][i]: ', ((entry[1][1:-1]).split(', ')[i])[1:-1]             )
                                        
                    disc_tag_collector[i] = ((entry[1][1:-1]).split(', ')[i])[1:-1]
    
    
    
    #EVOOO -- Same for the generators
    
    for i in range(len(gen_fit_collector)):
        if gen_fit_collector[i] == -1:
                        
            for entry in chain_evolve_run[-2][1:-1]:
                #print('entry: ', entry)
                if entry[0] == 'current tag/fitness tables':
                    #print('entry[3]: ', entry[3])
                    #print('entry[3][i]: ', float((entry[3][1:-2]).split(', ')[i]))
                    gen_fit_collector[i] = float((entry[4][1:-1]).split(', ')[i])
                    
                                
    
    for i in range(len(gen_tag_collector)):
        if gen_tag_collector[i] == 'None':
            
            for entry in chain_evolve_run[-2][1:-1]:
                if entry[0] == 'current tag/fitness tables':#(entry[1][1:-2]).split(', ')[i]
                    #print('entry ', entry)
                    
                    #print('entry[1][i]: ', ((entry[1][1:-1]).split(', ')[i])[1:-1]             )
                                        
                    gen_tag_collector[i] = ((entry[2][1:-1]).split(', ')[i])[1:-1]
    
    
    
    
    
    #EVO
    #print('gen_tag_collector', gen_tag_collector)
    #print('disc_tag_collector', disc_tag_collector)
            
    #EVOO        
    #print('gen_fit_collector: ', gen_fit_collector)
    #print('disc_fit_collector: ', disc_fit_collector)
    
    
    
    # parameters =
    # in cross-train: 4 / -3
    # in evo:

    for sub_run in chain_evolve_run[1:-1]:
        for entry in sub_run[1:-1]:
            
            #print('entry ', entry)
            
            
            #EVOOOO
            if entry[0] == 'against host':
                master_disc_fit_map[entry[2]] = float(entry[-2])
                
                disc_sweeps[entry[2]] = entry[-1]
            
            
            
            if entry[0] == 'sampled images from':
                master_fid_map[entry[2]] = float(entry[-4])
                master_is_map[entry[2]]  = float(entry[-3])
                
                master_gen_fit_map[entry[2]] = float(entry[-2])#EVO
                
                gen_sweeps[entry[2]] = entry[-1]#EVO
                

            if entry[0] in ['infection attempt:', 'pre-train:']:
                pre_train_buffer = entry

            if entry[0] in ['post-cross-train and match:', 'post-infection']:
                # print('ptb', pre_train_buffer)
                # print('entry', entry)
                gen_tag_trace[entry[4]] = [pre_train_buffer[4], entry[3]]
                disc_tag_trace[entry[3]] = [pre_train_buffer[3], entry[4]]
                encounter_record[(entry[3], entry[4])] = (float(entry[5]), float(entry[6]))

                disc_phases[entry[3]] = [sub_run[0][1], None]
                if sub_run[0][1] == 'cross-train':
                    disc_phases[entry[3]][1] = sub_run[0][4]

                    
    #print('')            
    
    #print('master_gen_fit_map', len(master_gen_fit_map), master_gen_fit_map)
    #print('master_disc_fit_map', len(master_disc_fit_map), master_disc_fit_map)
    
    
    #print('')
    
    #print('gen_sweeps', len(gen_sweeps), gen_sweeps)
    
    #print('disc_sweeps', len(disc_sweeps), disc_sweeps)
    
    #print('')
    
            
            
            
    #         if 'PYXTR1B9CC' in entry:
    #             print('dbg', entry)
    #             print('dbg ptb', pre_train_buffer)
    #
    #             if entry[0] in ['post-cross-train and match:', 'post-infection']:
    #                 raise Exception('debug')
    #
    # raise Exception('other debug')

    return disc_fit_collector, gen_fit_collector, is_collector, fid_collector, disc_tag_collector, gen_tag_collector, duration,\
           disc_sweeps, gen_sweeps


def extract_battle_royale_data(battle_royale_run):
    collector_list = []
    gen_set = set()
    disc_set = set()
    # print(battle_royale_run[1])
    # raise Exception('debugging')
    for entry in battle_royale_run[1]:
        if entry[0] == 'post-cross-train and match:':
            collector_list.append(entry[1:])
            gen_set.add(entry[2])
            disc_set.add(entry[1])

    gen_index = dict((name, _i) for _i, name in enumerate(list(gen_set)))
    disc_index = dict((name, _i) for _i, name in enumerate(list(disc_set)))

    real_error_matrix = np.ones((len(gen_set), len(disc_set))) * np.nan
    gen_error_matrix = np.ones((len(gen_set), len(disc_set))) * np.nan

    for disc, gen, real_err, gen_err in collector_list:
        real_error_matrix[gen_index[gen], disc_index[disc]] = float(real_err)
        gen_error_matrix[gen_index[gen], disc_index[disc]] = float(gen_err)

    return gen_index, disc_index, real_error_matrix, gen_error_matrix




def render_fid_is_performances(attribution_map, gen_soft_sweeps, gen_hard_sweeps, disc_soft_sweeps, disc_hard_sweeps):

    def draw_p_vals_table(_dataset):
        p_matrix = np.ones((len(_dataset), len(_dataset)))
        ratio_matrix = np.ones((len(_dataset), len(_dataset)))

        for (i_1, subset1), (i_2, subset2) in combinations(enumerate(_dataset), 2):
            _, pval = ttest_ind(subset1, subset2)
            p_matrix[i_1, i_2] = pval
            p_matrix[i_2, i_1] = pval

            ratio_matrix[i_1, i_2] = np.median(subset1) / np.median(subset2)
            ratio_matrix[i_2, i_1] = np.median(subset2) / np.median(subset1)

        # norm = mcol.BoundaryNorm([0., 0.01, 0.05, 0.1, 1.], ncolors=256)
        # plt.imshow(p_matrix, cmap='RdBu_r', interpolation=None, norm=norm)

        plt.imshow(ratio_matrix, cmap='RdYlGn', interpolation=None)#, vmin=0.3, vmax=1.8)
        plt.colorbar()
        stat_sig = np.argwhere(p_matrix < 0.05).T
        plt.scatter(stat_sig[0], stat_sig[1], marker='*', c='k')

        _method_names = [name_remap(method) for method in method_names]

        plt.xticks(range(len(_method_names)), _method_names)
        plt.yticks(range(len(_method_names)), _method_names)

        plt.xticks(rotation=30, rotation_mode="anchor", ha="right")
        plt.tight_layout()


    def draw_box_plot(_dataset):
        flatten = lambda l: [item for sublist in l for item in sublist]

        
        for name, data in zip(method_names, _dataset):
            
            
            load = [name_remap(name), '&',
                    '%.2f' % np.median(data), '&',
                    '%.2f' % np.mean(data), '&',
                    '%.2f' % np.std(data), '&',
                    '%d' % len(data), '\\\\']
            print(' '.join(load))
            
        
        plt.boxplot(_dataset)   #4 locs bec the dataset has 4 columns (first pass)
        x_pad = [[_i+1  # +random.random()/10.
                  for _ in range(len(_data))]
                 for _i, _data in enumerate(_dataset)]
        
                
        
        plt.scatter(flatten(x_pad), flatten(_dataset), c='k')
        locs, labels = plt.xticks()
        #locs = [1,2,3,4]
        #print('locs: ', locs)
        #print('labels: ', labels)
        
        _method_names = [name_remap(method) for method in method_names]
        
        plt.xticks(locs, _method_names)
        plt.xticks(rotation=30, rotation_mode="anchor", ha="right")
        plt.tight_layout()


    method_names = []
    best_fids_achieved = []
    all_fids_achieved = []
    exec_times = []
    
    best_is_achieved = []
    all_is_achieved = []
    
    #print('attribution_map.items()', attribution_map.items())

    for i, (key, value) in enumerate(attribution_map.items()):
                
        method_names.append(' '.join(key))
        best_fids_achieved.append([])
        exec_times.append([])
        all_fids_achieved.append([])
        
        best_is_achieved.append([])
        all_is_achieved.append([])
        
        
        #print('value: ', value)#EVO
        #print('value.items(): ', value.items())#EVO
        
        
        for sub_key, (exec_time, fids, is_scores, fitnesses, tags) in value.items():#EVO -- only modif in this method
            
            best_fids_achieved[i].append(min(fids))
            best_is_achieved[i].append(max(is_scores))
            
            exec_times[i].append(exec_time)
            all_fids_achieved[i] += fids
            all_is_achieved[i]   += is_scores

    #print('best_fids_achieved: ', best_fids_achieved)
    #print('best_is_achieved: ', best_is_achieved)
    #print('all_fids_achieved: ', all_fids_achieved)
    #print('all_is_achieved: ', all_is_achieved)
            
    plt.title('Minimum FID achieved per run by a method')
    plt.ylabel('FID')
    draw_box_plot(best_fids_achieved)
    #plt.show()
    plt.savefig("./post_analysis_images/Min_fid_achieved.png")
    plt.clf()
    
    plt.title('Maximum IS achieved per run by a method')
    plt.ylabel('IS')
    draw_box_plot(best_is_achieved)
    #plt.show()
    plt.savefig("./post_analysis_images/Max_is_achieved.png")
    plt.clf()
        
    
    
    plt.title('Relative performance of methods for best FID achieved')
    draw_p_vals_table(best_fids_achieved)
    #plt.show()
    plt.savefig("./post_analysis_images/relative_fid_best.png")
    plt.clf()
    
    plt.title('Relative performance of methods for best IS achieved')
    draw_p_vals_table(best_is_achieved)
    #plt.imshow(vmin=0.5, vmax=1.5)
    plt.savefig("./post_analysis_images/relative_is_best.png")
    plt.clf()


    
    plt.title('FID of all generators per method')
    plt.ylabel('FID')
    draw_box_plot(all_fids_achieved)
    #plt.show()
    plt.savefig("./post_analysis_images/Fid_all_gens_per_method.png")
    plt.clf()
    
    plt.title('IS of all generators per method')
    plt.ylabel('IS')
    draw_box_plot(all_is_achieved)
    #plt.show()
    plt.savefig("./post_analysis_images/Is_all_gens_per_method.png")
    plt.clf()
    
    
    
    plt.title('Relative performance of methods overall (in FID terms)')
    draw_p_vals_table(all_fids_achieved)
    #plt.show()
    plt.savefig("./post_analysis_images/relative_fid_overall.png")
    plt.clf()

    plt.title('Relative performance of methods overall (in IS terms)')
    draw_p_vals_table(all_is_achieved)
    #plt.imshow(vmin=0.5, vmax=1.5)
    plt.savefig("./post_analysis_images/relative_is_overall.png")
    plt.clf()
    
    
    
    plt.title('Single run time for each method')
    plt.ylabel('Run time')
    draw_box_plot(exec_times)
    #plt.show()
    plt.savefig("./post_analysis_images/runtime_per_method.png")
    plt.clf()
    
    

    plt.title('Best FID per run vs run time')
    # c_pad = [[i/len(exec_times)]*len(_data) for i, _data in enumerate(exec_times)]
    for i, lab in enumerate(method_names):
        plt.scatter(exec_times[i], best_fids_achieved[i], label=name_remap(lab))
    plt.legend()
    plt.xlabel('run time (mins)')
    plt.ylabel('minimal FID achieved')
    #plt.show()
    plt.savefig("./post_analysis_images/best_fid_vs_runtime.png")
    plt.clf()
    
    plt.title('Best IS per run vs run time')
    # c_pad = [[i/len(exec_times)]*len(_data) for i, _data in enumerate(exec_times)]
    for i, lab in enumerate(method_names):
        plt.scatter(exec_times[i], best_is_achieved[i], label=name_remap(lab))
    plt.legend()
    plt.xlabel('run time (mins)')
    plt.ylabel('maximum IS achieved')
    #plt.show()
    plt.savefig("./post_analysis_images/best_is_vs_runtime.png")
    plt.clf()
    
    
    
    
    
    ####EVOOOO - SWEEPS PLOTS
    
    #GEN
    sweeps_input_gen = [gen_soft_sweeps, gen_hard_sweeps]
    
    plt.title('Dominant Adaptation Modes - Pathogens')
    plt.ylabel('Count')
    Sweeps_box_plot(sweeps_input_gen)
    #plt.show()
    plt.savefig("./post_analysis_images/Pathogens_Sweeps.png")
    plt.clf()
    
    
    #DISC
    sweeps_input_disc = [disc_soft_sweeps, disc_hard_sweeps]
    
    plt.title('Dominant Adaptation Modes - Hosts')
    plt.ylabel('Count')
    Sweeps_box_plot(sweeps_input_disc)
    #plt.show()
    plt.savefig("./post_analysis_images/Hosts_Sweeps.png")
    plt.clf()
    
    
############################################################################################


def Sweeps_box_plot(_dataset):
    
    flatten = lambda l: [item for sublist in l for item in sublist]


    '''
    for name, data in zip(method_names, _dataset):

        print('name: ', name)
        print('data: ', data)

        load = [name_remap(name), '&',
                '%.2f' % np.median(data), '&',
                '%.2f' % np.mean(data), '&',
                '%.2f' % np.std(data), '&',
                '%d' % len(data), '\\\\']
        print(' '.join(load))
    '''


    #print('_dataset: ', _dataset)#EVO



    plt.boxplot(_dataset)   #4 locs bec the dataset has 4 columns (first pass)
    x_pad = [[_i+1  # +random.random()/10.
              for _ in range(len(_data))]
             for _i, _data in enumerate(_dataset)]



    plt.scatter(flatten(x_pad), flatten(_dataset), c='b')
    locs, labels = plt.xticks()
    #locs = [1,2,3,4]
    #print('locs: ', locs)
    #print('labels: ', labels)

    adaptation_modes = ['Soft Sweeps', 'Hard Sweeps']

    plt.xticks(locs, adaptation_modes)
    #plt.xticks(rotation=30, rotation_mode="anchor", ha="right")
    plt.tight_layout()


############################################################################################
    
    
    
    
    
    

def pull_best_fid_is_tags(attribution_map):
    method_names = []
    best_fid_gen_tags = []
    best_is_gen_tags = []
    select_disc_fid_tags = []
    select_disc_is_tags = []

    for i, (key, value) in enumerate(attribution_map.items()):
        method_names.append(' '.join(key))
        best_fid_gen_tags.append([])
        best_is_gen_tags.append([])
        for sub_key, (exec_time, fids, is_scores, fitnesses, tags) in value.items():#EVO -- only modif in this method
            
            fids = np.array(fids)            
            min_loc = np.argmin(fids)
            best_fid_gen_tags[i].append(tags[min_loc])
            
            is_scores = np.array(is_scores)
            max_loc = np.argmax(is_scores)
            best_is_gen_tags[i].append(tags[max_loc])

    # pprint(bruteforce_gen2disc)

    for method, load in zip(method_names, best_fid_gen_tags):
        if 'brute-force' in method:
            for best_fid_brute_force_tag in load:
                select_disc_fid_tags.append(bruteforce_gen2disc[best_fid_brute_force_tag])

                
    for method, load in zip(method_names, best_is_gen_tags):
        if 'brute-force' in method:
            for best_is_brute_force_tag in load:
                select_disc_is_tags.append(bruteforce_gen2disc[best_is_brute_force_tag])    
    
    
    return method_names, best_fid_gen_tags, best_is_gen_tags, select_disc_fid_tags, select_disc_is_tags


def buffer_gen_disc_fid_tags(best_fid_gen_tags, select_disc_tags):

    _best_gen_tags = [item for sublist in best_fid_gen_tags for item in sublist]

    with open(backflow_log, 'w') as target:
        writer = csv.writer(target, delimiter='\t')
        for gen_tag, disc_tag in product(_best_gen_tags, select_disc_tags):
            writer.writerow([gen_tag, disc_tag])


def render_relative_performances(gen_index, disc_index,
                                 real_error_matrix, gen_error_matrix,
                                 method_names, best_fid_gen_tags):

    disc_full_names = ['']*gen_error_matrix.shape[1]
    keep_discs = np.zeros((gen_error_matrix.shape[1], )).astype(np.bool)
    for key, value in disc_index.items():
        if bruteforce_disc2addchars[key] != ['5', '30']:
            disc_full_names[value] = ' '.join([key] + bruteforce_disc2addchars[key])
            keep_discs[value] = True

    disc_full_names = np.array(disc_full_names)[keep_discs].tolist()

    per_method_performance = []

    for method, gen_tag_set in zip(method_names, best_fid_gen_tags):
        gen_tag_perf = []
        disc_tag_base = []

        for gen_tag in gen_tag_set:
            # print(real_error_matrix[gen_index[gen_tag], :].shape)
            disc_tag_base.append(real_error_matrix[gen_index[gen_tag], keep_discs])
            gen_tag_perf.append(gen_error_matrix[gen_index[gen_tag], keep_discs])

        disc_tag_base = np.vstack(disc_tag_base)
        gen_tag_perf = np.vstack(gen_tag_perf)

        # print(disc_tag_base.shape)
        # print(gen_tag_perf.shape)

        real_column = np.mean(disc_tag_base, axis=0)
        average_disc_perf_on_gen = np.median(gen_tag_perf, axis=0)

        total_perf = np.vstack([gen_tag_perf, real_column])

        # total_perf = np.vstack([gen_tag_perf/average_disc_perf_on_gen[np.newaxis, :], real_column])

        average_gen_perf = np.median(total_perf, axis=1)
        # print(total_perf.shape)
        # print(average_gen_perf[:, np.newaxis].shape)

        total_perf = np.hstack([total_perf, average_gen_perf[:, np.newaxis]])

        raw_perf = total_perf.copy()

        # total_perf[total_perf < 1. / 60000.] = 1. / 600000.  # MNIST dataset size clipping

        # print(((1-total_perf) / (1-total_perf[:, -1:]) - 1)*100)
        # print(total_perf[:, -1:]/(1-total_perf))
        # raise Exception('debug')

        # total_perf = (1 - total_perf[:, -1:]) / total_perf - \
        #               total_perf / (1 - total_perf[:, -1:])

        # total_perf = - ((1-total_perf) / (1-total_perf[-1:, :]) - 1) * 100

        per_method_performance.append(total_perf[:, -1].copy())

        # print(real_column.shape)
        # print(total_perf.shape)

        # limit = np.max(np.abs(total_perf))

        # plt.title(method)
        # plt.imshow(total_perf, cmap='RdYlGn', interpolation=None,
        #            # vmin=-limit,
        #            # vmax=limit
        #            )
        #
        # plt.yticks(np.arange(len(gen_tag_set) + 1), gen_tag_set + ['real perf'])
        # plt.xticks(np.arange(len(disc_full_names) + 1), disc_full_names + ['average perf'],
        #            rotation=90.)
        #
        # # Loop over data dimensions and create text annotations.
        # for i in range(total_perf.shape[0]):
        #     for j in range(total_perf.shape[1]):
        #         text = plt.text(j, i, '%.2e' % total_perf[i, j],
        #                        ha="center", va="center", color="w")
        #
        # #
        # # ax.set_yticks(np.arange(len(gen_tag_set) + 1))
        # # ax.set_yticklabels(gen_tag_set + ['real perf'])
        # #
        # # plt.setp(ax.get_xticklabels(),
        # #          rotation=45, ha="right",
        # #             rotation_mode="anchor")
        #
        # plt.colorbar()
        #
        # # ax.set_title(method)
        # # fig.tight_layout()
        #
        # plt.show()
        #
        # plt.title(method)
        # flatten = lambda l: [item for sublist in l for item in sublist]
        # total_perf = total_perf[:, :-1]
        # plt.boxplot(total_perf.tolist())
        # x_pad = [[_i+1  # +random.random()/10.
        #           for _ in range(len(_data))]
        #          for _i, _data in enumerate(total_perf.tolist())]
        # cmap = plt.get_cmap('RdYlGn')
        # average_disc_perf_on_gen /= np.max(average_disc_perf_on_gen)
        # disc_cs = [[cmap(val) for val in average_disc_perf_on_gen.tolist()]
        #            for _ in total_perf.tolist()]
        # # for _i, method_perf in enumerate(total_perf.tolist()):
        # plt.scatter(flatten(x_pad), flatten(total_perf.tolist()), c=flatten(disc_cs))
        #
        # plt.xticks(np.arange(len(gen_tag_set) + 1)+1, gen_tag_set + ['real perf'])
        #
        # plt.yscale('log')
        #
        # plt.show()
        #
        # # raise Exception('debug')

    flatten = lambda l: [item for sublist in l for item in sublist]
    plt.boxplot(per_method_performance)
    x_pad = [[_i+1  # +random.random()/10.
              for _ in range(len(_data))]
             for _i, _data in enumerate(per_method_performance)]
    c = np.ones_like(np.array(x_pad)).astype(np.str)
    plt.scatter(flatten(x_pad), flatten(per_method_performance), c='k')
    locs, labels = plt.xticks()
    plt.xticks(locs, method_names)
    plt.yscale('log')
    plt.xticks(rotation=90)
    #plt.show()
    plt.savefig('./post_analysis_images/render_relative_perf')
    plt.clf()

# CURRENTPASS: [complexity] cyclomatic complexity=28 (!)
def render_training_history(method_names, disc_sweeps, gen_sweeps):

    type_map = {0: 'X',
                1: '*',
                2: 'P',
                3: 'X'}

    colors_map = {1: 'k',
                  2: 'b'}
    
    
    
    #EVO
    type_map_2 = {'No adaptation': '.',
                  'Coadaptation' : '*',
                  'Soft Sweeps'  : 'o',  #EVO- uppercase
                  'Hard Sweeps'  : 'o'}
    
    colors_map_2 = {'No adaptation': 'k',
                    'Coadaptation' : 'g',
                    'Soft Sweeps'  : 'b', #EVO- uppercase
                    'Hard Sweeps'  : 'r'}
    

    '''
    #EVOOOOOOOOOOOOOOOOO
    gen_soft_sweeps = [] #EVO - SWEEPS_BOXPLOTS
    gen_hard_sweeps = [] #EVO - SWEEPS_BOXPLOTS
    
    disc_soft_sweeps = [] #EVO - SWEEPS_BOXPLOTS
    disc_hard_sweeps = [] #EVO - SWEEPS_BOXPLOTS
    '''
    
    
    
    def render_score_progression(method, gen_tags_trace):

        plt.title(name_remap(method)) #  (method) instead of [method]

        cmap = plt.get_cmap('RdYlGn')
        # print('gtt', gen_tags_trace)
        # print('gtt[0]', gen_tags_trace[0])
        # print('gtt[0][0]', gen_tags_trace[0][0])
        scores_per_line = [gen_line[0][-1] for gen_line in gen_tags_trace]
        # print('fpl', fids_per_line)

        scores_per_line = np.array(scores_per_line)

        lines_colors = (scores_per_line - np.min(scores_per_line)) /\
                       (np.max(scores_per_line) - np.min(scores_per_line))

        lines_colors = np.array(range(len(scores_per_line))) / float(len(scores_per_line))

        argsort = np.argsort(-scores_per_line)

        lines_colors = [cmap(score) for score in lines_colors[argsort]]

        type_idx_map = {}
        c = 0

        c_t_annotation_map = {}
        shown_c = []
        shown_t = []

        secondary_label_buffer = []
        secondary_label_buffer_2 = []

        for l_color, gen_line in zip(lines_colors, gen_tags_trace):
            root = gen_line[0][0]
            scores = [score for _, _, _, _, score in reversed(gen_line)]
            xs = range(len(scores))
            plt.plot(xs, scores,
                     # label=root,
                     c=l_color,
                     # linewidth=3
                     )

            disc_state = [disc_phases[disc_tag] for _, disc_tag, _, _, _ in gen_line]
            # print(disc_state)
            color_list = []
            type_list = []
            t = 0
            c_mem = ''
            cross_train_counter = 0
            # print('new train')

            for state in reversed(disc_state):
                # print('\ts', state)
                if state[0] == 'cross-train':
                    cross_train_counter += 1
                if state[1] is not None and c_mem != state[1] \
                        or cross_train_counter % 6 == 0:  # temporary
                    cross_train_counter = 1
                    # print('xt', c_mem, '->', state[1], ':', t)
                    t += 1
                    c_mem = state[1]
                    c_t_annotation_map[type_map[t]] = c_mem
                # print('\tt', c_mem, ':', t)
                type_list.append(type_map[t])

                if state[0] not in type_idx_map.keys():
                    # print('xc', state[0], '! in', type_idx_map.keys())
                    c += 1
                    type_idx_map[state[0]] = c
                    c_t_annotation_map[colors_map[type_idx_map[state[0]]]] = state[0]
                # print('\tc', state[0], ':', type_idx_map[state[0]])
                color_list.append(colors_map[type_idx_map[state[0]]])

            # print(xs, fids)

            for _x, _s, _t, _c in zip(xs, scores, type_list, color_list):

                if _t not in shown_t or _c not in shown_c:
                    shown_t.append(_t)
                    shown_c.append(_c)

                    # secondary_label_buffer.append(
                    plt.plot(_x, _s, marker=_t, c=_c,
                             markersize=8,
                         # label='%s, %s' % (c_t_annotation_map[_t], c_t_annotation_map[_c])
                         )
                    # )
                    # secondary_label_buffer_2.append('%s, %s' % (c_t_annotation_map[_t],
                    #                                                   c_t_annotation_map[_c]))

                else:
                    plt.plot(_x, _s, marker=_t, c=_c, markersize=8)

            # TODO: add the relative performance wrt competition as well as
            # the lane of the  disc.

        # pprint(c_t_annotation_map)
        if len(c_t_annotation_map.keys()) == 0:
            # print('problem detected')
            pass
        #plt.ylabel('FID or IS')
        plt.xlabel('encounter')

        
        legend_elements = []
        all_types = []
        all_colors = []

        for elt in type_map.values():
            if elt in c_t_annotation_map.keys() and elt not in all_types:
                all_types.append(elt)
                legend_elements.append(Line2D([0], [0],
                                  marker=elt,
                                  color='w',
                                  label=c_t_annotation_map[elt],
                                  markerfacecolor='k',
                                  markersize=8))

        if len(all_types) == 0:
            legend_elements.append(Line2D([0], [0],
                                  marker='x',
                                  color='w',
                                  label='base',
                                  markerfacecolor='k',
                                  markersize=8))

        for elt in colors_map.values():
            if elt in c_t_annotation_map.keys():
                # print('color legend elt:', elt)
                legend_elements.append(Line2D([0], [0],
                                  marker='s',
                                  color='w',
                                  label=c_t_annotation_map[elt],
                                  markerfacecolor=elt,
                                  markersize=8))

        plt.legend(handles=legend_elements)
        #plt.show()
        #plt.savefig('./post_analysis_images/final_figs.png')
        #plt.clf()

        # raise Exception('debug')    
        
        
        
        
        
        
        
        
        
    #EVO
    ##########################################################################################################################        
        
        
    def render_fitness_progression(method, tags_trace, sweeps_data):

        #EVOOO
        soft_sweeps = []
        hard_sweeps = []
        
        plt.title(name_remap(method)) #  (method) instead of [method]

        cmap = plt.get_cmap('plasma')
        # print('gtt', gen_tags_trace)
        # print('gtt[0]', gen_tags_trace[0])
        # print('gtt[0][0]', gen_tags_trace[0][0])
        scores_per_line = [line[0][-1] for line in tags_trace]
        # print('fpl', fids_per_line)

        scores_per_line = np.array(scores_per_line)

        lines_colors = (scores_per_line - np.min(scores_per_line)) /\
                       (np.max(scores_per_line) - np.min(scores_per_line))

        lines_colors = np.array(range(len(scores_per_line))) / float(len(scores_per_line))

        argsort = np.argsort(-scores_per_line)

        lines_colors = [cmap(score) for score in lines_colors[argsort]]

        type_idx_map = {}
        c = 0

        c_t_annotation_map = {}
        shown_c = []
        shown_t = []

        secondary_label_buffer = []
        secondary_label_buffer_2 = []
        
        
        
        
        
        
        
        for l_color, line in zip(lines_colors, tags_trace):
            
            
            
            
            
            
            #print('gen_tags_trace ', gen_tags_trace)
            #print('gen_line ', gen_line)
            
            
            
            root = line[0][0]
            scores = [score for _, _, _, _, score in reversed(line)]
            xs = range(len(scores))
            plt.plot(xs, scores,
                     # label=root,
                     c=l_color,
                     # linewidth=3
                     )

            #print('root ', root)
            #print('scores ', scores)
            
            
            #disc_state = [disc_phases[disc_tag] for _, disc_tag, _, _, _ in gen_line]
            
            #print('disc_phases', disc_phases)
            
            
            #EVO
            sweeps_state = [sweeps_data[tag] for tag, _, _, _, _ in line]
            
            
            
            soft_sweeps.append(sweeps_state.count('Soft Sweeps'))   #EVO - SWEEPS_BOXPLOTS // next run with uppercase s of sweeps
            hard_sweeps.append(sweeps_state.count('Hard Sweeps'))   #EVO - SWEEPS_BOXPLOTS 
            
            
            
            # print(disc_state)
            color_list = []
            type_list = []
            
            
            type_list_2 = []#hedhi twali principale
            color_list_2 = []#hedhi twali principale
            
            
            t = 0   #potentiellement zeyda
            c_mem = ''  #potentiellement zeyda
            cross_train_counter = 0  #potentiellement zeyda
            # print('new train')
            
            
            
            #print('sweeps_state ', sweeps_state)
            
            
            
            #gen_index +=1 #EVO - SWEEPS_BOXPLOTS
            
            
            
            for state in reversed(sweeps_state):
                # print('\ts', state)
                
                
                #print('state', state)
                
                '''
                #EVO-- edhouma ezouz zeydin
                if state[0] == 'cross-train':
                    cross_train_counter += 1
                
                
                if state[1] is not None and c_mem != state[1] \
                        or cross_train_counter % 6 == 0:  # temporary
                    cross_train_counter = 1
                    # print('xt', c_mem, '->', state[1], ':', t)
                    t += 1
                    c_mem = state[1]
                    c_t_annotation_map[type_map[t]] = c_mem
                    
                    
                    print('c_mem', c_mem)
                    print('type_map', type_map)
                    print('type_map[t]', type_map[t])
                    print('c_t_annotation_map', c_t_annotation_map)
                '''    
                    
                #EVO    
                type_list_2.append(type_map_2[state])
                    
                    
                    
                # print('\tt', c_mem, ':', t)
                
                '''
                type_list.append(type_map[t])

                
                
                if state[0] not in type_idx_map.keys():
                    # print('xc', state[0], '! in', type_idx_map.keys())
                    c += 1
                    type_idx_map[state[0]] = c
                    c_t_annotation_map[colors_map[type_idx_map[state[0]]]] = state[0]
                # print('\tc', state[0], ':', type_idx_map[state[0]])
                color_list.append(colors_map[type_idx_map[state[0]]])

                '''
                
                #EVO    
                color_list_2.append(colors_map_2[state])
            
            #print('type_list_2', type_list_2)
            #print('c_t_annotation_map', c_t_annotation_map)
            #print('color_list_2', color_list_2)
            
            

            for _x, _s, _t, _c in zip(xs, scores, type_list_2, color_list_2):

                if _t not in shown_t or _c not in shown_c:
                    shown_t.append(_t)
                    shown_c.append(_c)

                    # secondary_label_buffer.append(
                    plt.plot(_x, _s, marker=_t, c=_c,
                             markersize=8,
                         # label='%s, %s' % (c_t_annotation_map[_t], c_t_annotation_map[_c])
                         )
                    # )
                    # secondary_label_buffer_2.append('%s, %s' % (c_t_annotation_map[_t],
                    #                                                   c_t_annotation_map[_c]))

                else:
                    plt.plot(_x, _s, marker=_t, c=_c, markersize=8)

            # TODO: add the relative performance wrt competition as well as
            # the lane of the  disc.
                        
        
            

        
        # pprint(c_t_annotation_map)
        #if len(c_t_annotation_map.keys()) == 0: EVOOOOOOOOO
            # print('problem detected')
            #pass
        #plt.ylabel('FID or IS')
        plt.xlabel('encounter')

        legend_elements = []
        all_types = []
        all_colors = []

        
        #EVOOOOOOOOOOOOOOOOOOO
        c_t_annotation_map = {'k.' : 'No Adaptation', 'g*': 'Coadaptation', 'bo': 'Soft Sweeps', 'ro': 'Hard Sweeps'}
        
        
        type_map_3 = {'No adaptation': 'k.',
                      'Coadaptation' : 'g*',
                      'Soft Sweeps'  : 'bo',
                      'Hard Sweeps'  : 'ro'}
        
        
        for elt in type_map_3.values():
        
            if elt in c_t_annotation_map and elt not in all_types: #EVOOO
                all_types.append(elt)
                legend_elements.append(Line2D([0], [0],
                                  marker=elt[1], #elt
                                  color=elt[0], #'w'
                                  label=c_t_annotation_map[elt],
                                  markerfacecolor=elt[0], #'k'
                                  markersize=8))

        plt.legend(handles=legend_elements)
        
        
        
        
        
        
        #Not happening anyway, our legend maps are created manually and are constant as of now!
        '''
        if len(all_types) == 0:
            legend_elements.append(Line2D([0], [0],
                                  marker='x',
                                  color='w',
                                  label='base',
                                  markerfacecolor='k',
                                  markersize=10))

        
        for elt in colors_map.values():
            if elt in c_t_annotation_map.keys():
                # print('color legend elt:', elt)
                legend_elements.append(Line2D([0], [0],
                                  marker='s',
                                  color='w',
                                  label=c_t_annotation_map[elt],
                                  markerfacecolor=elt,
                                  markersize=10))

                
        '''     
        return soft_sweeps, hard_sweeps
    ##########################################################################################################################
    
        
        
               
    
        
    

    method_names = []
    gen_run_tags = []
    disc_run_tags = []

    
    #Collecting gen run tags
    for i, (key, value) in enumerate(gen_attribution_map.items()):
        if key[0] == 'brute-force':
            continue
        method_names.append(' '.join(key))
        gen_run_tags.append([])
        
        #print('key ', key)
        #print('value ', value)
        #print('attribution_map.items() ', attribution_map.items())
        
        
        
        for sub_key, (exec_time, fids, is_scores, gen_fitnesses, tags) in value.items(): #here we collect all tags
            gen_run_tags[-1].append(tags)
            #print('value.items() ', value.items())#EVO
            #print('sub_key ', sub_key)#EVO
            #print('exec_time ', exec_time)#EVO
            #print('fids ', fids)#EVO
            #print('is_scores ', is_scores)#EVO
            #print('tags ', tags)#EVO
            #print('fitnesses ', fitnesses)
    
    #print('run_tags: ', run_tags)#EVO

    
    
    #Collecting disc run tags
    for i, (key, value) in enumerate(disc_attribution_map.items()):
        if key[0] == 'brute-force':
            continue
        method_names.append(' '.join(key))
        disc_run_tags.append([])
        
        for sub_key, (exec_time, fids, is_scores, disc_fitnesses, tags) in value.items(): #here we collect all tags
            disc_run_tags[-1].append(tags)
            
            
            
    #print('gen_run_tags', gen_run_tags)        
    #print('')
    #print('disc_run_tags', disc_run_tags)
    #print('')
    
    gen_fit_master_tag_trace = []#EVO
    disc_fit_master_tag_trace = []#EVO
    
    
    fid_master_tag_trace = []
    is_master_tag_trace = []


    #IS, FID & Gen fitnesses
    ###########################################################################
    for method, tag_sets in zip(method_names, gen_run_tags):
        
        fit_local_tag_trace = []#EVO
        fid_local_tag_trace = []
        is_local_tag_trace = []

        #print(method)#EVO

        #print(tag_sets)#EVO

        for tag_set in tag_sets:  # we have the last tags performances
            
            fit_tag_set_trace = []#EVO
            fid_tag_set_trace = []
            is_tag_set_trace = []
            tag_set_disc_trace = []

            #print('tag_set', tag_set)#EVO

            for tag in tag_set:
                
                fit_single_tag_trace = []#EVO
                fid_single_tag_trace = []
                is_single_tag_trace = []
                temp_tag = tag
                

                while temp_tag in gen_tag_trace.keys():  # does not enter if the tag is wrong
                    
                    #print('temp_tag ', temp_tag)#EVO
                    
                    #print('gen_tag_trace ', gen_tag_trace)#EVO
                    #print('gen_tag_trace.keys() ', gen_tag_trace.keys())#EVO
                    
                    current_disc = gen_tag_trace[temp_tag][1]
                    
                    #print('current_disc ', current_disc)#EVO
                    
                    #print('encounter_record[current_disc, temp_tag] ', encounter_record[current_disc, temp_tag])#EVO
                    
                    fid_single_tag_trace.append([temp_tag,
                                             current_disc,
                                             *encounter_record[current_disc, temp_tag],
                                             master_fid_map[temp_tag]])  # and here we add all the fids (not only best ones from argument)
                    
                    is_single_tag_trace.append([temp_tag,
                                             current_disc,
                                             *encounter_record[current_disc, temp_tag],
                                             master_is_map[temp_tag]])
                    
                    #EVO -- think about this
                    fit_single_tag_trace.append([temp_tag,
                                             current_disc,
                                             *encounter_record[current_disc, temp_tag],
                                             master_gen_fit_map[temp_tag]])
                    
                    
                    temp_tag = gen_tag_trace[temp_tag][0]
                    
                    #print('fit_single_tag_trace ', fit_single_tag_trace)#EVO
                    #print('fid_single_tag_trace ', fid_single_tag_trace)#EVO
                    #print('is_single_tag_trace ', is_single_tag_trace)#EVO
                    #print('temp_tag ', temp_tag)#EVO
                
                fit_tag_set_trace.append(fit_single_tag_trace)
                fid_tag_set_trace.append(fid_single_tag_trace)
                is_tag_set_trace.append(is_single_tag_trace)
                
                #print('fit_tag_set_trace', fit_tag_set_trace)#EVO
                #print('fid_tag_set_trace', fid_tag_set_trace)#EVO
                #print('is_tag_set_trace', is_tag_set_trace)#EVO
                

                ###################
                single_disc_trace = []
                # print(temp_tag, tag)
                disc_trace_root = gen_tag_trace[tag][1]  # we have a problem where the
                # evolutionary elts that have not evolved in the last round are not used.
                temp_disc = disc_trace_root

                while temp_disc in disc_tag_trace.keys():
                    single_disc_trace.append(temp_disc)
                    temp_disc = disc_tag_trace[temp_disc][0]

                tag_set_disc_trace.append(single_disc_trace)
                
                ###################

            fit_local_tag_trace.append([fit_tag_set_trace, tag_set_disc_trace])
            fid_local_tag_trace.append([fid_tag_set_trace, tag_set_disc_trace])
            is_local_tag_trace.append([is_tag_set_trace, tag_set_disc_trace])
            
        gen_fit_master_tag_trace.append(fit_local_tag_trace)
        fid_master_tag_trace.append(fid_local_tag_trace)
        is_master_tag_trace.append(is_local_tag_trace)
    
    #EVO
    #print('gen_tag_trace', gen_tag_trace)
    #print("")
    #print("gen_fit_master_tag_trace: ", gen_fit_master_tag_trace)
    #print("")
    ###########################################################################
    
    
    
    
    
    #Discs fitnesses
    ###########################################################################
    for method, tag_sets in zip(method_names, disc_run_tags):
        
        fit_local_tag_trace = []#EVO

        #print(method)#EVO

        #print(tag_sets)#EVO

        for tag_set in tag_sets:  # we have the last tags performances
            
            fit_tag_set_trace = []#EVO
            
            
            ####
            tag_set_gen_trace = []

            
            
            #print('tag_set', tag_set)#EVO

            for tag in tag_set:
                
                fit_single_tag_trace = []#EVO

                temp_tag = tag
                

                while temp_tag in disc_tag_trace.keys():  # does not enter if the tag is wrong
                    
                    #print('temp_tag ', temp_tag)#EVO
                    
                    #print('gen_tag_trace ', gen_tag_trace)#EVO
                    #print('gen_tag_trace.keys() ', gen_tag_trace.keys())#EVO
                    
                    current_gen = disc_tag_trace[temp_tag][1]
                    
                    #print('current_disc ', current_disc)#EVO
                    
                    #print('encounter_record[current_disc, temp_tag] ', encounter_record[current_disc, temp_tag])#EVO
                    '''
                    fid_single_tag_trace.append([temp_tag,
                                             current_gen,
                                             *encounter_record[temp_tag, current_gen],
                                             master_fid_map[temp_tag]])  # and here we add all the fids (not only best ones from argument)
                    
                    is_single_tag_trace.append([temp_tag,
                                             current_gen,
                                             *encounter_record[temp_tag, current_gen],
                                             master_is_map[temp_tag]])
                    '''

                    fit_single_tag_trace.append([temp_tag,
                                             current_gen,
                                             *encounter_record[temp_tag, current_gen],
                                             master_disc_fit_map[temp_tag]])
                    
                    
                    temp_tag = disc_tag_trace[temp_tag][0]
                    
                    #print('fit_single_tag_trace ', fit_single_tag_trace)#EVO
                    #print('fid_single_tag_trace ', fid_single_tag_trace)#EVO
                    #print('is_single_tag_trace ', is_single_tag_trace)#EVO
                    #print('temp_tag ', temp_tag)#EVO
                
                fit_tag_set_trace.append(fit_single_tag_trace)
                #fid_tag_set_trace.append(fid_single_tag_trace)
                #is_tag_set_trace.append(is_single_tag_trace)
                
                #print('fit_tag_set_trace', fit_tag_set_trace)#EVO
                #print('fid_tag_set_trace', fid_tag_set_trace)#EVO
                #print('is_tag_set_trace', is_tag_set_trace)#EVO
                

                ###################
                single_gen_trace = []
                # print(temp_tag, tag)
                gen_trace_root = disc_tag_trace[tag][1]  # we have a problem where the
                # evolutionary elts that have not evolved in the last round are not used.
                temp_gen = gen_trace_root

                while temp_gen in gen_tag_trace.keys():
                    single_gen_trace.append(temp_gen)
                    temp_gen = gen_tag_trace[temp_gen][0]

                tag_set_gen_trace.append(single_gen_trace)

                ###################
                
            fit_local_tag_trace.append([fit_tag_set_trace, tag_set_gen_trace])
            #fid_local_tag_trace.append([fid_tag_set_trace, tag_set_gen_trace])
            #is_local_tag_trace.append([is_tag_set_trace, tag_set_gen_trace])
            
        disc_fit_master_tag_trace.append(fit_local_tag_trace)
        #fid_master_tag_trace.append(fid_local_tag_trace)
        #is_master_tag_trace.append(is_local_tag_trace)
    
    
    
    ###########################################################################
    
    
    
    
    
    
    #FID PROGRESSION PLOT
    for method, method_specific_tag_trace in zip(method_names, fid_master_tag_trace):
        for gen_tags_trace, disc_tags_trace in method_specific_tag_trace:
            
            #not used
            disc_idx = {}           

            for i, disc_line in enumerate(disc_tags_trace):
                disc_idx.update(dict((tag, i) for tag in disc_line))
            ####
            
            
            for gen_tag_line in gen_tags_trace:
                # print(gen_tag_line[0][0])
                for entry in reversed(gen_tag_line):
                    # print(entry)
                    # print('\t%s - %s \t %.2e \t %.2e \t %.2f' % tuple(entry))
                    pass
            
            #print('gen_tags_trace', gen_tags_trace)#EVO
            #print('disc_tags_trace', disc_tags_trace)#EVO
            
            render_score_progression(method, gen_tags_trace)
            plt.ylabel('FID')
            plt.savefig("./post_analysis_images/fid_progression.png")
            plt.clf()
    
    #IS PROGRESSION PLOT
    for method, method_specific_tag_trace in zip(method_names, is_master_tag_trace):
        for gen_tags_trace, disc_tags_trace in method_specific_tag_trace:
            
            #not used
            disc_idx = {}

            for i, disc_line in enumerate(disc_tags_trace):
                disc_idx.update(dict((tag, i) for tag in disc_line))
            ####
            
            for gen_tag_line in gen_tags_trace:
                # print(gen_tag_line[0][0])
                for entry in reversed(gen_tag_line):
                    # print(entry)
                    # print('\t%s - %s \t %.2e \t %.2e \t %.2f' % tuple(entry))
                    pass
            
            
            render_score_progression(method, gen_tags_trace)
            plt.ylabel('IS')
            plt.savefig("./post_analysis_images/is_progression.png")
            plt.clf()


         
    ##GENS FITNESS PROGRESSION PLOT
    for method, method_specific_tag_trace in zip(method_names, gen_fit_master_tag_trace):
        
        if method.startswith('chain evolve'):
            
            for gen_tags_trace, disc_tags_trace in method_specific_tag_trace:
                
                #not used
                disc_idx = {}

                for i, disc_line in enumerate(disc_tags_trace):
                    disc_idx.update(dict((tag, i) for tag in disc_line))
                ####
                
                for gen_tag_line in gen_tags_trace:
                    # print(gen_tag_line[0][0])
                    for entry in reversed(gen_tag_line):
                        # print(entry)
                        # print('\t%s - %s \t %.2e \t %.2e \t %.2f' % tuple(entry))
                        pass         

                #print('gen_tags_trace: ', gen_tags_trace)
                
                
                #print('gen_tags_trace ', gen_tags_trace)
                #print('')
                
                
                gen_soft_sweeps, gen_hard_sweeps = render_fitness_progression(method, gen_tags_trace, gen_sweeps)
                plt.title('Fitness Progression - Pathogens Evolving')
                plt.ylabel('FITNESS')
                plt.savefig("./post_analysis_images/gen_fit_progression.png")
                plt.clf()
    
    
    
    ##DISCS FITNESS PROGRESSION PLOT
    for method, method_specific_tag_trace in zip(method_names, disc_fit_master_tag_trace):
        
        if method.startswith('chain evolve'):
            
            for disc_tags_trace, gen_tags_trace in method_specific_tag_trace:
                
                #not used
                gen_idx = {}

                for i, gen_line in enumerate(gen_tags_trace):
                    gen_idx.update(dict((tag, i) for tag in gen_line))
                ####
                
                for disc_tag_line in disc_tags_trace:
                    # print(gen_tag_line[0][0])
                    for entry in reversed(disc_tag_line):
                        # print(entry)
                        # print('\t%s - %s \t %.2e \t %.2e \t %.2f' % tuple(entry))
                        pass         

                #print('gen_tags_trace: ', gen_tags_trace)
                
                
                #print('disc_tags_trace ', disc_tags_trace)
                #print('')
                
                
                disc_soft_sweeps, disc_hard_sweeps = render_fitness_progression(method, disc_tags_trace, disc_sweeps)
                plt.title('Fitness Progression - Hosts Evolving')
                plt.ylabel('FITNESS')
                plt.savefig("./post_analysis_images/disc_fit_progression.png")
                plt.clf()
                
    return gen_soft_sweeps, gen_hard_sweeps, disc_soft_sweeps, disc_hard_sweeps
                
            
if __name__ == "__main__":

    stitch_run_traces('.', _dataset)

    run_skip = []

    master_table = parse_past_runs(trace_dump_file)
    
    #EVO
    gen_attribution_map = defaultdict(dict)
    disc_attribution_map = defaultdict(dict)

    collector_list = []
    

    
    for i_1, entry in enumerate(master_table):
        # print(i_1, entry[0])
        if i_1 in run_skip:
            # print('skipping')
            continue
        # print('not skipping')
        for i_2, sub_entry in enumerate(entry[1:-1]):
            # print(sub_entry[0])
            if sub_entry[0][1] == 'brute-force':
                extracted_is, extracted_fids, final_random_tags, duration = extract_bruteforce_data(sub_entry)
            elif sub_entry[0][1] == 'chain evolve' \
                or sub_entry[0][1] == 'chain progression' \
                or sub_entry[0][1] == 'chain evolve fit reset' \
                or sub_entry[0][1] == 'deterministic base round robin' \
                or sub_entry[0][1] == 'deterministic base round robin 2' \
                or sub_entry[0][1] == 'stochastic base round robin' \
                or sub_entry[0][1] == 'homogenous chain progression':
                disc_fits, gen_fits, extracted_is, extracted_fids, final_disc_tags,\
                                                final_gen_tags, duration, disc_sweeps, gen_sweeps = extract_evo_data(sub_entry)
                
                                
            elif sub_entry[0][1] == 'matching from tags':
                gen_index, disc_index, real_error_matrix, gen_error_matrix = \
                    extract_battle_royale_data(sub_entry)
                continue
            else:
                # print(sub_entry[0])
                raise Exception('unknown selection structure: %s' % sub_entry[0][1])

            #print('attribution_map[(sub_entry[0][1], sub_entry[0][2], sub_entry[0][3])][sub_entry[0][-1]]',\
            #      attribution_map[(sub_entry[0][1], sub_entry[0][2], sub_entry[0][3])][sub_entry[0][-1]])
            
            #print('[duration, extracted_fids, final_random_tags]', [duration, extracted_fids, final_random_tags])
            
            
            
            
            #EVOOOs
            gen_attribution_map[(sub_entry[0][1], sub_entry[0][2], sub_entry[0][3])][sub_entry[0][-1]] =\
                [duration, extracted_fids, extracted_is, gen_fits, final_gen_tags]
            
            
            disc_attribution_map[(sub_entry[0][1], sub_entry[0][2], sub_entry[0][3])][sub_entry[0][-1]] =\
                [duration, extracted_fids, extracted_is, disc_fits, final_disc_tags]

            
            
            
            #print('attribution_map[(sub_entry[0][1], sub_entry[0][2], sub_entry[0][3])][sub_entry[0][-1]]',\
                 #attribution_map[(sub_entry[0][1], sub_entry[0][2], sub_entry[0][3])][sub_entry[0][-1]])
            
            print('\t', i_1, i_2 + 1, sub_entry[0])
            for i_3, sub_sub_entry in enumerate(sub_entry[1:-1]):
                print('\t\t', i_1, i_2 + 1, i_3 + 1, sub_sub_entry[0])
                print('\t\t', i_1, i_2 + 1, i_3 + 1, sub_sub_entry[-1])
            print('\t', i_1, i_2 + 1, sub_entry[-1])

        print(i_1, entry[-1])

    
    # pprint(collector_list)

    # pprint(dict(attribution_map))
    
    
    
    
    attribution_map_filter = [
        # ('brute-force', '5', '15'),
        # ('chain evolve', '3', '3'),
        # ('chain evolve', '3', '4'),
        # ('chain progression', '5', '5'),
        # ('chain evolve fit reset', '3', '3'),
        # ('stochastic base round robin', '5', '5'),
        # ('deterministic base round robin', '5', '5'),
        # ('deterministic base round robin 2', '5', '5'),
        # ('brute-force', '10', '15'),
        # ('brute-force', '5', '30'),
        # ('homogenous chain progression', '5', '5')
    ]

    _name_remap = {
        ('stochastic base round robin 5 5'): 'stochastic round robin',
        ('chain evolve 3 4'): 'evolution with\nheterogeneous\npopulation jumps',
        ('brute-force 10 15'): 'reference',
        ('chain progression 5 5'): 'round-robin with\nheterogeneous\npopulation jumps',
        ('deterministic base round robin 2 5 5'): 'standard round robin',
        ('homogenous chain progression 5 5'): 'round-robin with\npopulation jumps'
    }

    
    name_remap = lambda x: _name_remap[x] if x in _name_remap.keys() else x

    #EVOOOs
    for key in attribution_map_filter:
        if key in gen_attribution_map.keys():
            del gen_attribution_map[key]
            
    for key in attribution_map_filter:
        if key in disc_attribution_map.keys():
            del disc_attribution_map[key]

    
    # new_attribution_map = {}
    #
    # for old_name, new_name in name_remap.items():
    #     new_attribution_map[new_name] = attribution_map[old_name]
    #
    # attribution_map = new_attribution_map

    
    #render_fid_is_performances(attribution_map)
    #print('ALL PLOTS FINISHED SUCCESSFULLY')
    
    
    #Apart from the method_names, do not use outputs from this function before checking 
    #this function had never been used so was never adapted to our actual code
    method_names, best_fid_gen_tags, best_is_gen_tags, select_disc_fid_tags,\
                                                            select_disc_is_tags = pull_best_fid_is_tags(gen_attribution_map) 
    
    
    # print('select disc tags:', select_disc_tags)
    # print('best fid gen tags:', best_fid_gen_tags)
    # buffer_gen_disc_fid_tags(best_fid_gen_tags, select_disc_tags)

    # pprint(gen_index)

    # render_relative_performances(gen_index, disc_index, real_error_matrix, gen_error_matrix,
    #                               method_names, best_fid_gen_tags)

    # print(attribution_map.keys())

    gen_soft_sweeps, gen_hard_sweeps, disc_soft_sweeps, disc_hard_sweeps = render_training_history(method_names, disc_sweeps, gen_sweeps)
    
    
    #EVO
    render_fid_is_performances(gen_attribution_map, gen_soft_sweeps, gen_hard_sweeps, disc_soft_sweeps, disc_hard_sweeps)
    
    
    print('EVERYTHING COMPLETED SUCCESSFULLY')
