from multiprocessing import Pool

import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt 
from scipy.optimize import minimize

import Models as models
import Model_Analysis as analysis

#NLL Func
def neg_log_likelihood(parameters, df, model):
    """
    NLL function same idea as before. now param has an exrtra term so looks either like
    [alpha beta pbeta] for mb/mf or [reversal beta pbeta] for bi

    for non preservation is [alpha beta] or [reversal beta]]
    """
    
    if model == "pmb":
        curr = models.Model_Based_Value_Perseveration(alpha=parameters[0],weight=parameters[1],pweight=parameters[2])
    elif model == "pmf":
        curr = models.Model_Free_Value_Perseveration(alpha=parameters[0], weight=parameters[1],pweight=parameters[2])
    elif model == "pmbi":
        curr = models.Model_Based_Inference_Perseveration(reversal=parameters[0],weight=parameters[1],pweight=parameters[2])
    
    elif model == "mb":
        curr = models.Model_Based_Value(alpha=parameters[0], weight=parameters[1])
    elif model == "mf":
        curr = models.Model_Free_Value(alpha=parameters[0], weight=parameters[1])
    elif model == "mbi":
        curr = models.Model_Based_Inference(reversal=parameters[0],weight=parameters[1])

    elif model == "mfi":
        curr = models.Model_Free_Inference(reversal=parameters[0],weight=parameters[1])
    elif model == "pmfi":
        curr = models.Model_Free_Inference_Perseveration(reversal=parameters[0],weight=parameters[1],pweight=parameters[2])

    LL = 0

    choices = df["choice"].values
    states = df["state"].values
    rewards = df["reward"].values

    for i in range(len(df)):
        p_left = curr.p_left()                  

        actual_choice = choices[i]
        if actual_choice == "Left":                    
            LL += np.log(p_left)
        else:                                           
            LL += np.log(1-p_left)

        choice = choices[i]
        state = states[i]
        reward = rewards[i]

        if model == "pmb" or model == "pmf" or model == "pmbi" or model == "mf":
            curr.update(choice, state, reward)
        elif model == "mb" or model == "mbi":
            curr.update(state,reward)

        elif model == "mfi" or model == "pmfi":
            curr.update(choice, reward)    
        
    return ( LL * -1 )




#BIC Func
def give_me_BIC(df, model):
    
    if model in ["mb", "mf"]:

        min_param = minimize(
            neg_log_likelihood,
            x0 = [ np.random.rand(), np.random.uniform(0,10) ],                       
            args = ( df, model ),
            bounds = [ (0.001, 1 ),
                       (0.01, 10 )],
            options={"maxiter":50}
        )
        k=2


    elif model in ["mbi", "mfi"]:

        min_param = minimize(
            neg_log_likelihood,
            x0 = [ np.random.uniform(0.001,0.2), np.random.uniform(0,10) ],                       
            args = ( df, model ),
            bounds = [ (0.001, 0.2 ),
                       (0.01, 10 )],
            options={"maxiter":50}
        )
        k=2


    elif model in ["pmb", "pmf"]: 

        min_param = minimize(
            neg_log_likelihood,
            x0 = [ np.random.rand(), np.random.uniform(0,10), np.random.uniform(0,10) ],                       
            args = ( df, model ),
            bounds = [ (0.001, 1 ),
                       (0.01, 10 ),
                       (0.0, 10 )],
            options={"maxiter":50}
        )
        k=3


    elif model in ["pmbi", "pmfi"]:

        min_param = minimize(
            neg_log_likelihood,
            x0 = [ np.random.uniform(0.001, 0.2), np.random.uniform(0,10), np.random.uniform(0,10) ],                       
            args = ( df, model ),
            bounds = [ (0.001, 0.2 ),
                       (0.01, 10 ),
                       (0.0, 10 )],
            options={"maxiter":50}
        )
        k=3

    
    LL = -min_param.fun 
    n = len(df) 
    return [ ( -2 * LL + k * np.log(n) ) , model ]




# Most important func, for given dataset compute the BIC for all 
# Use this one to reproduce 8x8 matrix for all current models 
def give_me_winner(df):
    mb_BIC = give_me_BIC(df, "mb")
    mf_BIC = give_me_BIC(df, "mf")
    mbi_BIC = give_me_BIC(df, "mbi")
    pmb_BIC = give_me_BIC(df, "pmb")
    pmf_BIC = give_me_BIC(df, "pmf")
    pmbi_BIC = give_me_BIC(df, "pmbi")

    mfi_BIC = give_me_BIC(df, "mfi")
    pmfi_BIC = give_me_BIC(df, "pmfi")

    # dBIC = np.array( [ mb_BIC[0], mf_BIC[0], bi_BIC[0], pmb_BIC[0], pmf_BIC[0], pbi_BIC[0], mfi_BIC[0], pmfi_BIC[0] ] )
    dBIC = np.array( [ mb_BIC[0], pmb_BIC[0], mf_BIC[0], pmf_BIC[0], mbi_BIC[0], pmbi_BIC[0], mfi_BIC[0], pmfi_BIC[0] ] )
    dBIC = dBIC - np.min(dBIC)

    return min( [mb_BIC, mf_BIC, mbi_BIC, pmb_BIC, pmf_BIC, pmbi_BIC, mfi_BIC, pmfi_BIC], key = lambda x:x[0])   , dBIC

# alternatives 

# Use this one to reproduce the 4x4 matrix for models w/o perseveration
def give_me_winner_no_pers(df):
    mb_BIC = give_me_BIC(df, "mb")
    mf_BIC = give_me_BIC(df, "mf")
    mbi_BIC = give_me_BIC(df, "mbi")

    mfi_BIC = give_me_BIC(df, "mfi")

    dBIC = np.array( [ mb_BIC[0], mf_BIC[0], mbi_BIC[0], mfi_BIC[0] ] )
    dBIC = dBIC - np.min(dBIC)

    return min( [mb_BIC, mbi_BIC, mf_BIC, mfi_BIC],key=lambda x: x[0] )   , dBIC

# Use this one to reproduce the 4x4 matrix for models w perseveration
def give_me_winner_pers(df):
    pmb_BIC = give_me_BIC(df, "pmb")
    pmf_BIC = give_me_BIC(df, "pmf")
    pmbi_BIC = give_me_BIC(df, "pmbi")

    pmfi_BIC = give_me_BIC(df, "pmfi")

    dBIC = np.array( [ pmb_BIC[0], pmf_BIC[0], pmbi_BIC[0], pmfi_BIC[0] ] )
    dBIC = dBIC - np.min(dBIC)

    return min( [pmb_BIC, pmbi_BIC, pmf_BIC, pmfi_BIC], key=lambda x: x[0] )   , dBIC




# Model recovery funcs, this is what you call. 

def one_mb_recovery(i):
    curr_df = analysis.Run_One_Model(
        selected_trials=1000,
        selected_alpha=np.random.rand(),
        selected_weight=np.random.uniform(0,10),
        modelt="mb"
    )

    curr_dict, dBIC = give_me_winner(curr_df)
    # curr_dict, dBIC = give_me_winner_no_pers(curr_df)
    return curr_dict[1], dBIC

def one_mf_recovery(i):
    curr_df = analysis.Run_One_Model(
        selected_trials=1000,
        selected_alpha=np.random.rand(),
        selected_weight=np.random.uniform(0,10),
        modelt="mf"
    )

    curr_dict, dBIC = give_me_winner(curr_df)
    # curr_dict, dBIC = give_me_winner_no_pers(curr_df)
    return curr_dict[1], dBIC

def one_mbi_recovery(i):
    curr_df = analysis.Run_One_Model(
        selected_trials=1000,
        selected_reversal=np.random.uniform(0.001, 0.2),
        selected_weight=np.random.uniform(0,10),
        modelt="mbi"
    )

    curr_dict, dBIC = give_me_winner(curr_df)
    # curr_dict, dBIC = give_me_winner_no_pers(curr_df)
    return curr_dict[1], dBIC

def one_pmb_recovery(i):

    curr_df = analysis.Run_One_Model(
        selected_trials=1000,
        selected_alpha=np.random.rand(),
        selected_weight=np.random.uniform(0,10),
        selected_pweight=np.random.uniform(0,10),
        modelt="pmb"
    )

    curr_dict, dBIC = give_me_winner(curr_df)
    # curr_dict, dBIC = give_me_winner_pers(curr_df)
    return curr_dict[1], dBIC

def one_pmf_recovery(i):
    curr_df = analysis.Run_One_Model(
        selected_trials=1000,
        selected_alpha=np.random.rand(),
        selected_weight=np.random.uniform(0,10),
        selected_pweight=np.random.uniform(0,10),
        modelt="pmf"
    )

    curr_dict, dBIC = give_me_winner(curr_df)
    # curr_dict, dBIC = give_me_winner_pers(curr_df)
    return curr_dict[1], dBIC

def one_pmbi_recovery(i):
    curr_df = analysis.Run_One_Model(
        selected_trials=1000,
        selected_reversal=np.random.uniform(0.001, 0.2),             # np.random.rand()
        selected_weight=np.random.uniform(0,10),
        selected_pweight=np.random.uniform(0,10),
        modelt="pmbi"
    )

    curr_dict, dBIC = give_me_winner(curr_df)
    # curr_dict, dBIC = give_me_winner_pers(curr_df)
    return curr_dict[1], dBIC

def one_mfi_recovery(i):
    curr_df = analysis.Run_One_Model(
        selected_trials=1000,
        selected_reversal=np.random.uniform(0.001,0.2),
        selected_weight=np.random.uniform(0,10),
        modelt="mfi"
    )

    curr_dict, dBIC = give_me_winner(curr_df)
    return curr_dict[1], dBIC

def one_pmfi_recovery(i):
    curr_df = analysis.Run_One_Model(
        selected_trials=1000,
        selected_reversal=np.random.uniform(0.001,0.2),
        selected_weight=np.random.uniform(0,10),
        selected_pweight=np.random.uniform(0,10),
        modelt="pmfi"
    )

    curr_dict, dBIC = give_me_winner(curr_df)
    return curr_dict[1], dBIC




# main for running model recovery on simulated agents 
if __name__ == "__main__":

    with Pool(processes=8) as p:
        results = p.map(one_pmfi_recovery, range(1000))

    winners = [x[0] for x in results]

    with open("records.txt", "a") as f:
        # f.write("8x8 full model recovery :)\n")
        f.write('\n')
        f.write("pmfi row: mb pmb mf pmf mbi pmbi mfi pmfi" + '\n')
        f.write(
            str(winners.count("mb")) + " " + 
            str(winners.count("pmb")) + " " + 
            str(winners.count("mf")) + " " + 
            str(winners.count("pmf")) + " " + 
            str(winners.count("mbi")) + " " + 
            str(winners.count("pmbi")) + " " + 
            str(winners.count("mfi")) + " " + 
            str(winners.count("pmfi")) + '\n')




# main for running model recovery on neural network data
''' 
if __name__ == "__main__":

    all_dBIC = []
    winners = []

    for run in range(12):

        df = pd.read_csv(f"run_{run}_prelesion.csv")

        winner, dBIC = give_me_winner(df)

        winners.append(winner[1])
        all_dBIC.append(dBIC)

        print(f"run_{run}")
        print("Winner:", winner[1])
        print("dBIC:", dBIC)
        print()

    all_dBIC = np.array(all_dBIC)

    print("12 x 8 matrix:")
    print(all_dBIC)

    print("Winners:")
    print(winners)
 '''

