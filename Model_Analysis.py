import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt

import Two_Step_Environment as env
import Models as models


'''
These 3 functions will help create stay probility graph. The requirement is you must pass in a 
pandas df that has the following cols: "trial", "choice", "state", "reward", "high_state"
'''

def update_df(df):

    copy = df.copy()


    # Need to add some coluns to my_df before doing bar graph 

    # Next choice will give the next choice 
    copy["next_choice"] = copy["choice"].shift(-1)
    
    # Stay Will be true if curr choice and next choice are the same 
    copy["stay"] = (copy["choice"] == copy["next_choice"]).astype(int)
    


    # Now need to add a column that will tell me the transition was common or rare 

    # Start by letting every row be rare 
    copy["transition"] = "Rare"

    # Now include commons 
    copy.loc[((copy["choice"] == "Left") & (copy["state"] == "A")), "transition"] = "Common"
    copy.loc[((copy["choice"] == "Right") & (copy["state"] == "B")), "transition"] = "Common"

    return copy 

def Stay_Probability_Graph(updated_df, model_name = ""):

    # Now compute the stay probabilities 
    stay_probabilities = updated_df.groupby(["transition","reward"])[["stay"]].mean()

    # the Four groups are common reward, rare reward, common no reward, rare no reward
    common_reward = stay_probabilities.loc[("Common",1), "stay"]
    rare_reward = stay_probabilities.loc[("Rare",1), "stay"]
    common_no_reward = stay_probabilities.loc[("Common",0), "stay"]
    rare_no_reward = stay_probabilities.loc[("Rare",0), "stay"]


    # Now plot the bar graph 
    fig, ax = plt.subplots(1,1)
    ax.bar(
            ["C\nReward", "R\nReward", "C\nNo Reward","R\nNo Reward"],
            [common_reward, rare_reward, common_no_reward, rare_no_reward],
            color = ['orange', 'blue', 'orange','blue'])

    ax.set_ylabel("Stay Probability")
    ax.set_ylim(0,1)
    string = "Stay Probability Bar Graph for " + model_name
    ax.set_title(string)

    plt.close(fig)
    return fig 

# Only run this AFTER running the graph 
def Stay_Probability_Table(updated_df):
    stay_probabilities = updated_df.groupby(["transition","reward"])[["stay"]].mean()
    common_reward = stay_probabilities.loc[("Common",1), "stay"]
    rare_reward = stay_probabilities.loc[("Rare",1), "stay"]
    common_no_reward = stay_probabilities.loc[("Common",0), "stay"]
    rare_no_reward = stay_probabilities.loc[("Rare",0), "stay"]
    
    table = pd.DataFrame(stay_probabilities)
    
    return table 




'''
These 2 functions will help create the logistic regression coefficient bar graph
'''

def update_my_df(df):
    dummy = df.copy()

    # Add "correct" column, -1 or 1
    dummy["correct"] = -1
    dummy.loc[ ((dummy["high_state"] == "A") & (dummy["choice"] == "Left")), "correct"] = 1
    dummy.loc[ ((dummy["high_state"] == "B") & (dummy["choice"] == "Right")), "correct"] = 1

    # Add "outcome" column, -1 or 1 
    dummy["outcome"] = -1
    dummy.loc[ (dummy["reward"] == 1), "outcome" ] = 1

    # Add "transition_num" column, -1 or 1 (different from "tranisition")
    dummy["transition_num"] = -1 
    dummy.loc[ (dummy["transition"] == "Common"), "transition_num" ] = 1

    # Add "transition_outcome" column whic is "transition_num" x "outcome"
    dummy["transition_outcome"] = ( dummy["transition_num"] * dummy["outcome"])

    # Add "bias" column, gonna say all models have bias for "Left" atm
    dummy["bias"] = -1
    dummy.loc[ (dummy["choice"] == "Left"), "bias" ] = 1

    return dummy 


def perform_LR(dummy, model =""):
    
    # Logistic Regression 
    X = dummy[["correct","bias","transition_num","outcome","transition_outcome"]]
    y = dummy["stay"]

    LR = LogisticRegression(max_iter = 1000)
    LR.fit(X,y)
    coefficients = LR.coef_[0]

    names = ["Correct","Bias", "Transition", "Outcome","Transition x Outcome"]
    coefs = LR.coef_[0]

    fig, ax = plt.subplots(1,1)
    ax.bar(names, coefs)
    ax.axhline(y=0, color="black", linestyle="--")

    ax.set_ylabel("Regression Coefficient")
    s = "Regression Coefficient for " + model
    ax.set_title(s)
    ax.tick_params(axis="x", rotation=45)
    
    
    plt.close(fig)
    return coefficients, fig 




'''
Main model analysis functions!
'''


def Run_One_Model(selected_trials = 1000, selected_alpha = 0.5, selected_reversal = 0.025, selected_weight = 5, selected_pweight = 0, modelt = ""):

    # Phase 1: Initialize
    internal_records = []
    two_step = env.Naive_Two_Step()

    # Phase 2: Discern model
    if modelt == "mb":
        model = models.Model_Based_Value(alpha=selected_alpha,weight=selected_weight)
    elif modelt == "mf":
        model = models.Model_Free_Value(alpha=selected_alpha,weight=selected_weight)
    elif modelt == "mbi":
        model = models.Model_Based_Inference(reversal=selected_reversal,weight=selected_weight)
    elif modelt == "pmb":
        model = models.Model_Based_Value_Perseveration(alpha=selected_alpha,weight=selected_weight, pweight=selected_pweight)
    elif modelt == "pmf":
        model = models.Model_Free_Value_Perseveration(alpha=selected_alpha,weight=selected_weight,pweight=selected_pweight)
    elif modelt == "pmbi":
        model = models.Model_Based_Inference_Perseveration(reversal=selected_reversal, weight=selected_weight, pweight=selected_pweight)

    elif modelt == "mfi":
        model = models.Model_Free_Inference(reversal =selected_reversal, weight=selected_weight)
    elif modelt == "pmfi":
        model = models.Model_Free_Inference_Perseveration(reversal=selected_reversal, weight=selected_weight, pweight=selected_pweight)

    for trial in range(selected_trials):
        choice = model.action_choice()
        state, reward, high_state = two_step.a_trial(choice)

        if modelt in ["mb", "mbi"]:
            model.update(state, reward)
        elif modelt in ["mf"]:
            model.update(choice, state, reward)
        elif modelt in ["pmb","pmf","pmbi"]:
            model.update(choice, state, reward)

        elif modelt in ["mfi","pmfi"]:
            model.update(choice, reward)

        internal_records.append({
            "trial": trial,
            "choice": choice,
            "state": state,
            "reward": reward,
            "high_state": high_state 
        })

    return pd.DataFrame(internal_records)




def Run_Multiple_Models(
        model_type = "", number_mice = 20, trialS = 1000, alphA = 0.8, reversaL = 0.025, weightS = 4, pweightS = 1):
    
    local_mice_sp = []
    local_mice_coef = []

    for i in range(number_mice):
        df = Run_One_Model(selected_trials=trialS,selected_alpha=alphA, selected_reversal=reversaL, selected_weight=weightS, selected_pweight=pweightS, modelt=model_type)

        updated_df = update_df(df)
        sp_table = Stay_Probability_Table(updated_df)

        dummy = update_my_df(updated_df)
        coef, _ = perform_LR(dummy)

        curr_mouse_sp = {
            "Common Reward": sp_table.loc[("Common",1), "stay"],
            "Rare Reward": sp_table.loc[("Rare",1), "stay"],
            "Common No Reward": sp_table.loc[("Common",0), "stay"],
            "Rare No Reward": sp_table.loc[("Rare",0), "stay"]
        }

        curr_mouse_coef = {
            "Correct": float(coef[0]),
            "Bias": float(coef[1]),
            "Transition": float(coef[2]),
            "Outcome": float(coef[3]),
            "Transition x Outcome": float(coef[4])
        }

        local_mice_sp.append(curr_mouse_sp)
        local_mice_coef.append(curr_mouse_coef)
        
    sp_df = pd.DataFrame(local_mice_sp)
    coef_df = pd.DataFrame(local_mice_coef)

    sp_means = sp_df.mean()
    sp_sems = sp_df.sem()

    coef_means = coef_df.mean()
    coef_sems = coef_df.sem()

    fig1, ax1 = plt.subplots()

    ax1.bar(
        ["C\nReward","R\nReward","C\nNo Reward","R\nNo Reward"],
        sp_means.values,
        yerr=sp_sems.values,
        capsize=15,
        color=["orange","steelblue","orange","steelblue"]
    )

    ax1.set_ylabel("Stay Probability")
    ax1.set_ylim(0,1)

    title = {
        "mb":"Model Based Value",
        "mf":"Model Free Value",
        "mbi":"Model Based Inference",
        "pmb":"Model Based Value Perseveration",
        "pmf":"Model Free Value Perseveration",
        "pmbi":"Model Based Inference Perseveration",
        
        "mfi":"Model Free Inference",
        "pmfi":"Model Free Inference Perseveration"
    }
    ax1.set_title(f"{title[model_type]} ({number_mice} agents, {trialS} trials)")
    plt.close(fig1)

    fig2, ax2 = plt.subplots()

    ax2.bar(
        ["Correct","Bias","Transition","Outcome","Transition x Outcome"],
        coef_means.values,
        yerr=coef_sems.values,
        capsize=15,
        color="steelblue"
    )

    ax2.axhline(0, color="black", linestyle="--")
    ax2.set_ylabel("Regression Coefficient")
    ax2.tick_params(axis="x", rotation=45)
    ax2.set_title(f"{title[model_type]} ({number_mice} agents, {trialS} trials)")
    plt.close(fig2)

    return fig1, fig2