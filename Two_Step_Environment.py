import numpy as np


# FOR PERCEMT CORRECT CALCULATE BASED ON THE LAST 8 TRIALS 

# Two Step Environment 

# Last Updated: 15 June 2026 

# I need to add the BLOCK SWITCH in this task.
#       At first high reward state is chosen then block switch triggered once agent 
#       gets 70% corrrect rn between 5 - 20 drawn for additional trials in the block
#       before the contingency swithces ie high reward state is switched 

class Naive_Two_Step:
    """
    This class creates the two step environment described above. 

    Actions:
        Choose Left or Right 

    States:
        There are states A and B
    
    Transition from Actions to States:
        Left --> A = 0.8 (Common transition)
        Left --> B = 0.2 (Rare transition)

        Right --> A = 0.2 (Rare transition)
        Right --> B = 0.8 (Common Transition)

    Rewards:
        A is rewarded 0.9
        B is rewared 0.1
    """
    
    # Updated Constructor
    def __init__(self, high_state = "A"):
        self.p_common = 0.8                     # Common Transition is 0.8 
        self.high_state = high_state            # Either "A" or "B"

        self.switch_criterion = False           # This is whether or not agent has 70% correct
        self.trials_until_switch = None         # How many trials before Block Switch


        # Admittedly, I do not know what exactly is the crtieria for 70% ...
        self.correct_trials = []                # To Track how many Trials are Correct
        self.window = 8                         # I will just leave it at last 8 Trials 


    # Not Changed 
    def transition(self, choice):
        """
        Given the first step choice (Left or Right) what state (A or B) is reached 

        Input:
            Choice, of type "str" and needs to be "Right" or "Left"

        Output:
            State, of type "str" and needs to be "A" or "B"
        """

        dice = np.random.rand()         # rn       
        
        if choice == "Left":
            if dice < self.p_common:    # If rn is below common transition probability, go Right 
                return "A"
            else:
                return "B"              # If rn is above common transition probability, go Left 
            
        elif choice == "Right":
            if dice < self.p_common:
                return "B"
            else:
                return "A"
        
        else:
            raise ValueError("Choise needs to be given as 'Left' or 'Right'")

    
    # Was Updated 
    def reward(self, state):
        """
        Given the current state, what reward will I get 

        Input:
            State of type "str" and needs to be "A" "B"
        Output:
            Reward of type "int" where 1 will mean reward was delivered 
        """

        if self.high_state == "A":                      # Now defining these locally 
            p_reward_A = 0.9    
            p_reward_B = 0.1
        else:
            p_reward_A = 0.1
            p_reward_B = 0.9 

        dice = np.random.rand()                         # rn
        
        if state == "A":                                # If rn is below reward probability, give it
            return(int(dice < p_reward_A))
        elif state == "B":
            return(int(dice < p_reward_B))
        else:
            raise ValueError("State needs to be given as 'A' or 'B'")
        

    # Need to Update 
    def a_trial(self, choice):
        """
        Given a chose of "Left" or "Right," simulate a trial that culminates in a State and a Reward

        Input:
            Choice of type "str" and needs to be "Left" or "Right"

        Output:
            State as "str" and Reward as "int"
        """
        state = self.transition(choice)
        reward = self.reward(state)


        # Now begin to implement that 70% Criteria 

        # We will denote whether trial was "Correct" based on action given high state 
        if self.high_state == "A":
            is_correct = int(choice == "Left")
        else:
            is_correct = int(choice == "Right")

        # Now append the value to our records 
        self.correct_trials.append(is_correct)

        # Now Checking whether we can start thinking about Block Switch
        if not self.switch_criterion:
            if len(self.correct_trials) >= self.window:
                curr_accuracy = np.mean(self.correct_trials[-self.window:])      # check last 8

                if curr_accuracy >= 0.7:
                    self.switch_criterion = True        # Now Start Block Switch!
                    self.trials_until_switch = np.random.randint(5,21)      # draw between 5 and 20

        else:
            self.trials_until_switch -= 1               # decrement

            if self.trials_until_switch == 0:           # if done with these trials

                if self.high_state == "A":
                    self.high_state = "B"
                else:
                    self.high_state = "A"

                # Now reset everything ... 
                self.switch_criterion = False
                self.trials_until_switch = None
                self.correct_trials = []


        # return self.high_state for debugging

        return state, reward, self.high_state