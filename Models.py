import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt


'''
Models without perseveration term
'''

class Model_Based_Value:
    
    # Constructor
    def __init__(self, alpha = 0.5, weight = 1):
        self.alpha = alpha

        self.weight = weight                        # Weight 
        
        # Prior is nothing 
        self.V_A = 0
        self.V_B = 0

    def compute_Q(self):
        """
        Compute Q_Left and Q_Right
        """
        Q_Left = (0.8 * self.V_A + 0.2 * self.V_B)
        Q_Right = (0.2 * self.V_A + 0.8 * self.V_B)

        return Q_Left, Q_Right

    def action_choice(self): 
        """
        Based on current Q_Left and Q_Right, choose action "Left" or "Right"
        """
        Q_Left, Q_Right = self.compute_Q()  # Compute Q based on V

        # Softmax, for right now I will just assume that Q_net = Q_MB

        # Probability of going left (If I add a weight ie w = 5, I get cleaner results)
        p_left = np.exp(self.weight * Q_Left) / (np.exp(self.weight * Q_Left) + (np.exp(self.weight * Q_Right)))

        dice = np.random.rand()         # rn
        if dice < p_left:
            return "Left"
        else:
            return "Right"

    def update(self, state, reward): 
        """
        Update V_A and V_B based on State and Reward
        """

        if state == "A":
            self.V_A = (1-self.alpha) * self.V_A + self.alpha * reward
        elif state == "B":
            self.V_B = (1-self.alpha) * self.V_B + self.alpha * reward

    def p_left(self):
        """
        Compute prob of choosing left 
        """

        Q_Left, Q_Right = self.compute_Q()
        p_left = np.exp(self.weight * Q_Left) / (  (np.exp(self.weight * Q_Left)) + (np.exp(self.weight * Q_Right))  )
        return p_left




class Model_Free_Value:

    # Constructor
    def __init__(self, alpha = 0.5,  weight = 1):          
        
        self.alpha = alpha                  # Learning Rate 
        self.lambdas = 1              # Eligibility Trace Parameter 

        self.weight = weight                # Weight 

        self.V_A = 0
        self.V_B = 0

        self.Q_Left = 0
        self.Q_Right = 0

    def action_choice(self):
        """
        Choose action "Left" or "Right" Based on current parameters 
        """
        Q_Left = self.Q_Left
        Q_Right = self.Q_Right

        p_left = np.exp(self.weight * Q_Left) / (np.exp(self.weight * Q_Left) + (np.exp(self.weight * Q_Right)))

        dice = np.random.rand()         # rn 
        if dice < p_left:
            return "Left"
        else:
            return "Right"

    def update(self, choice, state, reward):
        """
        Update V_A, V_B, Q_Left, and Q_Right
        """

        # Now for Updating Choice Action values need to consider the lambda 

        if state == "A":
            curr_V = self.V_A 
        elif state == "B":
            curr_V = self.V_B
        
        end_term = ((1-self.lambdas) * curr_V + self.lambdas * reward)

        if choice == "Left":
            self.Q_Left = (1-self.alpha) * self.Q_Left + self.alpha * end_term
        elif choice == "Right":
            self.Q_Right = (1-self.alpha) * self.Q_Right + self.alpha * end_term


        # Now updating state values 
        if state == "A":
            self.V_A = (1-self.alpha) * self.V_A + self.alpha * reward
        elif state == "B":
            self.V_B = (1-self.alpha) * self.V_B + self.alpha * reward

    def p_left(self):
        """
        Compute prob of choosing left 
        """

        p_left = np.exp(self.weight * self.Q_Left) / (  (np.exp(self.weight * self.Q_Left)) + (np.exp(self.weight * self.Q_Right))  )
        return p_left




class Model_Based_Inference:

    def __init__(self, reversal = 0.025, weight = 1):
        """
        Constructor
        up_good means State A is the curr high reward state
        """

        self.weight = weight                    # Weight
        self.reversal = reversal                # the reversal rate  

        self.P_up_good = 0.5                    # Probability up is good 

    def action_choice(self):
        """
        Choose action either Left or Right
        """

        P_down_good = 1 - self.P_up_good        # Get opposite 

        # Computing state values
        V_A = 0.9 * self.P_up_good + 0.1 * P_down_good
        V_B = 0.1 * self.P_up_good + 0.9 * P_down_good

        # Computing action values now 
        Q_Left = 0.8 * V_A + 0.2 * V_B
        Q_Right = 0.2 * V_A + 0.8 * V_B

        p_left = np.exp(self.weight * Q_Left) / (np.exp(self.weight * Q_Left) + (np.exp(self.weight * Q_Right)))

        dice = np.random.rand()                 # rn 
        if dice < p_left:
            return "Left"
        else:
            return "Right"
        
    def update(self, state, reward):
        """
        The parameter we need to update will be self.P_up_good
        """

        # First lets get p_rs_up_good and p_rs_down_good 

        if state == "A":

            if reward == 1:
                p_rs_up_good = 0.9
                p_rs_down_good = 0.1
            else:
                p_rs_up_good = 0.1
                p_rs_down_good = 0.9
        
        elif state == "B":
            
            if reward == 1:
                p_rs_up_good = 0.1
                p_rs_down_good = 0.9
            else:
                p_rs_up_good = 0.9
                p_rs_down_good = 0.1

        # We already have P_up_good and P_down_good implicitly 

        self.P_up_good = ( p_rs_up_good * self.P_up_good) / ((p_rs_up_good * self.P_up_good) + (p_rs_down_good * (1 - self.P_up_good)))

        # Update with the reversal
        self.P_up_good = ( (1 - self.reversal) * self.P_up_good + self.reversal * (1 - self.P_up_good) )

    def p_left(self):
        """
        Compute prob of choosing left 
        """
        P_down_good = 1 - self.P_up_good

        V_A = 0.9 * self.P_up_good + 0.1 * P_down_good
        V_B = 0.1 * self.P_up_good + 0.9 * P_down_good

        # Computing action values now 
        Q_Left = 0.8 * V_A + 0.2 * V_B
        Q_Right = 0.2 * V_A + 0.8 * V_B

        p_left = np.exp(self.weight * Q_Left) / (np.exp(self.weight * Q_Left) + (np.exp(self.weight * Q_Right)))
        return p_left




class Model_Free_Inference:

    def __init__(self, reversal = 0.025, weight = 1):
        self.weight = weight 
        self.reversal = reversal

        self.P_left_good = 0.5 

    def action_choice(self):

        p_left = self.p_left()

        dice = np.random.rand() 

        if dice < p_left:
            return "Left"
        else:
            return "Right"

    def update(self, choice, reward):

        if choice == "Left":
            if reward == 1:
                p_obs_left_good = 0.74 
                p_obs_right_good = 0.26
            else:
                p_obs_left_good = 0.26
                p_obs_right_good = 0.74
        elif choice == "Right":
            if reward == 1:
                p_obs_left_good = 0.26
                p_obs_right_good = 0.74
            else:
                p_obs_left_good = 0.74
                p_obs_right_good = 0.26

        self.P_left_good = (p_obs_left_good * self.P_left_good) / ( (p_obs_left_good * self.P_left_good) + (p_obs_right_good * (1 - self.P_left_good)) )
        self.P_left_good = ( (1 - self.reversal) * self.P_left_good + self.reversal * (1 - self.P_left_good))

    def p_left(self):
        P_right_good = 1 - self.P_left_good 

        Q_left = 0.74 * self.P_left_good + 0.26 * P_right_good
        Q_right = 0.26 * self.P_left_good + 0.74 * P_right_good 

        p_left = np.exp(self.weight * Q_left) / ( np.exp(self.weight * Q_left) + np.exp(self.weight * Q_right))

        return p_left


'''
Models with perseveration term
'''

class Model_Based_Value_Perseveration:
    
    def __init__(self, alpha = 0.5, weight = 1, pweight = 0):
        self.alpha = alpha                          # alpha 
        self.weight = weight                        # Weight 
        self.pweight = pweight                      # pweight 
        
        # Prior is nothing 
        self.V_A = 0
        self.V_B = 0

        self.prev_prev_choice = None
        self.previous_choice = None 

       
    def compute_Q(self):
        """
        Compute Q_Left and Q_Right
        """
        Q_Left = (0.8 * self.V_A + 0.2 * self.V_B)
        Q_Right = (0.2 * self.V_A + 0.8 * self.V_B)

        return Q_Left, Q_Right

    
    def action_choice(self): 
        """
        Based on current Q_Left and Q_Right, choose action "Left" or "Right"
        """
        Q_Left, Q_Right = self.compute_Q()  # Compute Q based on V

        QL_p = 0 
        QR_p = 0
        if self.prev_prev_choice == "Left" and self.previous_choice == "Left":
            QL_p = 1
        elif self.prev_prev_choice == "Right" and self.previous_choice == "Right":
            QR_p = 1 

        Q_left_net = self.weight * Q_Left + self.pweight * QL_p
        Q_right_net = self.weight * Q_Right + self.pweight * QR_p 

        p_left = np.exp(Q_left_net) / ( np.exp(Q_left_net) + np.exp(Q_right_net) )

        dice = np.random.rand()         # rn
        if dice < p_left:
            return "Left"
        else:
            return "Right"


    def update(self, choice, state, reward): 
        """
        Update V_A and V_B based on State and Reward
        """

        if state == "A":
            self.V_A = (1-self.alpha) * self.V_A + self.alpha * reward
        elif state == "B":
            self.V_B = (1-self.alpha) * self.V_B + self.alpha * reward
        
        self.prev_prev_choice = self.previous_choice 
        self.previous_choice = choice


    def p_left(self):
        """
        Compute prob of choosing left 
        """

        Q_Left, Q_Right = self.compute_Q()
        QL_p = 0 
        QR_p = 0

        if self.prev_prev_choice == "Left" and self.previous_choice == "Left":
            QL_p = 1
        elif self.prev_prev_choice == "Right" and self.previous_choice == "Right":
            QR_p = 1 

        Q_left_net = self.weight * Q_Left + self.pweight * QL_p
        Q_right_net = self.weight * Q_Right + self.pweight * QR_p 

        # Probability of going left (If I add a weight ie w = 5, I get cleaner results)
        p_left = np.exp(Q_left_net) / ( np.exp(Q_left_net) + np.exp(Q_right_net) )
        return p_left
    



class Model_Free_Value_Perseveration:

    def __init__(self, alpha = 0.5,  weight = 1, pweight = 0):          
        
        self.alpha = alpha                  # Learning Rate 
        self.weight = weight                # Weight 
        self.pweight = pweight               # perseveration weight 

        self.lambdas = 1                    # Eligibility Trace Parameter 

        self.V_A = 0
        self.V_B = 0

        self.Q_Left = 0
        self.Q_Right = 0

        self.prev_prev_choice = None
        self.previous_choice = None 


    def action_choice(self):
        """
        Choose action "Left" or "Right" Based on current parameters 
        """
        Q_Left = self.Q_Left
        Q_Right = self.Q_Right

        QL_p = 0 
        QR_p = 0
        if self.prev_prev_choice == "Left" and self.previous_choice == "Left":
            QL_p = 1
        elif self.prev_prev_choice == "Right" and self.previous_choice == "Right":
            QR_p = 1 

        Q_left_net = self.weight * Q_Left + self.pweight * QL_p
        Q_right_net = self.weight * Q_Right + self.pweight * QR_p 

        p_left = np.exp(Q_left_net) / ( np.exp(Q_left_net) + np.exp(Q_right_net) )

        dice = np.random.rand()         # rn 
        if dice < p_left:
            return "Left"
        else:
            return "Right"


    def update(self, choice, state, reward):
        """
        Update V_A, V_B, Q_Left, and Q_Right
        """

        # Now for Updating Choice Action values need to consider the lambda 

        if state == "A":
            curr_V = self.V_A 
        elif state == "B":
            curr_V = self.V_B
        
        end_term = ((1-self.lambdas) * curr_V + self.lambdas * reward)

        if choice == "Left":
            self.Q_Left = (1-self.alpha) * self.Q_Left + self.alpha * end_term
        elif choice == "Right":
            self.Q_Right = (1-self.alpha) * self.Q_Right + self.alpha * end_term


        # Now updating state values 
        if state == "A":
            self.V_A = (1-self.alpha) * self.V_A + self.alpha * reward
        elif state == "B":
            self.V_B = (1-self.alpha) * self.V_B + self.alpha * reward

        self.prev_prev_choice = self.previous_choice 
        self.previous_choice = choice


    def p_left(self):
        """
        Compute prob of choosing left 
        """

        Q_Left = self.Q_Left
        Q_Right = self.Q_Right

        QL_p = 0 
        QR_p = 0
        if self.prev_prev_choice == "Left" and self.previous_choice == "Left":
            QL_p = 1
        elif self.prev_prev_choice == "Right" and self.previous_choice == "Right":
            QR_p = 1 

        Q_left_net = self.weight * Q_Left + self.pweight * QL_p
        Q_right_net = self.weight * Q_Right + self.pweight * QR_p 

        p_left = np.exp(Q_left_net) / ( np.exp(Q_left_net) + np.exp(Q_right_net) )     
        return p_left
    



class Model_Based_Inference_Perseveration:

    def __init__(self, reversal = 0.025, weight = 1, pweight = 0 ):

        self.weight = weight                    # Weight
        self.reversal = reversal                # the reversal rate  
        self.pweight = pweight

        self.prev_prev_choice = None
        self.previous_choice = None 

        self.P_up_good = 0.5                    # Probability up is good 


    def action_choice(self):
        """
        Choose action either Left or Right
        """

        P_down_good = 1 - self.P_up_good        # Get opposite 

        # Computing state values
        V_A = 0.9 * self.P_up_good + 0.1 * P_down_good
        V_B = 0.1 * self.P_up_good + 0.9 * P_down_good

        # Computing action values now 
        Q_Left = 0.8 * V_A + 0.2 * V_B
        Q_Right = 0.2 * V_A + 0.8 * V_B

        
        QL_p = 0 
        QR_p = 0

        if self.prev_prev_choice == "Left" and self.previous_choice == "Left":
            QL_p = 1
        elif self.prev_prev_choice == "Right" and self.previous_choice == "Right":
            QR_p = 1 

        Q_left_net = self.weight * Q_Left + self.pweight * QL_p
        Q_right_net = self.weight * Q_Right + self.pweight * QR_p 

        p_left = np.exp(Q_left_net) / ( np.exp(Q_left_net) + np.exp(Q_right_net) )


        dice = np.random.rand()                 # rn 
        if dice < p_left:
            return "Left"
        else:
            return "Right"
        

    def update(self, choice, state, reward):
        """
        The parameter we need to update will be self.P_up_good
        """

        # First lets get p_rs_up_good and p_rs_down_good 

        if state == "A":

            if reward == 1:
                p_rs_up_good = 0.9
                p_rs_down_good = 0.1
            else:
                p_rs_up_good = 0.1
                p_rs_down_good = 0.9
        
        elif state == "B":
            
            if reward == 1:
                p_rs_up_good = 0.1
                p_rs_down_good = 0.9
            else:
                p_rs_up_good = 0.9
                p_rs_down_good = 0.1

        # We already have P_up_good and P_down_good implicitly 

        self.P_up_good = ( p_rs_up_good * self.P_up_good) / ((p_rs_up_good * self.P_up_good) + (p_rs_down_good * (1 - self.P_up_good)))

        # Update with the reversal
        self.P_up_good = ( (1 - self.reversal) * self.P_up_good + self.reversal * (1 - self.P_up_good) )


        self.prev_prev_choice = self.previous_choice 
        self.previous_choice = choice

    
    def p_left(self):
        """
        Compute prob of choosing left 
        """
        P_down_good = 1 - self.P_up_good        # Get opposite 

        # Computing state values
        V_A = 0.9 * self.P_up_good + 0.1 * P_down_good
        V_B = 0.1 * self.P_up_good + 0.9 * P_down_good

        Q_Left = 0.8 * V_A + 0.2 * V_B
        Q_Right = 0.2 * V_A + 0.8 * V_B

        
        QL_p = 0 
        QR_p = 0

        if self.prev_prev_choice == "Left" and self.previous_choice == "Left":
            QL_p = 1
        elif self.prev_prev_choice == "Right" and self.previous_choice == "Right":
            QR_p = 1 

        Q_left_net = self.weight * Q_Left + self.pweight * QL_p
        Q_right_net = self.weight * Q_Right + self.pweight * QR_p 

        p_left = np.exp(Q_left_net) / ( np.exp(Q_left_net) + np.exp(Q_right_net) )

        return p_left




class Model_Free_Inference_Perseveration:
    def __init__(self, reversal = 0.025, weight = 1 , pweight = 0):
        self.weight = weight
        self.reversal = reversal
        self.pweight = pweight

        self.prev_prev_choice = None
        self.previous_choice = None

        self.P_left_good = 0.5 

    def action_choice(self):
        p_left = self.p_left()
        dice = np.random.rand()

        if dice < p_left:
            return "Left"
        else:
            return "Right"

    def update(self, choice, reward):
        if choice == "Left":
            if reward == 1:
                p_obs_left_good = 0.74
                p_obs_right_good = 0.26 
            else:
                p_obs_left_good = 0.26
                p_obs_right_good = 0.74
        elif choice == "Right":
            if reward == 1:
                p_obs_left_good = 0.26
                p_obs_right_good = 0.74
            else:
                p_obs_left_good = 0.74
                p_obs_right_good = 0.26
    
        self.P_left_good = (p_obs_left_good * self.P_left_good) / ( (p_obs_left_good * self.P_left_good) + (p_obs_right_good * (1 - self.P_left_good)) )
        self.P_left_good = ( (1 - self.reversal) * self.P_left_good + self.reversal * (1 - self.P_left_good))

        self.prev_prev_choice = self.previous_choice
        self.previous_choice = choice

    def p_left(self):
        P_right_good = 1 - self.P_left_good 
    
        Q_left = 0.74 * self.P_left_good + 0.26 * P_right_good
        Q_right = 0.26 * self.P_left_good + 0.74 * P_right_good 

        QL_p = 0
        QR_p = 0

        if self.prev_prev_choice == "Left" and self.previous_choice == "Left":
            QL_p = 1
        elif self.prev_prev_choice == "Right" and self.previous_choice == "Right":
            QR_p = 1

        Q_left_net = self.weight * Q_left + self.pweight * QL_p
        Q_right_net = self.weight * Q_right + self.pweight * QR_p

        p_left = np.exp(Q_left_net) / ( np.exp(Q_left_net) + np.exp(Q_right_net))

        return p_left