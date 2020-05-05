import random, copy
import json
import numpy as np
class UserSimulator:
    """Simulates a real user, to train the agent with reinforcement learning."""

    def __init__(self, disease_symptom_path, disease_symptom_mapping_path):
        """
        The constructor for UserSimulator. Sets dialogue config variables.

        Parameters:
            disease_symptom_path (list): JSON file containing disease and symptom list
            disease_symptom_mapping_path (dict): JSON file containing disease to symptom mapping.
        """
        with open(disease_symptom_path) as f:
            disease_symptom = json.load(f)

        self.disease_name = disease_symptom['disease_list']

        self.symptom_name = disease_symptom['symptom_name']

        self.num_disease = len(self.disease_name)
        # Number of Symptom.
        self.num_symptom = len(self.symptom_name)

        # Initial State.
        with open(disease_symptom_mapping_path) as f:
            self.disease_symptom_mapping_dict = json.load(f)

        self.initial_state = np.array([[0, 0, 1.] for _ in range(self.num_symptom)])

        # Symptom mapping
        self.symptom_dict = {s : n for n, s in enumerate(self.symptom_name)}

        # Action Space
        self.action_type = {'type' : None, 'value' : None}

        # Make action space
        self.action_space = []

        # Diagnosis Action Space
        self.diagnosis_action_space = []
        
        # Symptom Action Space
        self.symptom_action_space = [] 

        # Add symptom action
        for s in self.symptom_name:
            val = {}
            val['type'] = 'symptom_check'
            val['value'] = s
            self.action_space.append(val)
            self.symptom_action_space.append(val)

        # Add Diagnosis Action
        for d in self.disease_name:
            val = {}
            val['type'] = 'diagnosis_check'
            val['value'] = d
            self.action_space.append(val)
            self.diagnosis_action_space.append(val)

        self.action_shape = len(self.action_space)
        self.state_shape = (self.num_symptom, 3)

    def reset(self):
        """
        Resets the user sim. by emptying the state and returning the initial action.

        Returns:
            dict: The initial action of an episode
        """
        # Goal Disease
        self.goal = random.choice(self.disease_name)
        
        # Reset State
        self.state = np.array(self.initial_state)

        return self._return_init_action()

    def _return_init_action(self):
        """
        Returns the initial action of the episode.

        The initial action has an intent of request, required init. inform slots and a single request slots.

        Returns:
            dict: Initial user response
        """

        # Get first symptom from the user
        self.symptom_list = self.disease_symptom_mapping_dict[self.goal].keys()

        num_symptom = len(self.symptom_list)

        num_init_sample = random.randint(1, num_symptom / 2.)
        
        init_symptom = random.sample(self.symptom_list, num_init_sample)

        for s in init_symptom:
            self.state[self.symptom_dict[s]] = np.array([1.,0,0])

        return self.state.flatten()

    def step(self, agent_action):
        """
        Return the response of the user sim. to the agent by using rules that simulate a user.

        Given the agent action craft a response by using deterministic rules that simulate (to some extent) a user.
        Some parts of the rules are stochastic. Check if the agent has succeeded or lost or still going.

        Parameters:
            agent_action : Discrete Number representing action number.

        Returns:
            dict: User state
            int: Reward
            bool: Done flag
            int: Success: -1, 0 or 1 for loss, neither win nor loss, win
        """

        # Assertions -----

        action = self.action_space[agent_action]
        # print (action, agent_action)
        agent_action_type = action['type']
        if agent_action_type == 'symptom_check':
            # print ('symptom check')
            symptom_to_check = action['value']
            if symptom_to_check in self.symptom_list:
                # Bool : To check if the symptom exists in our state space.
                check_if_symptom_already_in_state = (self.state[self.symptom_dict[symptom_to_check]] == np.array([1.,0,0])).all()
                # Return -1 if state already checked.
                if check_if_symptom_already_in_state:
                    return self.state.flatten(), -1, False, {}
                else:
                    # If symptom not in state, append it to state.
                    self.state[self.symptom_dict[symptom_to_check]] = np.array([1., 0, 0])
                    return self.state.flatten(), 0, False, {}
            else:
                # If symptom not present in the state.
                self.state[self.symptom_dict[symptom_to_check]] = np.array([0, 1., 0])
                return self.state.flatten(), 0, False, {} 
        elif agent_action_type == 'diagnosis_check':

            disease_to_check = action['value']
            # If Goal equals Disease to check.
            if disease_to_check == self.goal:
                return self.state.flatten(), 1, True, {}
            # If Goal is not equal to Disease to check. 
            else:
                return self.state.flatten(), 0, True, {}

    # Get top 5 - top 10 disease.
    def get_top_diseases(self, q_values):
        q_values = list(q_values)
        idx = sorted(range(len(q_values)), key=lambda k: q_values[k], reverse=True)
        top_10_idx = idx[:10]
        top_10_disease = []
        top_5_disease = []
        for n, i in enumerate(top_10_idx):
            d_act = self.diagnosis_action_space[i]
            top_10_disease.append(d_act)
            if n < 5:
                top_5_disease.append(d_act)
        return top_5_disease, top_10_disease
