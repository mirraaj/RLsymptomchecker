# RL Symptom Checker

This work is based on usage of Reinforcement Learning for symptom checker. 


## Model

The model m p accepts a state s that comprises symptom statuses inquired by our model. Formally,
we describe the state encoding scheme as follows: First, each symptom i ∈ I p can be one of the
following statuses: true, false, and unknown. We can use a three-element one-hot vector b i ∈ B 3
to encode the status of a symptom i. Second, the status of a symptom is determined based on the
following rule. If a user responded yes to a symptom inquired by our model, that symptom is marked
as true. On the other hand, if the user responded no, the symptom is marked as false. Symptoms not
inquired by our model are marked as unknown. Finally, a state s then concatenates all the symptom
statuses into a Boolean vector, i.e., s = [b<sup>T</sup>1 , b<sup>T</sup>2 , . . . , b<sup>T</sup> |Ip |]T.


Given a state s, our model m p outputs the Q-value of each action a ∈ A p . In our definition,
each action a has two types: an inquiry (symptom) action (a ∈ I<sub>p</sub> ) or a diagnosis (disease) action (a ∈ D<sub>p</sub> ). If the maximum Q-value of the outputs corresponds to an inquiry action, then our model inquires the
corresponding symptom to a user, obtains a feedback, and proceeds to the next time step. The
feedback is incorporated into the next state s t+1 according to our state encoding scheme. Otherwise,
the maximum Q-value corresponds to a diagnosis action. In the latter case, our model predicts the
maximum-Q-value disease and then terminates.

Paper : http://infolab.stanford.edu/~echang/NIPS_DeepRL_2016_Symptom_Checker.pdf

## Reward System

### Success: -1, 0 or 1 for loss, neither win nor loss, win

### Loss : When the agent action chooses the symptom action already checked

### Neither win nor loss : 

a) When the agent action chooses symptom action and it is not there in symptom list.

b) When the agent action chooses symptom action and it is there in symptom list.

c) When the agent action chooses disease action and it is not equal to the goal disease.

### Win : When the agent action chooses the symptom action which equals goal state.

## Todo

Current Implementation doesn't incorporate the hierarchical part. Need to add it when anatomical data is ready.