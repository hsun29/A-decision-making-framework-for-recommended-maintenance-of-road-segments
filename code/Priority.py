import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization

# Generate sample data
treat_decision = pd.read_excel('Treatment_decision.xlsx')
proprocessed_data =  treat_decision[treat_decision['LX']=='G218'].reset_index()
patients = proprocessed_data.index.astype('str').to_list()
disease_degree = np.array(proprocessed_data['PCI_'])
area = np.array(proprocessed_data['Area'])
Tech_class_2019 = np.array(proprocessed_data['Tech_class_2019'])
effect = np.array(proprocessed_data['effect'])
long_effect = np.array(proprocessed_data['effect']+proprocessed_data['PCI_'])


# Create a DataFrame to display the data as a table
data = {'Segment': patients, 'Disease Degree': disease_degree, 'Tech_class': Tech_class_2019, 'Effect': effect, 'Long_effect': long_effect}
df = pd.DataFrame(data)

# Display the data as a table
print(df)

# Define the scoring function
def score_function(disease_degree,area, Tech_class_2019, effect, long_effect ):
    # Define weights for each factor
    weights = {'Disease Degree': -0.4, 'Tech_class': 0.2, 'Effect': 0.4, 'Long_effect': 0.8}
    
    # Calculate the overall score using weighted sum
    score = disease_degree * weights['Disease Degree']  + Tech_class_2019 * weights['Tech_class'] + effect * weights['Effect'] + long_effect * weights['Long_effect']
    return score

# Define the objective function
def objective_function(patient_index):
    patient = patients[int(patient_index)]
    score = score_function(disease_degree[int(patient_index)], area=int(patient_index), Tech_class_2019=int(patient_index), effect=int(patient_index), long_effect=int(patient_index))
    return -score

# Define the search space
pbounds = {'patient_index': (0, len(patients) - 1)}

# Run Bayesian optimization
optimizer = BayesianOptimization(f=objective_function, pbounds=pbounds, allow_duplicate_points=True)
optimizer.maximize(init_points=5, n_iter=5)

# Retrieve the optimal patient
optimal_patient_index = int(optimizer.max['params']['patient_index'])
optimal_patient = patients[optimal_patient_index]

# Print the optimal patient
print("The patient with the highest expected impact is:", optimal_patient)

# Generate a priority list based on the scores
scores = [-objective_function(i) for i in range(len(patients))]  # Negate the scores to sort in descending order
priority_list = [patient for _, patient in sorted(zip(scores, patients), reverse=True)]

# Print the priority list
print("\nPriority List:")
for i, patient in enumerate(priority_list):
    print(f"{i+1}. {patient}")
