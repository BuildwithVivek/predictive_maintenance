import numpy as np
import pandas as pd


failure_probabilities = [0.08, 0.6, 0.08, 0.08, 0.08, 0.08]
data = pd.read_csv('predictive_maintenance.csv')

def generate_data(num_rows=1):
    # Generate random data
    random_data = {
        'Product ID': np.random.choice(data['Product ID'], size=num_rows),

        'Location': np.random.choice(['Rajasthan', 'Maharashtra', 'Karnataka', 'Tamil Nadu', 'Uttar Pradesh', 'Gujarat', 'West Bengal', 'Madhya Pradesh', 'Bihar', 'Andhra Pradesh', 'Kerala', 'Punjab', 'Haryana', 'Odisha', 'Chhattisgarh', 'Jharkhand', 'Assam', 'Uttarakhand', 'Himachal Pradesh', 'Tripura'], size=num_rows),
                                   
        'Type': np.random.choice(['L', 'M', 'H'], size=num_rows),
        'Air temperature [K]': np.random.normal(300, 2, num_rows),
        'Process temperature [K]': np.random.normal(310, 1, num_rows),
        'Rotational speed [rpm]': np.random.randint(1000, 2500, size=num_rows),
        'Torque [Nm]': np.abs(np.random.normal(40, 100, num_rows)),
        'Tool wear [min]': np.random.randint(0, 250, size=num_rows),
        'Failure Type': np.random.choice(['Heat Dissipation', 'No Failure', 'Overstrain', 'Power Failure', 'Random Failures', 'Tool Wear'], size=num_rows, p=failure_probabilities)
    }
    
    # Create a DataFrame
    df = pd.DataFrame(random_data)
    
    return df
