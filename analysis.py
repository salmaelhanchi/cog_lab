import torch
import numpy as np
from sklearn.metrics import mutual_info_score

def discretize_states(states, n_bins=8):
    """Converts continuous hidden states into discrete bins for MI calculation."""
    # This is a simple but effective way to prepare data for mutual_info_score
    binned_states = np.empty_like(states, dtype=np.int32)
    
    # We discretize each neuron's activity independently
    for neuron_idx in range(states.shape[1]):
        neuron_activity = states[:, neuron_idx]
        bins = np.linspace(neuron_activity.min(), neuron_activity.max(), n_bins + 1)
        binned_states[:, neuron_idx] = np.digitize(neuron_activity, bins)
        
    return binned_states

def calculate_predictive_information(hidden_states, max_tau=10):
    """
    Calculates the mutual information I(H_t; H_{t+τ}) for a range of τ.
    
    Args:
        hidden_states (np.array): A numpy array of shape (num_samples, seq_len, hidden_size).
        max_tau (int): The maximum time delay to test.
        
    Returns:
        dict: A dictionary mapping each tau to its calculated mutual information.
    """
    num_samples, seq_len, hidden_size = hidden_states.shape
    results = {}

    print(f"\n--- Starting Predictive Information Analysis (max τ = {max_tau}) ---")

    for tau in range(1, max_tau):
        if tau >= seq_len:
            break
            
        # Prepare the "past" (H_t) and "future" (H_{t+τ}) states
        # We flatten the batch and sequence dimensions to treat each time step as a sample
        past_states = hidden_states[:, :-tau, :].reshape(-1, hidden_size)
        future_states = hidden_states[:, tau:, :].reshape(-1, hidden_size)
        
        # Discretize the continuous states into bins
        past_discrete = discretize_states(past_states)
        future_discrete = discretize_states(future_states)
        
        # Calculate MI for each neuron and average them
        mi_scores = []
        for neuron_idx in range(hidden_size):
            mi = mutual_info_score(past_discrete[:, neuron_idx], future_discrete[:, neuron_idx])
            mi_scores.append(mi)
            
        average_mi = np.mean(mi_scores)
        results[tau] = average_mi
        print(f"τ = {tau}: Average Mutual Information = {average_mi:.4f}")
        
    print("--- Analysis Complete ---")
    return results