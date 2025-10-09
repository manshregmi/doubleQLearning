import numpy as np
import os
from typing import Dict, Any, Union

# Define the default filenames used by your save_tables function
FILENAME_VALUE = 'value_table.npy'
FILENAME_POLICY = 'policy_table.npy'

def print_table_content(
    filename: str, 
    table_name: str, 
    limit: int = 10
) -> None:
    """
    Loads an R-L table (expected to be a dictionary) from a .npy file
    and prints its headers (keys) and values.
    """
    print("\n" + "="*70)
    print(f"## {table_name} Content (File: {filename})")
    print("="*70)

    if not os.path.exists(filename):
        print(f"ERROR: File not found at path: {filename}")
        return

    try:
        # Load the numpy file and convert it back to the original dictionary (.item())
        # This mirrors how your load_tables function retrieves the data
        table: Dict[Any, Any] = np.load(filename, allow_pickle=True).item()
        
        if not table:
            print("The loaded table is empty.")
            return

        print(f"Total entries (States/Headers): {len(table)}")
        
        print("\n--- Sample Entries ---")
        
        # Iterate through the table dictionary
        for i, (state, value) in enumerate(table.items()):
            if i >= limit:
                print(f"... Showing first {limit} entries. Total entries: {len(table)}")
                break
            
            # State/Header is the key
            state_str = str(state)
            
            # Value is the stored data (V(s) or action probabilities)
            value_str: str = ""
            
            # Format the output based on the type of value
            if isinstance(value, (int, float)):
                # Simple value (likely V(s) for the value table)
                value_str = f"Value: {value:.4f}"
            elif isinstance(value, dict):
                # Policy probabilities stored as a dict (sparse policy)
                snippet = dict(list(value.items())[:3])
                value_str = f"Policy Probs (Dict, size {len(value)}): {snippet}..."
            elif isinstance(value, (np.ndarray, list)):
                # Policy probabilities stored as an array/list (dense policy)
                snippet = value[:min(3, len(value))]
                value_str = f"Policy Probs (Array, size {len(value)}): {snippet}..."
            else:
                # Catch-all for other complex types
                value_str = f"Raw Value ({type(value).__name__}): {value}"

            print(f"[{i+1:02}] Header (State): {state_str} | {value_str}")

    except Exception as e:
        print(f"An error occurred while loading or printing {filename}: {e}")


if __name__ == "__main__":
    # Load and print the Critic's Value Table
    print_table_content(
        filename=FILENAME_VALUE, 
        table_name="CRITIC'S VALUE TABLE ($V(s)$)", 
        limit=5 # Only print the first 5 entries
    )
    
    # Load and print the Actor's Policy Table
    print_table_content(
        filename=FILENAME_POLICY, 
        table_name="ACTOR'S POLICY TABLE ($\pi(a|s)$)", 
        limit=15 # Only print the first 5 entries
    )
