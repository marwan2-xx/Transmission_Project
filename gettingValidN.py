def get_reuse_parameters(n_array):
    """
    Takes an array of reuse factors (N) and returns a dictionary 
    mapping N to its valid (i, j) tuple based on N = i^2 + ij + j^2.
    """
    results = {}
    
    for N in n_array:
        found = False
        # Limit search range based on N; i cannot exceed sqrt(N)
        limit = int(N**0.5) + 1
        
        for i in range(limit + 1):
            for j in range(i + 1):  # Ensures i >= j
                if (i**2 + i*j + j**2) == N:
                    results[N] = (i, j)
                    found = True
                    break
            if found:
                break
        
        # If no valid (i, j) pair exists for the given N
        if not found:
            results[N] = None
            
    return results

n_values = []
for i in range(3,20):
    n_values.append(i)

reuse_dict = get_reuse_parameters(n_values)
print(reuse_dict)