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

def is_valid_N(N):
    for i in range(0, N + 1):
        for j in range(0, i + 1):
            val = i*i + i*j + j*j
            if val == N:  return True
            if val > N:   break
    return False

valid_Ns = []
for i in range(3,20):
    if(is_valid_N(i) == True):
        valid_Ns.append(i)
print(valid_Ns)
print(get_reuse_parameters(valid_Ns))
valid_Ns = [3,4,7,9,12,13,16,19]
values = ["no sectoring" , "60-degrees" , "120-degrees" , "180-degrees"]
valid_ns = valid_ns = {3:[6,2,3,2] , 
            4:[6,1,2,2],
            7:[6,3,2,2],
            9:[6,3,2,2],
            12:[6,2,2,2],
            13:[6,3,2,2],
            16:[6,3,2,2],
            19:[6,2,2,2]}

# reuse_dict = get_reuse_parameters(n_values)
# print(reuse_dict)

valid_Ns = [3,4,7,9,12,13,16,19]
values = ["no sectoring" , "60-degrees" , "120-degrees" , "180-degrees"]
valid_ns = {3:[6,2,3,2] , 
            4:[6,1,2,2],
            7:[6,3,2,2],
            9:[6,3,2,2],
            12:[6,2,2,2],
            13:[6,3,2,2],
            16:[6,3,2,2],
            19:[6,2,2,2]}


valid_Ns = [3, 4, 7, 9, 12, 13, 16, 19]

sectoring_labels = ["no sectoring", "60-degrees", "120-degrees", "180-degrees"]

valid_ns = {
    3:  [6, 2, 3, 2],
    4:  [6, 1, 2, 2],
    7:  [6, 3, 2, 2],
    9:  [6, 3, 2, 2],
    12: [6, 2, 2, 2],
    13: [6, 3, 2, 2],
    16: [6, 3, 2, 2],
    19: [6, 2, 2, 2],
}


def find_min_reuse(ns_table: dict, required_CIR: float) -> dict:
    """
    For each N in ns_table, iterates over its n values (one per sectoring type)
    and picks the first n where 3N/n <= required_CIR.

    Returns
    -------
    { N: (n, sectoring_label) }  for every N that has at least one match.
    """
    results = {}

    for N, ns in ns_table.items():
        for n, label in zip(ns, sectoring_labels):
            if (3 * N) / n >= required_CIR:
                results[N] = (n, label)
                break           # first match wins

    return results


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = find_min_reuse(valid_ns, 10)
    for N, (n, label) in result.items():
        print(f"N={N}: n={n}, sectoring='{label}', C/I={3*N/n:.4f}")

