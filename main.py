import numpy as np
import random
import time
import pandas as pd
import matplotlib.pyplot as plt

# Function to read data from the input file
def read_input_file(filename):
    node_weights = {}
    edges = {}
    L = []
    U = []
    num_clusters = 0

    with open(filename, 'r') as file:
        lines = file.readlines()
        
        # Read the first line
        first_line = lines[0].strip().split()
        M = int(first_line[0])
        num_clusters = int(first_line[1])
        cluster_limits = list(map(int, first_line[2:num_clusters * 2 + 2:2]))
        cluster_limits += list(map(int, first_line[3:num_clusters * 2 + 3:2]))
        L = cluster_limits[:num_clusters] 
        U = cluster_limits[num_clusters:] 
        
        # Read node weights
        node_weights_list = list(map(int, first_line[num_clusters * 2 + 3:]))
        for i in range(M):
            node_weights[i] = node_weights_list[i]

        # Read edges
        for line in lines[1:]:
            if line.strip():  # Skip empty lines
                node_a, node_b, edge_weight = line.strip().split()
                node_a, node_b, edge_weight = int(node_a), int(node_b), float(edge_weight)
                edges[(node_a, node_b)] = edge_weight
                edges[(node_b, node_a)] = edge_weight  # Since the graph is undirected

    return node_weights, edges, L, U

# Calculate internal edge weights within clusters
def calculate_internal_edge_weights(clusters, edges):
    total_weight = 0
    for cluster in clusters.values():
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                node_i, node_j = cluster[i], cluster[j]
                # Check if the edge exists in either direction
                total_weight += edges.get((node_i, node_j), 0)  # Avoid redundant checks
    return total_weight

# Greedy Heuristic (GH)
def greedy_heuristic(node_weights, edges, L, U):
    clusters = {i: [] for i in range(len(L))}  # Create clusters based on the number of limits
    cluster_weights = {i: 0 for i in range(len(L))}

    nodes = list(node_weights.keys())
    
    # Sort all edges by weight in descending order to start from the strongest connections
    sorted_edges = sorted(edges.items(), key=lambda x: x[1], reverse=True)

    # Iterate through sorted edges and assign nodes
    for (node1, node2), weight in sorted_edges:
        if node1 in nodes and node2 in nodes:  # Check if both nodes are still available
            # Try to place the edge's nodes in the same cluster
            for cluster_index in range(len(L)):
                if cluster_weights[cluster_index] + node_weights[node1] + node_weights[node2] <= U[cluster_index]:
                    # Place the best edge nodes in the cluster
                    clusters[cluster_index].extend([node1, node2])
                    cluster_weights[cluster_index] += node_weights[node1] + node_weights[node2]
                    nodes.remove(node1)
                    nodes.remove(node2)

                    # Now try to add individual nodes that maximize internal impact to this cluster
                    while True:
                        best_node = None
                        max_impact = -1

                        # Find the node that has the highest impact with current cluster members
                        for node in nodes:
                            impact = sum(edges.get((node, n), 0) for n in clusters[cluster_index])

                            # Check if the node fits within the cluster's capacity and has the best impact
                            if impact > max_impact and cluster_weights[cluster_index] + node_weights[node] <= U[cluster_index]:
                                max_impact = impact
                                best_node = node

                        # If a valid node is found, add it to the cluster
                        if best_node:
                            clusters[cluster_index].append(best_node)
                            cluster_weights[cluster_index] += node_weights[best_node]
                            nodes.remove(best_node)
                        else:
                            # No more valid nodes to add to this cluster
                            break

                    # Break after successfully adding nodes from this edge to the cluster
                    break
    
    # Redistribute nodes to meet lower constraint L
    def redistribute_nodes():
        for cluster_index in range(len(L)):
            if cluster_weights[cluster_index] < L[cluster_index]:
                # Try to find nodes from other clusters or remaining nodes
                for i in range(len(clusters)):
                    if i != cluster_index:
                        # Try moving nodes from other clusters if it fits
                        for node in clusters[i]:
                            if (cluster_weights[cluster_index] + node_weights[node] <= U[cluster_index] and
                                cluster_weights[i] - node_weights[node] >= L[i]):
                                # Transfer node from cluster i to cluster_index
                                clusters[i].remove(node)
                                clusters[cluster_index].append(node)
                                cluster_weights[i] -= node_weights[node]
                                cluster_weights[cluster_index] += node_weights[node]
                                break  # Exit after one successful transfer
                # After attempting transfers, check if the cluster meets the lower limit
                if cluster_weights[cluster_index] < L[cluster_index]:
                    print(f"Cluster {cluster_index+1} still lower than L after redistribution.")
    
    redistribute_nodes()
    return clusters, cluster_weights

# Hill Climbing Algorithm
def hill_climbing(clusters, node_weights, edges, L, U, max_iterations=10, no_improvement_limit=5, node_threshold=100):
    best_clusters = {i: clusters[i][:] for i in clusters}  # Deep copy of clusters
    best_internal_weight = calculate_internal_edge_weights(best_clusters, edges)
    
    improved = True
    no_improvement_counter = 0
    total_nodes = sum(len(cluster) for cluster in clusters.values())

    use_swaps = total_nodes < node_threshold  # Use swaps only if the total number of nodes is below the threshold

    while improved and no_improvement_counter < no_improvement_limit:
        improved = False
        
        for _ in range(max_iterations):
            cluster_indices = list(best_clusters.keys())

            # Node swapping logic (if the total number of nodes is under the threshold)
            if use_swaps:
                for cluster_1_idx in cluster_indices:
                    for cluster_2_idx in cluster_indices:
                        if cluster_1_idx != cluster_2_idx:
                            cluster_1 = best_clusters[cluster_1_idx]
                            cluster_2 = best_clusters[cluster_2_idx]
                            
                            if cluster_1 and cluster_2:
                                # Try all possible swaps between cluster_1 and cluster_2
                                best_swap = None
                                best_swap_gain = 0

                                for node_from_1 in cluster_1:
                                    for node_from_2 in cluster_2:
                                        current_weight_1 = sum(edges.get((node_from_1, n), 0) for n in cluster_1)
                                        current_weight_2 = sum(edges.get((node_from_2, n), 0) for n in cluster_2)
                                        
                                        # Perform the swap
                                        cluster_1.remove(node_from_1)
                                        cluster_2.append(node_from_1)
                                        cluster_2.remove(node_from_2)
                                        cluster_1.append(node_from_2)
                                        
                                        # Recalculate the new internal weights after the swap
                                        new_weight_1 = sum(edges.get((node_from_2, n), 0) for n in cluster_1)
                                        new_weight_2 = sum(edges.get((node_from_1, n), 0) for n in cluster_2)
                                        
                                        # Ensure the clusters are within their weight bounds after the swap
                                        if (L[cluster_1_idx] <= sum(node_weights[n] for n in cluster_1) <= U[cluster_1_idx]) and \
                                            (L[cluster_2_idx] <= sum(node_weights[n] for n in cluster_2) <= U[cluster_2_idx]):

                                            swap_gain = (new_weight_1 + new_weight_2) - (current_weight_1 + current_weight_2)
                                            
                                            if swap_gain > best_swap_gain:
                                                best_swap_gain = swap_gain
                                                best_swap = (node_from_1, node_from_2)

                                        # Undo the swap for the next iteration
                                        cluster_1.remove(node_from_2)
                                        cluster_2.append(node_from_2)
                                        cluster_2.remove(node_from_1)
                                        cluster_1.append(node_from_1)

                                # If a beneficial swap is found, apply it
                                if best_swap:
                                    node_from_1, node_from_2 = best_swap
                                    cluster_1.remove(node_from_1)
                                    cluster_2.append(node_from_1)
                                    cluster_2.remove(node_from_2)
                                    cluster_1.append(node_from_2)
                                    
                                    best_internal_weight += best_swap_gain
                                    improved = True
                                    no_improvement_counter = 0  # Reset the counter

            # Move logic
            for cluster_idx in cluster_indices:
                for other_cluster_idx in cluster_indices:
                    if cluster_idx != other_cluster_idx:
                        cluster = best_clusters[cluster_idx]
                        other_cluster = best_clusters[other_cluster_idx]
                        
                        if cluster:
                            # Try all possible moves from cluster to other_cluster
                            best_move = None
                            best_move_gain = 0

                            for node_to_move in cluster:
                                current_cluster_weight = sum(edges.get((node_to_move, n), 0) for n in cluster)
                                new_cluster_weight = sum(edges.get((node_to_move, n), 0) for n in other_cluster)
                                
                                # Ensure the move respects the weight bounds of both clusters
                                if (L[other_cluster_idx] <= sum(node_weights[n] for n in other_cluster) + node_weights[node_to_move] <= U[other_cluster_idx]) and \
                                    (L[cluster_idx] <= sum(node_weights[n] for n in cluster) - node_weights[node_to_move] <= U[cluster_idx]):
                                    
                                    move_gain = new_cluster_weight - current_cluster_weight
                                    
                                    if move_gain > best_move_gain:
                                        best_move_gain = move_gain
                                        best_move = node_to_move

                            # If a beneficial move is found, apply it
                            if best_move:
                                node_to_move = best_move
                                cluster.remove(node_to_move)
                                other_cluster.append(node_to_move)
                                
                                best_internal_weight += best_move_gain
                                improved = True
                                no_improvement_counter = 0

            # Early termination if no improvement for multiple iterations
            if not improved:
                no_improvement_counter += 1
            else:
                no_improvement_counter = 0

    return best_clusters, best_internal_weight

# Genetic Algorithm
def genetic_algorithm(initial_solution, node_weights, edges, L, U, population_size=5000, generations=100, mutation_rate=0.1):
    def fitness(clusters):
        # Penalize solutions that violate the cluster weight bounds
        penalty = 0
        for cluster_idx, cluster in clusters.items():
            cluster_weight = sum(node_weights[n] for n in cluster)
            if cluster_weight < L[cluster_idx] or cluster_weight > U[cluster_idx]:
                penalty += abs(cluster_weight - U[cluster_idx]) if cluster_weight > U[cluster_idx] else abs(L[cluster_idx] - cluster_weight)
        # Fitness is internal edge weights minus any penalties
        return calculate_internal_edge_weights(clusters, edges) - penalty

    def selection(population):
        # Select the best two individuals
        population.sort(key=lambda x: fitness(x), reverse=True)
        return population[:2]

    def crossover(parent1, parent2):
        # Uniform crossover: randomly select clusters from either parent
        child1, child2 = {}, {}
        for i in parent1.keys():
            if random.random() < 0.5:
                child1[i] = parent1[i][:]
                child2[i] = parent2[i][:]
            else:
                child1[i] = parent2[i][:]
                child2[i] = parent1[i][:]
        return child1, child2

    def mutate(solution):
        # Move a random node from one cluster to another
        cluster_keys = list(solution.keys())
        cluster_from = random.choice(cluster_keys)
        cluster_to = random.choice(cluster_keys)
        if solution[cluster_from]:
            node = random.choice(solution[cluster_from])
            # Ensure move respects the weight bounds
            if sum(node_weights[n] for n in solution[cluster_to]) + node_weights[node] <= U[cluster_to] and \
               sum(node_weights[n] for n in solution[cluster_from]) - node_weights[node] >= L[cluster_from]:
                solution[cluster_from].remove(node)
                solution[cluster_to].append(node)

    def repair(solution):
        # Repair mechanism to fix clusters exceeding capacity constraints
        for cluster_idx in solution:
            cluster_weight = sum(node_weights[n] for n in solution[cluster_idx])
            
            # If the cluster exceeds the upper limit, we need to redistribute nodes
            if cluster_weight > U[cluster_idx]:
                overflow_nodes = []
                
                # Sort nodes based on their weight (heaviest first)
                for node in sorted(solution[cluster_idx], key=lambda n: node_weights[n], reverse=True):
                    if cluster_weight <= U[cluster_idx]:
                        break
                    overflow_nodes.append(node)
                    cluster_weight -= node_weights[node]

                # Try to move overflow nodes to other clusters with capacity
                for node in overflow_nodes:
                    for other_cluster_idx in solution:
                        if other_cluster_idx != cluster_idx:
                            other_cluster_weight = sum(node_weights[n] for n in solution[other_cluster_idx])
                            if other_cluster_weight + node_weights[node] <= U[other_cluster_idx]:
                                # Move the node to the other cluster
                                solution[cluster_idx].remove(node)
                                solution[other_cluster_idx].append(node)
                                break

        return solution

    # Initialize population with valid initial solution copies
    population = [initial_solution.copy() for _ in range(population_size)]

    best_solution = initial_solution.copy()
    best_fitness = fitness(best_solution)

    for generation in range(generations):
        new_population = []
        
        # Selection of best parents
        parents = selection(population)
        
        # Generate new population through crossover and mutation
        while len(new_population) < population_size:
            parent1, parent2 = random.choice(parents), random.choice(parents)
            child1, child2 = crossover(parent1, parent2)
            
            if random.random() < mutation_rate:
                mutate(child1)
                mutate(child2)
            
            # Apply the repair mechanism after mutation and crossover
            child1 = repair(child1)
            child2 = repair(child2)

            new_population.append(child1)
            new_population.append(child2)

        # Combine old and new population (elitism)
        combined_population = population + new_population
        combined_population.sort(key=lambda x: fitness(x), reverse=True)
        population = combined_population[:population_size]  # Keep only the top solutions

        # Evaluate new population and keep the best
        for individual in population:
            current_fitness = fitness(individual)
            if current_fitness > best_fitness:
                best_solution = individual
                best_fitness = current_fitness

        # Convergence check: terminate early if fitness improvement is negligible
        if generation > 1 and abs(best_fitness - fitness(population[0])) < 1e-6:
            break  # Terminate if no significant improvement

    return best_solution, best_fitness



def run_optimization(node_weights, edges, L, U, num_experiments=10):
    greedy_scores = []
    hill_climbing_scores = []
    genetic_scores = []

    greedy_times = []
    hill_climbing_times = []
    genetic_times = []

    greedy_cluster_weights = []
    hill_climbing_cluster_weights = []
    genetic_cluster_weights = []

    for i in range(num_experiments):
        print(f"\nRunning Experiment {i+1}...")

        # Step 1: Run the Greedy Heuristic
        start_time = time.time()
        initial_clusters, initial_weights = greedy_heuristic(node_weights, edges, L, U)
        initial_score = calculate_internal_edge_weights(initial_clusters, edges)
        greedy_elapsed_time = time.time() - start_time

        # Capture results for greedy heuristic
        greedy_scores.append(initial_score)
        greedy_times.append(greedy_elapsed_time)
        
        # Sum the weights for each cluster
        final_greedy_weights = {cluster_idx: sum(node_weights[node] for node in initial_clusters[cluster_idx])
                                for cluster_idx in initial_clusters}
        greedy_cluster_weights.append(final_greedy_weights)

        # Step 2: Run Hill Climbing
        start_time = time.time()
        improved_clusters, improved_score = hill_climbing(initial_clusters, node_weights, edges, L, U)
        hill_climbing_elapsed_time = time.time() - start_time

        # Capture results for hill climbing
        hill_climbing_scores.append(improved_score)
        hill_climbing_times.append(hill_climbing_elapsed_time)
        
        # Sum the weights for each cluster
        final_hill_climbing_weights = {cluster_idx: sum(node_weights[node] for node in improved_clusters[cluster_idx])
                                       for cluster_idx in improved_clusters}
        hill_climbing_cluster_weights.append(final_hill_climbing_weights)

        # Step 3: Run Genetic Algorithm
        start_time = time.time()
        genetic_solution, genetic_score = genetic_algorithm(initial_clusters, node_weights, edges, L, U)
        genetic_elapsed_time = time.time() - start_time

        # Capture results for genetic algorithm
        genetic_scores.append(genetic_score)
        genetic_times.append(genetic_elapsed_time)
        
        # Sum the weights for each cluster
        final_genetic_weights = {cluster_idx: sum(node_weights[node] for node in genetic_solution[cluster_idx])
                                 for cluster_idx in genetic_solution}
        genetic_cluster_weights.append(final_genetic_weights)

    # Display and return results
    print("\n===== Optimization Results Summary =====\n")

    print("Greedy Heuristic:")
    print(f"  Best score: {max(greedy_scores):.4f}")
    print(f"  Average score: {np.mean(greedy_scores):.4f}")
    print(f"  Standard deviation: {np.std(greedy_scores):.4f}")
    print(f"  Average time (seconds): {np.mean(greedy_times):.4f}")

    print("\nHill Climbing:")
    print(f"  Best score: {max(hill_climbing_scores):.4f}")
    print(f"  Average score: {np.mean(hill_climbing_scores):.4f}")
    print(f"  Standard deviation: {np.std(hill_climbing_scores):.4f}")
    print(f"  Average time (seconds): {np.mean(hill_climbing_times):.4f}")

    print("\nGenetic Algorithm:")
    print(f"  Best score: {max(genetic_scores):.4f}")
    print(f"  Average score: {np.mean(genetic_scores):.4f}")
    print(f"  Standard deviation: {np.std(genetic_scores):.4f}")
    print(f"  Average time (seconds): {np.mean(genetic_times):.4f}")

    # Print cluster weights for each experiment
    print("\nCluster Weights per Experiment:")
    for i in range(num_experiments):
        print(f"\nExperiment {i + 1}:")
        print(f"  Greedy Heuristic cluster weights: {greedy_cluster_weights[i]}")
        print(f"  Hill Climbing cluster weights: {hill_climbing_cluster_weights[i]}")
        print(f"  Genetic Algorithm cluster weights: {genetic_cluster_weights[i]}")

    return greedy_scores, hill_climbing_scores, greedy_times, hill_climbing_times, genetic_scores, genetic_times

# Visualization and reporting function
def plot_and_display_results(greedy_scores, hill_climbing_scores, genetic_scores, greedy_times, hill_climbing_times, genetic_times):
    # Create a DataFrame for table display
    data = {
        "Experiment": range(1, len(greedy_scores) + 1),
        "Greedy Scores": greedy_scores,
        "Hill Climbing Scores": hill_climbing_scores,
        "Genetic Algorithm Scores": genetic_scores,
        "Greedy Time (s)": greedy_times,
        "Hill Climbing Time (s)": hill_climbing_times,
        "Genetic Algorithm Time (s)": genetic_times
    }
    df = pd.DataFrame(data)
    print(df)

    # Plot scores and times
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Plot the scores on the left
    axs[0].plot(df["Experiment"], df["Greedy Scores"], label="Greedy Heuristic", marker='o', color='blue')
    axs[0].plot(df["Experiment"], df["Hill Climbing Scores"], label="Hill Climbing", marker='x', color='green')
    axs[0].plot(df["Experiment"], df["Genetic Algorithm Scores"], label="Genetic Algorithm", marker='s', color='red')
    axs[0].set_xlabel("Experiment")
    axs[0].set_ylabel("Score")
    axs[0].set_title("Comparison of Algorithm Scores")
    axs[0].legend()
    axs[0].grid(True)

    # Plot the times on the right
    axs[1].plot(df["Experiment"], df["Greedy Time (s)"], label="Greedy Time", marker='o', color='blue')
    axs[1].plot(df["Experiment"], df["Hill Climbing Time (s)"], label="Hill Climbing Time", marker='x', color='green')
    axs[1].plot(df["Experiment"], df["Genetic Algorithm Time (s)"], label="Genetic Algorithm Time", marker='s', color='red')
    axs[1].set_xlabel("Experiment")
    axs[1].set_ylabel("Time (seconds)")
    axs[1].set_title("Comparison of Algorithm Times")
    axs[1].legend()
    axs[1].grid(True)

    # Ensure there's no overlap of titles/labels
    plt.tight_layout()
    plt.show()

# Read input from the file
filename = r'Sparse82.txt' # RanReal480 - Sparse82
node_weights, edges, L, U = read_input_file(filename)

# Run the optimization with multiple experiments
num_experiments = 10
greedy_scores, hill_climbing_scores, greedy_times, hill_climbing_times, genetic_scores, genetic_times = run_optimization(node_weights, edges, L, U, num_experiments)

# Generate tables and figures
plot_and_display_results(greedy_scores, hill_climbing_scores, genetic_scores, greedy_times, hill_climbing_times, genetic_times)