# Cluster Optimization Using Greedy, Hill Climbing, and Genetic Algorithms
 
This project addresses a **constrained clustering problem** in which a set of nodes needs to be grouped into clusters. The objective is to maximize the sum of internal edge weights within each cluster, ensuring that each cluster remains within specified weight bounds.
 
The project implements and compares three optimization algorithms: **Greedy Heuristic**, **Hill Climbing**, and **Genetic Algorithm**, each tailored to tackle this problem by balancing efficiency and solution quality.
 
## Problem Overview
 
Given:
- A graph with nodes, each assigned a specific weight
- Edges between nodes, each with a weight representing their connection strength
- Clusters, each with an upper and lower weight bound
 
The goal is to group nodes into clusters in such a way that:
1. **Internal Connectivity is Maximized**: The sum of edge weights within each cluster is as high as possible.
2. **Constraints are Satisfied**: Each cluster’s total node weight stays within its specified upper and lower limits.
 
The project’s algorithms use these constraints to optimize the clustering process.
 
## Project Structure
 
The repository contains:
- `main.py`: The primary code file with the implementation of each clustering algorithm
- `Sparse82`: A smaller dataset for testing clustering
- `RanReal480`: A larger dataset for scalability testing
- `Instance_format`: Description file for both datasets, providing details on node/edge structures and constraints
- `README.md`: A description of the project, its algorithms, and instructions for use
 
## Optimization Algorithms
 
1. **Greedy Heuristic (GH)**:
   - Starts by sorting edges by weight and clustering nodes with the strongest connections first, within weight limits.
   - Quickly provides a feasible initial solution but may not reach the global optimum.
 
2. **Hill Climbing**:
   - Begins with the Greedy Heuristic solution and makes incremental changes (node moves and swaps) to improve internal connectivity.
   - Balances speed with improvement over the initial Greedy solution but may reach a local optimum.
 
3. **Genetic Algorithm (GA)**:
   - Uses a population of possible solutions, applying selection, crossover, and mutation over generations to explore a wide solution space.
   - Aims to improve upon the Greedy solution with a focus on achieving higher connectivity within clusters.
 
## Installation and Usage
 
To run this project, install the necessary dependencies:
```bash
pip install numpy pandas matplotlib
```
 
To execute the code, run:
```bash
python main.py
```
 
### Sample Code Usage
 
The following demonstrates a full clustering process using the provided datasets.
 
```python
# Read data from a specified file (e.g., Sparse82)
filename = 'Sparse82'
node_weights, edges, L, U = read_input_file(filename)
 
# Run optimization across multiple experiments to compare results
num_experiments = 10
greedy_scores, hill_climbing_scores, greedy_times, hill_climbing_times, genetic_scores, genetic_times = run_optimization(node_weights, edges, L, U, num_experiments)
 
# Generate tables and plots showing performance of each algorithm
plot_and_display_results(greedy_scores, hill_climbing_scores, genetic_scores, greedy_times, hill_climbing_times, genetic_times)
```
 
### Expected Output
 
- **Tables** summarizing clustering scores and computation times for each algorithm.
- **Plots** comparing each algorithm's clustering score and efficiency across multiple runs.
 
## Dataset Descriptions
 
- **Sparse82**: A smaller dataset suitable for quick testing. Contains fewer nodes and edges, allowing for rapid experimentation.
- **RanReal480**: A larger dataset with more nodes and edges, intended to test scalability and performance.
 
Details on the dataset structure and constraints can be found in `Instance_format`.
 
## Results and Analysis
 
Upon running the code, the project outputs:
- **Scores and timings** for each algorithm across multiple runs.
- **Visualizations** comparing the performance of each method, illustrating the trade-offs between solution quality and computation time.
 
### Results for `Sparse82`
 
#### Performance Summary Table
 
| Algorithm         | Best Score  | Average Score | Standard Deviation | Average Time (s) |
|-------------------|-------------|---------------|---------------------|-------------------|
| Greedy Heuristic  | 824.8570    | 824.8570      | 0.0000             | 0.0053           |
| Hill Climbing     | 1249.2587   | 1249.2587     | 0.0000             | 1.0288           |
| Genetic Algorithm | 1119.1388   | 1038.5282     | 47.4647            | 4.0574           |
 
#### Graph (Plot)
![Sparse82 Graph](https://github.com/user-attachments/assets/15096e56-11f6-480a-a89e-f8b5f1c3c66e)

 
 
### Results for `RanReal480`
 
#### Performance Summary Table
 
| Algorithm         | Best Score     | Average Score | Standard Deviation | Average Time (s) |
|-------------------|----------------|---------------|---------------------|-------------------|
| Greedy Heuristic  | 346823.0840    | 346823.0840   | 0.0000             | 0.5146           |
| Hill Climbing     | 379137.7350    | 379137.7350   | 0.0000             | 2.9898           |
| Genetic Algorithm | 353253.7760    | 352262.3310   | 611.8258           | 75.9309          |
 
#### Graph (Plot)
![RanReal480 Graph](https://github.com/user-attachments/assets/ada47cda-9420-44c9-8e65-ffa60787b35c)

 
---
 
## Contributors

This project was developed by:
- **[Faisal Alkhrayef](https://www.linkedin.com/in/fkhrayef/)** <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="16" alt="LinkedIn Logo">
- **[Abdulaziz Al Frayan](https://www.linkedin.com/in/aziizdev/)** <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="16" alt="LinkedIn Logo">
- Yaser Alshareef

**Instructor**: **[Abdullah Alsheddy](https://www.linkedin.com/in/abdullah-alsheddy/)** <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="16" alt="LinkedIn Logo">

---
 
Feel free to explore the code, tweak the parameters, and experiment with different datasets. Enjoy clustering!
