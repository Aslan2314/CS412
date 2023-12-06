import time
import random
import numpy as np
from itertools import product
import matplotlib.pyplot as plt


def evaluate_clause(clause, assignment):
    for literal in clause:
        variable, is_negated = abs(literal), (literal < 0)
        if (assignment[variable] == 1 and not is_negated) or (assignment[variable] == 0 and is_negated):
            return 1
    return 0


def evaluate_clause_brute(clause, assignment):
    for literal in clause:
        variable, is_negated = abs(literal), (literal < 0)
        if (assignment[variable - 1] == 1 and not is_negated) or (assignment[variable - 1] == 0 and is_negated):
            return 1
    return 0


def max_3sat_brute_force(clauses, num_vars):
    best_assignment = None
    best_score = 0

    for assignment in product([0, 1], repeat=num_vars):
        assignment = list(assignment)
        score = sum(evaluate_clause_brute(clause, assignment) for clause in clauses)
        if score > best_score:
            best_score = score
            best_assignment = assignment

    return best_assignment, best_score


def max_3sat_approx(clauses, num_vars):
    assignment = [0] * (num_vars + 1)
    for clause in clauses:
        selected_literal = clause[0]
        variable = abs(selected_literal)
        is_negated = (selected_literal < 0)
        assignment[variable] = 1 if not is_negated else 0
    max_satisfied_clauses = sum(1 for clause in clauses if evaluate_clause(clause, assignment))
    return assignment[1:], max_satisfied_clauses


def generate_random_cnf(num_vars, num_clauses):
    cnf = []

    for _ in range(num_clauses):
        clause = []
        for _ in range(3):
            literal = random.randint(-num_vars, num_vars)
            while literal == 0:
                literal = random.randint(-num_vars, num_vars)
            clause.append(literal)
        cnf.append(clause)

    return cnf


def create_graphs_approx():
    random.seed(456348)
    y1 = []
    sample = np.array([i for i in range(1, 101)])
    for x in sample:
        clauses = generate_random_cnf(x, x)
        start = time.time()
        max_3sat_approx(clauses, x)
        end = time.time()
        y1.append(end - start)

    plt.plot(sample, y1)

    # Graph setup
    plt.title("7/8 Approximation Max 3-SAT")
    plt.xlabel("Number of clauses and variables")
    plt.ylabel("time (s)")
    plt.grid()
    plt.legend()
    plt.show()


def create_graphs_exact():
    random.seed(456348)
    y1 = []
    sample = np.array([i for i in range(1, 21)])
    for x in sample:
        clauses = generate_random_cnf(x, x)
        start = time.time()
        max_3sat_brute_force(clauses, x)
        end = time.time()
        y1.append(end - start)

    plt.plot(sample, y1)

    # Graph setup
    plt.title("Brute Force Max 3-SAT")
    plt.xlabel("Number of clauses and variables")
    plt.ylabel("time (s)")
    plt.grid()
    plt.legend()
    plt.show()


def create_graphs_compare():
    random.seed(456348)
    y1 = []
    sample = np.array([i for i in range(1, 21)])
    for x in sample:
        clauses = generate_random_cnf(x, x)
        ba1, score1 = max_3sat_brute_force(clauses, x)
        ba2, score2 = max_3sat_approx(clauses, x)
        diff_count = 0

        print(clauses)
        print(score1)
        print(ba1)
        print(score2)
        print(ba2)
        for i in range(len(ba1)):
            if i < len(ba2) and ba1[i] != ba2[i]:
                diff_count += 1
            if i >= len(ba2):
                diff_count += 1
        print(diff_count)
        y1.append(diff_count)

    plt.plot(sample, y1)

    # Graph setup
    plt.title("Comparisons of Exact vs Approximation for Max 3-SAT")
    plt.xlabel("Number of clauses and variables")
    plt.ylabel("Number of Differences")
    plt.grid()
    plt.legend()
    plt.show()


def create_randomness_graph(num_runs=10):
    random.seed(456348)
    y1 = []
    sample = np.array([i for i in range(1, 101)])
    for x in sample:
        diff_count = 0
        prev_assignment, _ = max_3sat_approx(generate_random_cnf(x, x), x)
        for _ in range(num_runs):
            clauses = generate_random_cnf(x, x)
            assignment, _ = max_3sat_approx(clauses, x)
            for i in range(len(assignment)):
                if assignment[i] != prev_assignment[i]:
                    diff_count += 1
            prev_assignment = assignment
        diff_count = diff_count // num_runs
        y1.append(diff_count)

    plt.plot(sample, y1)

    # Graph setup
    plt.title("Randomness of Approximation for Max 3-SAT")
    plt.xlabel("Number of clauses and variables")
    plt.ylabel("Number of Differences")
    plt.grid()
    plt.legend()
    plt.show()


def main():
    create_graphs_exact()
    create_graphs_approx()
    create_graphs_compare()
    create_randomness_graph()


if __name__ == "__main__":
    main()