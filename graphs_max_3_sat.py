import time
import random
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


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
    size = []  # Number of variables and clauses
    variations = []  # diff_counts
    y1 = []
    sample = np.array([i for i in range(1, 101)])
    for x in sample:
        size.append(x)
        diff_count = 0
        prev_assignment, _ = max_3sat_approx(generate_random_cnf(x, x), x)
        for n in range(num_runs):
            clauses = generate_random_cnf(x, x)
            assignment, _ = max_3sat_approx(clauses, x)
            for i in range(len(assignment)):
                if assignment[i] != prev_assignment[i]:
                    diff_count += 1
            prev_assignment = assignment
        diff_count = diff_count // num_runs
        variations.append(diff_count)
        y1.append(diff_count)

    plt.plot(sample, y1)

    # Graph setup
    plt.title("Randomness of Approximation for Max 3-SAT (Number of runs = {})".format(num_runs))
    plt.xlabel("Number of clauses and variables")
    plt.ylabel("Number of Differences")
    plt.grid()
    plt.legend()
    plt.show()

    data = []
    for i in range(len(size)):
        arr = [size[i], variations[i]]
        data.append(arr)
    frame = pd.DataFrame(np.array(data), columns=["Number of Variables and Clauses", "Number of Differences"])
    dataframe_to_pdf(frame, 'tables.pdf')


def _draw_as_table(df, pagesize):
    alternating_colors = [['white'] * len(df.columns), ['lightgray'] * len(df.columns)] * len(df)
    alternating_colors = alternating_colors[:len(df)]
    fig, ax = plt.subplots(figsize=pagesize)
    ax.axis('tight')
    ax.axis('off')
    the_table = ax.table(cellText=df.values,
                         rowLabels=df.index,
                         colLabels=df.columns,
                         rowColours=['lightblue'] * len(df),
                         colColours=['lightblue'] * len(df.columns),
                         cellColours=alternating_colors,
                         loc='center')
    return fig


def dataframe_to_pdf(df, filename, numpages=(1, 1), pagesize=(11, 8.5)):
    with PdfPages(filename) as pdf:
        nh, nv = numpages
        rows_per_page = len(df) // nh
        cols_per_page = len(df.columns) // nv
        for i in range(0, nh):
            for j in range(0, nv):
                page = df.iloc[(i * rows_per_page):min((i + 1) * rows_per_page, len(df)),
                       (j * cols_per_page):min((j + 1) * cols_per_page, len(df.columns))]
                fig = _draw_as_table(page, pagesize)
                if nh > 1 or nv > 1:
                    # Add a part/page number at bottom-center of page
                    fig.text(0.5, 0.5 / pagesize[0],
                             "Part-{}x{}: Page-{}".format(i + 1, j + 1, i * nv + j + 1),
                             ha='center', fontsize=8)
                pdf.savefig(fig, bbox_inches='tight')

                plt.close()


def main():
    # create_graphs_exact()
    # create_graphs_approx()
    # create_graphs_compare()
    create_randomness_graph(num_runs=50)


if __name__ == "__main__":
    main()
