from itertools import product


def evaluate_clause(clause, assignment):
    for literal in clause:
        variable, is_negated = abs(literal), (literal < 0)
        if (assignment[variable] == 1 and not is_negated) or (assignment[variable] == 0 and is_negated):
            return 1
    return 0


def max_3sat_brute_force(clauses, num_vars):
    best_assignment = None
    best_score = 0

    for assignment in product([0, 1], repeat=num_vars + 1):
        score = sum(evaluate_clause(clause, assignment) for clause in clauses)
        if score > best_score:
            best_score = score
            best_assignment = assignment

    return best_assignment, best_score


def main():
    n, m = input().split(" ")
    m = int(m)
    n = int(n)
    str_clauses = [input().split(" ") for _ in range(m)]
    clauses = [[int(literal) for literal in clause] for clause in str_clauses]
    best_assignment, best_score = max_3sat_brute_force(clauses, n)
    print(best_score)
    for i in range(1, n + 1):
        print(f"{i}: ", end="")
        if best_assignment[i] == 0:
            print("F")
        else:
            print("T")


if __name__ == "__main__":
    main()