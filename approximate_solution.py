def evaluate_clause(clause, assignment):
    for literal in clause:
        variable, is_negated = abs(literal), (literal < 0)
        if (assignment[variable] == 1 and not is_negated) or (assignment[variable] == 0 and is_negated):
            return 1
    return 0


def max_3sat_approx(clauses, num_vars):
    assignment = [0] * (num_vars + 1)
    for clause in clauses:
        selected_literal = clause[0]
        variable = abs(selected_literal)
        is_negated = (selected_literal < 0)
        assignment[variable] = 1 if not is_negated else 0
    max_satisfied_clauses = sum(1 for clause in clauses if evaluate_clause(clause, assignment))
    return assignment[1:], max_satisfied_clauses


def main():
    n, m = input().split(" ")
    m = int(m)
    n = int(n)
    str_clauses = [input().split(" ") for _ in range(m)]
    clauses = [[int(literal) for literal in clause] for clause in str_clauses]
    best_assignment, best_score = max_3sat_approx(clauses, n)
    print(best_score)
    for i in range(1, n + 1):
        print(f"{i}: ", end="")
        if best_assignment[i - 1] == 0:
            print("F")
        else:
            print("T")


if __name__ == "__main__":
    main()
