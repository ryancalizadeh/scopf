from algorithms import centralized, admm_vanilla, admm_parallel

ALGORITHMS = {
    "centralized": centralized.solve,
    "admm_vanilla": admm_vanilla.solve,
    "admm_parallel": admm_parallel.solve
}
