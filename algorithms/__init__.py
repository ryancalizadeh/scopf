from algorithms import centralized, admm_vanilla, admm_parallel, admm_parallel_relaxed

ALGORITHMS = {
    "centralized": centralized.solve,
    "admm_vanilla": admm_vanilla.solve,
    "admm_parallel": admm_parallel.solve,
    "admm_parallel_relaxed": admm_parallel_relaxed.solve,
}
