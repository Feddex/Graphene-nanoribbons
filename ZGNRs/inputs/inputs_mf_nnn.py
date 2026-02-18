# inputs_mf_nnn.py

MODE = "single"     # "single" | "sweep_N" | "sweep_U"

# --- single
N_SINGLE = 2
U_SINGLE = 2.7

# --- sweep over N (fixed U)
N_MIN = 2
N_MAX = 8
U_FIXED_FOR_SWEEP_N = 2.7

# --- sweep over U/t1 (fixed N)
N_FIXED_FOR_SWEEP_U = 6
Umin_over_t1 = 1.0
Umax_over_t1 = 3.0
dU_over_t1   = 0.10

# Geometry (ASE)
C_C = 1.42
VACUUM = 10.0
SATURATED = True
M = 1

# Hoppings (NN and NNN)
t1 = 2.4
t2 = 0.00     #check if hte result is the same as i teh case of absence of NNN

# SCF paramteres
Nk = 400
filling = 1.0
max_iter = 300
mix = 0.10
tol = 1e-5
m0 = 0.05
verbose = True

# DOS
dos_nE = 600
dos_eta = 0.05

base_dir = "postproc_outputs_mf_nnn"
