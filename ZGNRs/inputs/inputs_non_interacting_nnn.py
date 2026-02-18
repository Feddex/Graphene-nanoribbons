# inputs_non_interacting_nnn.py

MODE = "single"   # "single" or "sweep_N"

N_SINGLE = 2
N_MIN = 1
N_MAX = 6

# Geometry
C_C = 1.42
VACUUM = 10.0
SATURATED = True
M = 1              # cells along periodic direction

# Hopping parameters (eV)
t1 = -2.6          # NN hopping
t2 =  0.10         # NNN hopping (typical graphene ~ 0.1*|t1|)

# k-grid
nk = 1000
eta = 0.05
n_E = 1000
mu = 0.0

base_dir = "postproc_outputs_noninteracting_nnn"
