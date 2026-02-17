# inputs_non_interacting_nnn.py

MODE = "single"   # "single" or "sweep_N"

N_SINGLE = 2
N_MIN = 1
N_MAX = 6

# Geometry (ASE graphene_nanoribbon)
C_C = 1.42         # Å
VACUUM = 10.0      # Å
SATURATED = True   # H-terminated edges
M = 1              # cells along periodic direction (keep 1 for TB Bloch)

# Hopping parameters (eV)
t1 = -2.6          # NN hopping (sign convention: usually negative)
t2 =  0.10         # NNN hopping (typical graphene ~ 0.1*|t1|)

# k-grid
nk = 1000
eta = 0.05
n_E = 1000
mu = 0.0

base_dir = "postproc_outputs_noninteracting_nnn"
