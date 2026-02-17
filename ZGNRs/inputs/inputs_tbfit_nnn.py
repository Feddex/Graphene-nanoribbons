# inputs_tbfit_nnn.py
BASE_OUTDIR = "postproc_outputs_spindft"

NK_TB_FACTOR = 2
FILLING = 1.0
MAX_ITER = 300
MIX = 0.10
TOL = 1e-5

USE_SCIPY_IF_AVAILABLE = True

# initial guesses
T1_0 = 2.7
T2_0 = 0.20
U0   = 2.0

# fallback grids
T1_GRID = [2.2, 2.5, 2.7, 2.9, 3.1]
T2_GRID = [0.00, 0.10, 0.20, 0.30]
U_GRID  = [1.0, 1.5, 2.0, 2.5, 3.0]
