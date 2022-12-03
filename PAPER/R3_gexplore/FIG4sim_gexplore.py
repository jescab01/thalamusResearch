
## Simulate and check
import time
import pickle
import warnings
warnings.filterwarnings('ignore')  # For a clean output: omitting "overflow encountered in exp" warning.
from report.functions import simulate

# Simulate :: NEMOS_035 - may take a while
output = []
g_sel = [2, 5, 15, 35]
for g in g_sel:
    output.append(simulate("NEMOS_035", "jr", g=g, p_th=0.09, sigma=0.022, th='pTh', t=60, mode="FIG"))

## Save simulations results using pickle
open_file = open("PAPER/R3_gexplore/gexplore_data/g_explore-FIG4_N35-initialConditions-" + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss") + ".pkl", "wb")
pickle.dump([g_sel, output], open_file)
open_file.close()

