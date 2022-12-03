
## Simulate and check
import time
import pickle
import warnings
warnings.filterwarnings('ignore')  # For a clean output: omitting "overflow encountered in exp" warning.
from report.functions import simulate

# Simulate :: NEMOS_035 - may take a while
output = []
sigma_sel = [0, 0.022, 0.15, 0.5, 1]
for sigma in sigma_sel:
    output.append(simulate("NEMOS_035", "jr", g=2, p_th=0.09, sigma=sigma, th='pTh', t=60, mode="FIG"))

## Save simulations results using pickle
open_file = open("PAPER/R4.2_sigmaadjust/gexplore_data/g_explore-FIG8b_N35-sigmaadjust-" + time.strftime("m%md%dy%Y-t%Hh.%Mm.%Ss") + ".pkl", "wb")
pickle.dump([sigma_sel, output], open_file)
open_file.close()

