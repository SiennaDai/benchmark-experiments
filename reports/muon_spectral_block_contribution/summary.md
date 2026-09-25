# 3x3 spectral block contribution analysis

Coverage: 10 formal snapshots, 300 eligible 2D Muon tensors. Runtime: 763.11s CPU.
Production INT4 dynamic b2048 and production Muon Newton--Schulz are reused unchanged.
The FP32 reduced SVD defines the established active/head/middle/tail bands; primary blocks are HH, HM, HT, MH, MM, MT, TH, TM, TT. Other-mode blocks and unresolved reduced-SVD energy are retained for accounting.
For block ij, E_ij = U_i (U_i.T E V_j) V_j.T. Direct restoration subtracts exactly E_ij from Q(M), and gains are measured against both production K=5 and exact polar readouts.

## Mean primary block results

- HH: mean_energy_fraction=0.019681661420298948, median_energy_fraction=0.018079990455876792, mean_K5_gain=0.000360617736975352, median_K5_gain=0.0003413856029510498, mean_polar_gain=0.00010662575562795004, mean_efficiency=0.019358827034561144
- HM: mean_energy_fraction=0.00886310439593218, median_energy_fraction=0.009124398499002088, mean_K5_gain=8.818308512369792e-05, median_K5_gain=5.3942203521728516e-05, mean_polar_gain=8.679896593093871e-05, mean_efficiency=0.009801757660094152
- HT: mean_energy_fraction=0.00850663949415252, median_energy_fraction=0.008636947377301476, mean_K5_gain=7.018983364105224e-05, median_K5_gain=5.194544792175293e-05, mean_polar_gain=7.479101419448853e-05, mean_efficiency=0.00879100405969896
- MH: mean_energy_fraction=0.006958049807768941, median_energy_fraction=0.006161730267803571, mean_K5_gain=8.094459772109986e-05, median_K5_gain=4.836916923522949e-05, mean_polar_gain=7.771025101343791e-05, mean_efficiency=0.010313351052253568
- MM: mean_energy_fraction=0.006385336946096066, median_energy_fraction=0.00497918937321654, mean_K5_gain=0.0021297504504521688, median_K5_gain=0.0011017322540283203, mean_polar_gain=0.0017833876609802245, mean_efficiency=0.28411016349169826
- MT: mean_energy_fraction=0.005910611007490823, median_energy_fraction=0.004328298963133271, mean_K5_gain=0.000771040121714274, median_K5_gain=0.0005024969577789307, mean_polar_gain=0.0007367116212844848, mean_efficiency=0.1245883506781766
- TH: mean_energy_fraction=0.00568555383358559, median_energy_fraction=0.003937743351802723, mean_K5_gain=4.3391088644663494e-05, median_K5_gain=3.7550926208496094e-05, mean_polar_gain=4.467487335205078e-05, mean_efficiency=0.006747769291381811
- TM: mean_energy_fraction=0.004908249303336098, median_energy_fraction=0.0037393758226578564, mean_K5_gain=0.0004450420538584391, median_K5_gain=0.0003980100154876709, mean_polar_gain=0.0004277380307515462, mean_efficiency=0.04606797205328399
- TT: mean_energy_fraction=0.004774023667528809, median_energy_fraction=0.0036534892506355863, mean_K5_gain=0.00043391048908233645, median_K5_gain=0.0010919272899627686, mean_polar_gain=0.000986579954624176, mean_efficiency=0.19657913379736214

## Grouped interpretation

Grouped restoration is direct and nonlinear; sums of individual gains are not treated as exact additive contributions.
Diagonal and off-diagonal groups, H<->M, M<->T, H<->T, and left/right row/column associations are recorded in grouped_restoration.csv.
