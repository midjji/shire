%% the final version ! result 007, 7k images
close all; clc;clear;
shire = [ 1, 0.748638, 0.748638, 0.748638, 0.748638, 0.748638, 0.748638, 0.748638, 0.748638, 0.748638, 0.748638, 0.748638, 0.748638, 0.748638, 0.748638, 0.748638, 0.748638, 0.748638, 0.748638, 0.748638, 0.748638, 0.747583, 0.746177, 0.745825, 0.745649, 0.745649, 0.745649, 0.745474, 0.744771, 0.744595, 0.744595, 0.74354, 0.742837, 0.74231, 0.740728, 0.737915, 0.736333, 0.735103, 0.731236, 0.728423, 0.724556, 0.722095, 0.718931, 0.715591, 0.710845, 0.707154, 0.701705, 0.697311, 0.691158, 0.68483, 0.678854, 0.668131, 0.655827, 0.649147, 0.642819, 0.636667, 0.628054, 0.616629, 0.608894, 0.598523, 0.589383, 0.578133, 0.568641, 0.557567, 0.545439, 0.531904, 0.521181, 0.510459, 0.497275, 0.481455, 0.466339, 0.453507, 0.437863, 0.420812, 0.407101, 0.391457, 0.37511, 0.361399, 0.346458, 0.329056, 0.317279, 0.295834, 0.276323, 0.259448, 0.23607, 0.21357, 0.186149, 0.159255, 0.137282, 0.115838, 0.0956231, 0.0734751, 0.0481631, 0.0283002, 0.017402, 0.0108982, 0.00562489, 0.00228511, 0.000351556, 0 ]
shire_bootstrap_labels = [ 1, 0.841448, 0.841448, 0.841448, 0.841448, 0.841448, 0.841448, 0.841448, 0.841448, 0.841448, 0.841448, 0.841448, 0.841448, 0.841448, 0.841448, 0.841448, 0.841448, 0.841448, 0.841448, 0.841448, 0.841448, 0.841097, 0.840042, 0.839691, 0.83846, 0.83723, 0.836175, 0.83512, 0.834066, 0.83266, 0.831078, 0.82721, 0.822464, 0.817718, 0.813324, 0.808402, 0.803832, 0.799613, 0.796098, 0.792582, 0.787836, 0.783618, 0.780278, 0.777817, 0.774829, 0.772895, 0.769907, 0.767622, 0.764809, 0.760591, 0.757778, 0.755317, 0.750923, 0.74688, 0.741958, 0.736861, 0.731411, 0.725435, 0.720162, 0.712955, 0.706275, 0.700123, 0.691158, 0.682897, 0.670944, 0.656882, 0.642116, 0.632273, 0.618211, 0.600633, 0.58411, 0.56829, 0.552118, 0.535243, 0.514677, 0.495518, 0.476358, 0.454386, 0.428371, 0.408508, 0.384602, 0.356829, 0.327474, 0.296713, 0.268413, 0.235894, 0.205836, 0.175426, 0.144665, 0.113552, 0.0875374, 0.0627527, 0.0416593, 0.0246089, 0.0144138, 0.00791, 0.003164, 0.000527333, 0, 0 ]
shire_labels = [ 1, 0.695904, 0.695904, 0.695904, 0.695904, 0.695904, 0.695904, 0.695904, 0.695904, 0.695904, 0.695904, 0.695904, 0.695904, 0.695904, 0.695904, 0.695904, 0.695904, 0.695904, 0.695904, 0.695904, 0.695904, 0.695553, 0.694498, 0.694498, 0.693971, 0.693443, 0.693268, 0.692916, 0.692565, 0.692213, 0.691861, 0.690983, 0.690455, 0.690104, 0.689576, 0.688697, 0.687467, 0.686588, 0.685885, 0.684127, 0.683073, 0.681842, 0.68026, 0.678502, 0.676217, 0.673756, 0.672174, 0.668659, 0.665319, 0.662507, 0.660221, 0.655827, 0.652311, 0.64739, 0.642292, 0.635964, 0.62946, 0.621726, 0.615925, 0.607488, 0.600105, 0.590262, 0.580594, 0.569169, 0.559852, 0.547724, 0.535068, 0.525048, 0.513095, 0.497803, 0.484092, 0.469327, 0.455616, 0.43839, 0.423273, 0.409562, 0.394094, 0.381262, 0.367376, 0.348567, 0.330462, 0.311127, 0.293725, 0.274741, 0.255054, 0.23062, 0.201617, 0.173493, 0.148708, 0.125681, 0.101248, 0.0747056, 0.0485147, 0.0279487, 0.0147653, 0.00615222, 0.00263667, 0.000175778, 0, 0 ]
shire_bootstrap_no_labels = [ 1, 0.735454, 0.735454, 0.735454, 0.735454, 0.735454, 0.735454, 0.735454, 0.735454, 0.735454, 0.735454, 0.735454, 0.735454, 0.735454, 0.735454, 0.735454, 0.735454, 0.735454, 0.735454, 0.735454, 0.735454, 0.735279, 0.734751, 0.734048, 0.733872, 0.733697, 0.733345, 0.732818, 0.73229, 0.731763, 0.730884, 0.728599, 0.725787, 0.723677, 0.721919, 0.720513, 0.71981, 0.718931, 0.71647, 0.713658, 0.709088, 0.705748, 0.702232, 0.69942, 0.695377, 0.687643, 0.679909, 0.673229, 0.666022, 0.657057, 0.648093, 0.640886, 0.632624, 0.626121, 0.618035, 0.609422, 0.599754, 0.590086, 0.579364, 0.568114, 0.558973, 0.549833, 0.535947, 0.526103, 0.513095, 0.500967, 0.487783, 0.476534, 0.462471, 0.44964, 0.436281, 0.421515, 0.405695, 0.393742, 0.37968, 0.365091, 0.351907, 0.335208, 0.319213, 0.301107, 0.280893, 0.261206, 0.242398, 0.22148, 0.203726, 0.185094, 0.160309, 0.135349, 0.117068, 0.093338, 0.07119, 0.0515029, 0.0330462, 0.018984, 0.00861311, 0.00351556, 0.00193356, 0.00123044, 0.000351556, 0.000175778 ]

close all;
x=0:0.01:1; x=x(1:100);
figure()
plot(x,shire,x, shire_labels,x, shire_bootstrap_no_labels,x,shire_bootstrap_labels,'LineWidth',4); 
title('Intersection Over Union (IoU) Scores');
ylabel('Best iou > threshold'); xlabel('Threshold'); ylim([0,1]);
legend('Shire','Shire-labels','Shire-bootstrap','Shire-bootstrap-labels')

