import numpy as np
import math
import h5py as h5
import sklearn
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection
import pandas as pd
import argparse
import os,sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--datafile", type=str, default=None,
                    dest="datafile", help="h5 file containing halo data")
parser.add_argument("-k", "--kernels", type=str, nargs="+", choices=["linear","poly","rbf","sigmoid"], default=None,
                    dest="kernel", help="SVM kernel for decision boundary")
parser.add_argument("-d", "--degrees", type=int, nargs="+", default=0,
                    dest="degree", help="degree of polynomial kernel")
parser.add_argument("-C", "--logCs", type=float, nargs="+", default=0.0,
                    dest="logC", help="log of regularization parameter")
parser.add_argument("-w", "--logws", type=float, nargs="+", default=0.0,
                    dest="logw", help="log of class weight")
parser.add_argument("-c", "--code", type=str, choices=["dm_main","dm_full","gas_main","gas_full","star_main","star_full"], default=None,
                    dest="code", help="code for feature subset")
parser.add_argument("-s", "--seed", type=int, default=0,
                    dest="seed", help="RNG seed for reproducibility")
parser.add_argument("-o", "--outfolder", type=str, default=None,
                    dest="outfolder", help="folder to save output to")
args = parser.parse_args()

if (args.seed != None):
    np.random.seed(args.seed)

halos = {
    "mass": [],
    "density": [],
    "temp": [],
    "metallicity": [],
    "H2_fraction": [],
    "tan_vel": [],
    "rad_vel": [],
    "LW": [],
    "sp_gas": [],
    "sp_dm": [],
    "rms_vel": [],
    "closest_galaxy": [],
    "overdensity": [],
    "radial_mass_flux": [],
    "av_mass": [],
    "av_dMdt": [],
    "av_dMdz": [],
    "t1": [],
    "stellar_mass": [],
    "dcbh": []
}

f = h5.File(args.datafile, "r")
for halo in f.keys():
    halos["mass"].append(f[f"{halo}/mass-Msun"][()])
    halos["density"].append(f[f"{halo}/density-g_cm**3"][()])
    halos["temp"].append(f[f"{halo}/temperature-K"][()])
    halos["metallicity"].append(f[f"{halo}/metallicity"][()])
    halos["H2_fraction"].append(f[f"{halo}/H2_fraction"][()])
    halos["tan_vel"].append(f[f"{halo}/tangential_velocity-cm_s"][()])
    halos["rad_vel"].append(f[f"{halo}/radial_velocity-cm_s"][()])
    halos["LW"].append(f[f"{halo}/J_LW-erg_cm**2"][()])
    halos["sp_gas"].append(f[f"{halo}/spin_parameter_gas"][()])
    halos["sp_dm"].append(f[f"{halo}/spin_parameter_gas"][()])
    halos["rms_vel"].append(f[f"{halo}/rms_turbulent_velocity-km_s"][()])
    halos["closest_galaxy"].append(f[f"{halo}/closest_galaxy-kpc"][()])
    halos["overdensity"].append(f[f"{halo}/overdensity"][()])
    halos["radial_mass_flux"].append(f[f"{halo}/radial_mass_flux-g_s"][()])
    halos["av_mass"].append(f[f"{halo}/average_mass-Msun"][()])
    halos["av_dMdt"].append(f[f"{halo}/average_dMdt-Msun_Myr"][()])
    halos["av_dMdz"].append(f[f"{halo}/average_dMdz-Msun"][()])
    halos["t1"].append(f[f"{halo}/t1_tidal_field"][()])
    halos["stellar_mass"].append(f[f"{halo}/stellar_mass-Msun"][()])
    halos["dcbh"].append(f[f"{halo}/dcbh_host"][()])
f.close()

halos_df = pd.DataFrame.from_dict(halos)
halos_df_clean = halos_df.dropna()

# Transformations
halos_df = pd.DataFrame.from_dict(halos)

halos_df_transform = pd.DataFrame()
halos_df_transform["mass"] = np.log10(halos_df_clean["mass"])
halos_df_transform["density"] = np.log10(halos_df_clean["density"])
halos_df_transform["temp"] = np.log10(halos_df_clean["temp"])
halos_df_transform["metallicity"] = np.log10(halos_df_clean["metallicity"])
halos_df_transform["H2_fraction"] = np.log10(halos_df_clean["H2_fraction"])
halos_df_transform["tan_vel"] = np.log10(1+halos_df_clean["tan_vel"])
halos_df_transform["rad_vel_sign"] = np.sign(halos_df_clean["rad_vel"])
halos_df_transform["rad_vel"] = np.log10(1+np.abs(halos_df_clean["rad_vel"]))
halos_df_transform["LW"] = np.log10(halos_df_clean["LW"])
halos_df_transform["sp_gas"] = np.log10(halos_df_clean["sp_gas"])
halos_df_transform["sp_dm"] = np.log10(halos_df_clean["sp_dm"])
halos_df_transform["rms_vel"] = np.log10(1+halos_df_clean["rms_vel"])
halos_df_transform["closest_galaxy"] = np.log10(halos_df_clean["closest_galaxy"])
halos_df_transform["overdensity"] = np.log10(halos_df_clean["overdensity"])
halos_df_transform["radial_mass_flux_sign"] = np.sign(halos_df_clean["radial_mass_flux"])
halos_df_transform["radial_mass_flux"] = np.log10(1+np.abs(halos_df_clean["radial_mass_flux"]))
halos_df_transform["av_mass"] = np.log10(halos_df_clean["av_mass"])
halos_df_transform["av_dMdt_sign"] = np.sign(halos_df_clean["av_dMdt"])
halos_df_transform["av_dMdt"] = np.log10(1+np.abs(halos_df_clean["av_dMdt"]))
halos_df_transform["av_dMdz_sign"] = np.sign(halos_df_clean["av_dMdz"])
halos_df_transform["av_dMdz"] = np.log10(np.abs(1+halos_df_clean["av_dMdz"]))
halos_df_transform["t1"] = np.log10(1+halos_df_clean["t1"])
halos_df_transform["stellar_mass"] = np.log10(1+halos_df_clean["stellar_mass"])
halos_df_transform["dcbh"] = halos_df_clean["dcbh"]

halos_df_transform_clean = halos_df_transform.dropna()

print("Total halos:", len(halos_df_clean["dcbh"]))
print("DCBH halos:", int(np.sum(halos_df_clean["dcbh"])))

X = halos_df_transform_clean.drop("dcbh", axis=1)
y = halos_df_transform_clean["dcbh"]


parameter_codes = {
    "dm_main": ["mass", "sp_dm"],
    "dm_full": ["mass", "sp_dm", "av_dMdz_sign", "av_dMdz"],
    "gas_main": ["mass", "temp", "rad_vel_sign", "rad_vel", "sp_gas", "sp_dm", "rms_vel", "radial_mass_flux_sign", "radial_mass_flux"],
    "gas_full": ["mass", "density", "temp", "rad_vel_sign", "rad_vel", "sp_gas", "sp_dm", "rms_vel", "radial_mass_flux_sign", "radial_mass_flux", "av_dMdz_sign", "av_dMdz"],
    "star_main": ["mass", "temp", "metallicity", "rad_vel_sign", "rad_vel", "LW", "sp_gas", "sp_dm", "rms_vel", "radial_mass_flux_sign", "radial_mass_flux", "stellar_mass"],
    "star_full": ["mass", "density", "temp", "metallicity", "H2_fraction", "rad_vel_sign", "rad_vel", "LW", "sp_gas", "sp_dm", "rms_vel", "radial_mass_flux_sign", "radial_mass_flux", "av_dMdz_sign", 
                  "av_dMdz", "stellar_mass"]
}
grid_search_features = parameter_codes[args.code]

for feature in X.keys():
    if feature not in grid_search_features:
        X.drop(feature, axis=1, inplace=True)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# Grid search params

# The following lines provide full functionality for a grid search performed in a single script
svm_params = []
for kernel in args.kernels:
    if (kernel == "poly"):
        svm_params.append({
            "kernel": [kernel],
            "degree": args.degrees,
            "C": [10**logC for logC in args.logCs],
            "class_weight": [{1: logw} for logw in args.logws]
        })
    else:
        svm_params.append({
            "kernel": [kernel],
            "C": [10**logC for logC in args.logCs],
            "class_weight": [{1: logw} for logw in args.logws]
        })

# The following lines were used when the grid search was parallelized outside the script using HPC resources (that is, only one value is passed for each hyperparameter)
# svm_params = {
#     "kernel": args.kernels,
#     "C": [10**args.logCs[0]],
#     "class_weight": [{1: 10**args.logws[0]}]
# }
#
# if (args.kernels[0] == "poly"):
#     svm_params["degree"] = args.degrees

scoring = ["precision", "recall", "f1", "precision_weighted", "recall_weighted", "f1_weighted"]

t1 = time.time()
classifier = svm.SVC()
classifier_gs = model_selection.GridSearchCV(classifier, svm_params, scoring=scoring, refit="f1", verbose=2, error_score="raise")
classifier_gs.fit(X_train, y_train)
t2 = time.time()
y_pred = classifier_gs.predict(X_test)

print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
print("F1:", metrics.f1_score(y_test, y_pred))
print("Weighted precision:", metrics.precision_score(y_test, y_pred, average="weighted"))
print("Weighted recall:", metrics.recall_score(y_test, y_pred, average="weighted"))
print("Weighted F1:", metrics.f1_score(y_test, y_pred, average="weighted"))

print("Grid search time: %.2f min"%((t2-t1)/60))

outfile = f"{args.outfolder}results_{args.code}"
if (len(args.kernels) == 1):
    k = args.kernels[0]
    if (k == "poly") and (len(args.degrees) == 0):
        d = args.degrees[0]
        outfile += f"_{k}{d}"
    else:
        outfile += f"_{k}"
if (len(args.logCs) == 1):
    C = args.logCs[0]
    outfile += f"_logC{C}"
if (len(args.logws) == 1):
    w = args.logws[0]
    outfile += f"_logw{w}"
outfile += ".npy"

np.save(outfile, classifier_gs.cv_results_)
print("Saving results to %s"%outfile)
