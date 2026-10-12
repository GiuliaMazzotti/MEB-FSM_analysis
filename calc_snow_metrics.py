### Code to compute snow descriptors from simulation output 
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime

"""
Based on Felix pour_GIULIA.py script
Fonction permettant de coder l'ensemble des descripteurs suivant Peak-SWE,SSD,SDD,PSD et ablation rate
input: path   = chemin vers le fichier .nc
       period = liste contenant une date de début et de fin avec comme format attendu :[datetime(YYYY,MM,JJ,HH,SS),datetime(YYYY,MM,JJ,HH,SS)]
       Forest et Open sont des options a mettre en True uniquement si on veut exploiter des fichiers venant de fsm
"""
def descripteurs(path,period,Forest=False,Open=False,All=False):
    ds=xr.open_dataset(path,decode_times=True)
    seuil=10 ## seuil pour l"enneigement continu
    start_time,end_time=period[0],period[1] ## On récupere la période d'étude 
    ds_winter = ds.sel(time=slice(start_time, end_time))  # On selectionne le data set sur la période désirée
    if Forest == True : 
        ds_winter = ds_winter.rename({'swet_for': 'swet_curr'})
        ds_winter = ds_winter.where(ds_winter['swet_curr']>=0,0)
    elif Open == True :
        ds_winter = ds_winter.rename({'swet_opn': 'swet_curr'})
    elif All == True :    
        ds_winter = ds_winter.rename({'swet_all_corr': 'swet_curr'})
    else : 
        ds_winter = ds_winter.rename({'swet_all': 'swet_curr'})

    ### Peak-SWE
    Peak_SWE=ds_winter['swet_curr'].max(dim='time') ## on récupère le swe max et on s'assure de regarder uniquement sur des valeurs +
    ### PSD 
    idx_max=ds_winter['swet_curr'].argmax(dim='time') ## on récupère l'indice du max
    PSD=ds_winter['time'].isel(time=idx_max) ## On récupère le jour du max de swe

    ### SSD et SDD
    valid_times1=ds_winter.where((ds_winter['swet_curr']<seuil)&(ds_winter['time']<PSD))  ## on récupère les valeurs valides antérieur au PSD
    valid_times2= ds_winter.where((ds_winter['swet_curr']<seuil)&(ds_winter['time']>PSD)) ## On recupére les dates valides postérieur au PSD
    
    index_start=valid_times1["swet_curr"].notnull().cumsum(dim='time').argmax(dim='time')
    index_end_tmp=valid_times2["swet_curr"].notnull().cumsum(dim='time')
    index_end = index_end_tmp.where(index_end_tmp > 0, np.inf).argmin(dim='time')

    SSD=ds_winter['time'].isel(time=index_start+1)                                      ## On prend le jour d'après pour avoir + de 10mm
    SDD=ds_winter['time'].isel(time=index_end-1)
    ##SCD
    Snow_duration=(SDD-SSD).dt.days                                                      ## On récupére le nombre de jour avec .days
    Snow_duration=Snow_duration.where(Peak_SWE>0)

    ## Ablation rate
    Ablation_rate=Peak_SWE/(SDD-PSD).dt.days
       
    return(Peak_SWE,PSD,SSD,SDD,Snow_duration,Ablation_rate)




"""
Ce Script permet de creer un fichier .nc regroupant pour l'ensemble des simulations les descripteurs suivants:
"Peak_SWE", "PSD", "SSD", "SDD", "Snow_duration", "Ablation_rate"

Le script utilise la fonction descripteur qui calcule l'ensemble des descripeurs explicité ci-dessus
Bref explication du code:
    .  Pour chaque simulation on va calculer les descripteurs pour l'ensemble des saisons.
    . Ainsi pour chaque simulation:
         On  initalise une liste vide dans laquelle on ajoute les dataset contenant l'ensembles des descripteurs
         Une fois l'ensemble des saisons calculé on merge les dataset pour n'en creer qu'un seul par simulation.
         On obtient donc pour une simulation un datatset contenznt l'ensemble des descripteurs pour chaque année
    Une fois l'ensemble des simulations calculé on obtient n dataset avec n le nombre de simulation et on concatene le 
    tout dans un seul dataset.

    Au final si on veut acceder au DESCRIPTEURS_X de la simulation_Y pour la saison XXXX/YYYY il suffit de faire la commande suivante.
     
     DESCRIPTEURS_X=ds_global.sel(simulation="simulation_Y",season="XXXX/YYYY").DESCRIPTEURS_X
     Avec ds_global=xr.opendataset('chemin/vers/fichier.nc')
    
"""


# Définition des chemins vers les fichiers NetCDF
simulations = {
    "FSM_ALL": "data/FSM_all_corr.nc",
    "FSM_OPN": "data/FSM_all.nc",
    "FSM_FOR": "data/FSM_all.nc",
    "MEB_DEF": "data/MEBCro_def.nc",
    "CRO_OPN": "data/Cro_opn.nc",
    "MEB_FOR": "data/MEBCro_for.nc",
    "MEB_ALL": "data/MEBCro_all_corr.nc",
    "MXT_FOR": "data/MEBCroXT_for.nc",
    "MXT_ALL": "data/MEBCroXT_all.nc"
}
# Liste avec les saisons
seasons = [f"{year}/{year+1}" for year in range(2016, 2023)]

# Liste des paramètres à extraire
params = ["Peak_SWE", "PSD", "SSD", "SDD", "Snow_duration", "Ablation_rate"]

# Dictionnaire pour stocker les datasets des simulations
data_dict = {}
ds_fsm = xr.open_dataset('data/FSM_all.nc')
xx = ds_fsm.xx
yy = ds_fsm.yy
for sim_name, sim_path in simulations.items():
    print(f"Traitement de {sim_name}...") ## On se balade dans tt les simulations
    season_datasets = []
    for season in seasons: ## On calcul les descripteurs pour chaque saison
        year_start = int(season.split("/")[0])

        print(f"  -> Calcul de la saison {season}...")

        # On specifie les arguments pour fsm car open et forest sont dans le même .nc
        kwargs = {}
        if "FSM" in sim_name :
            if "FOR" in sim_name :
                kwargs["Forest"] = True
            elif "OPN" in sim_name :
                kwargs["Open"] = True
            else : 
                kwargs["All"] = True   
        elif sim_name == 'MEB_ALL':
           kwargs["All"] = True   
        else :
           kwargs["Forest"],kwargs["Open"],kwargs["All"] = False, False, False

        # On récupere les differents descripteurs cf fonction descripteur
        results = descripteurs(sim_path, [datetime(year_start, 9, 2, 6, 0), datetime(year_start + 1, 7, 31, 5, 0)], **kwargs)

        # Création du Dataset pour cette saison
        season_ds = xr.Dataset(
            {param: (["yy", "xx"], results[i].data) for i, param in enumerate(params)}, ## Chaque descripteur a comme dimenesion (xx,yy)
            
            coords={"yy": yy, "xx": xx, "season": season}
        )

        # Ajout du dataset de la saison à la liste
        season_datasets.append(season_ds)

    # Concaténation des saisons sur la dimension "season"
    data_dict[sim_name] = xr.concat(season_datasets, dim="season")

# Fusion des datasets en ajoutant une dimension "simulation"
ds_globale = xr.concat(list(data_dict.values()), dim="simulation")

# Ajout des noms de simulation et de season en tant que coordonnées
ds_globale = ds_globale.assign_coords(
    simulation=list(data_dict.keys()),
    season=seasons
)

# Save NetCDF
output_path = f'data/snow_metrics.nc'
ds_globale.to_netcdf(output_path)
print(f"NetCDF file saved : {output_path}")




