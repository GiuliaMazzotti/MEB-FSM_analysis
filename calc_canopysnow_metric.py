import xarray as xr
import numpy as np
from datetime import datetime

def descripteurs_canopy(ds_model,ds_meteo,period,FSM=False):
    start_time,end_time=period[0],period[1]
    ds_winter = ds_model.sel(time=slice(start_time, end_time))  # On selectionne le data set sur la période désirée
    ds_prec = ds_meteo.sel(time=slice(start_time, end_time))
    ds_prec = ds_prec.assign_coords(time=ds_winter.time)
    if FSM == True :
        ds_winter = ds_winter.rename({'swet_for': 'swet_curr'})
        ds_winter = ds_winter.rename({'sbsc_for': 'sbsc_curr'})
        ds_winter = ds_winter.rename({'rotc_for': 'rotc_curr'})
        ds_prec['Prec_cum'] = ds_prec['Prec_sum_for'].cumsum(dim='time')

    else :
        ds_winter = ds_winter.rename({'swet_all': 'swet_curr'})
        ds_winter = ds_winter.rename({'SUBL_isba': 'sbsc_curr'})
        ds_winter = ds_winter.rename({'runoff': 'rotc_curr'})
        ds_prec['Prec_cum'] = ds_prec['Prec_sum'].cumsum(dim='time')

    ds_winter['swe_depletion']  = xr.ufuncs.add(ds_winter['sbsc_curr'],ds_winter['rotc_curr'])
    ds_winter['Depl_cum'] = ds_winter['swe_depletion'].cumsum(dim='time')
    seuil=10 ## seuil pour l"enneigement continu

    ### PSD 
    idx_max=ds_winter['swet_curr'].argmax(dim='time') ## on récupère l'indice du max
    PSD=ds_winter['time'].isel(time=idx_max) ## On récupère le jour du max de swe

    ### SSD et SDD
    valid_times1=ds_winter.where((ds_winter['swet_curr']<seuil)&(ds_winter['time']<PSD))  ## on récupère les valeurs valides antérieur au PSD
    valid_times2= ds_winter.where((ds_winter['swet_curr']<seuil)&(ds_winter['time']>PSD)) ## On recupére les dates valides postérieur au PSD
        
    index_start=valid_times1["swet_curr"].notnull().cumsum(dim='time').argmax(dim='time')
    index_end_tmp=valid_times2["swet_curr"].notnull().cumsum(dim='time')
    index_end = index_end_tmp.where(index_end_tmp > 0, np.inf).argmin(dim='time')

    input = ds_prec['Prec_cum'].isel(time=index_end) # - ds_prec['Prec_cum'].isel(time=index_start)   
    output = ds_winter['Depl_cum'].isel(time=index_end) # - ds_winter['Depl_cum'].isel(time=index_start)
    Canopy_loss= xr.ufuncs.subtract(input,output)
   # swe_residual = ds_winter['swet_curr'].isel(time=index_end) - ds_winter['swet_curr'].isel(time=index_start)
   # Canopy_loss = xr.ufuncs.subtract(canopy_loss_raw,swe_residual)

    return(Canopy_loss)

## psf = 1
## psr = 0.1

#Sf(i,j) = pmultf(i,j)*Sf(i,j)
# Sf(i,j) = (psf - psr*fveg(i,j))*Sf(i,j) 
# pmultf = (1 - (1-prec_multi)*(1-forest*pmultwf_for))/lus.prec_multi;
"""
canopy snow loss calculation equivalent to calc_snow_metrics
    
"""

path_snow = "data/FORCING_snow.nc"
path_rain = "data/FORCING_rain.nc"
path_fsm=r'data/FSM_all_corr.nc'
path_preccor=r'data/prec_multi_for.nc'
ds_snow=xr.open_dataset(path_snow,decode_times=True)
ds_rain=xr.open_dataset(path_rain,decode_times=True)
ds_fsm=xr.open_dataset(path_fsm,decode_times=True)
ds_prec = xr.open_dataset(path_preccor,decode_times=True)
ds_snow = ds_snow.coarsen(time=24, boundary="trim").sum()
ds_rain = ds_rain.coarsen(time=24, boundary="trim").sum()
ds_snow['Snowf_sum'] = ds_snow['Snowf']*3600
ds_snow['Snowf_sum_for'] = xr.ufuncs.multiply(ds_snow['Snowf_sum'],ds_prec['prec_multi_for'])
ds_rain['Rainf_sum'] = ds_rain['Rainf']*3600
ds_snow['Prec_sum'] = xr.ufuncs.add(ds_snow['Snowf_sum'],ds_rain['Rainf_sum'])
ds_snow['Prec_sum_for'] = xr.ufuncs.add(ds_snow['Snowf_sum_for'],ds_rain['Rainf_sum'])
ds_snow = ds_snow.assign_coords(xx=ds_fsm.xx.values,yy=ds_fsm.yy.values)
ds_rain = ds_rain.assign_coords(xx=ds_fsm.xx.values,yy=ds_fsm.yy.values)

# Définition des chemins vers les fichiers NetCDF
simulations = {
    "FSM_FOR": "data/FSM_all.nc",
    "MEB_FOR": "data/MEBCro_for.nc",
    "MXT_FOR": "data/MEBCroXT_for.nc"
}
# Liste avec les saisons
seasons = [f"{year}/{year+1}" for year in range(2016, 2023)]

# Liste des paramètres à extraire
params = ["Canopy_loss"]

# Dictionnaire pour stocker les datasets des simulations
data_dict = {}
xx = ds_fsm.xx
yy = ds_fsm.yy

for sim_name, sim_path in simulations.items():
    print(f"Traitement de {sim_name}...") ## On se balade dans tt les simulations
    season_datasets = []

    ds_model=xr.open_dataset(sim_path,decode_times=True)
    for season in seasons: ## On calcul les descripteurs pour chaque saison
        year_start = int(season.split("/")[0])

        print(f"  -> Calcul de la saison {season}...")

        # On specifie les arguments pour fsm car open et forest sont dans le même .nc
        kwargs = {}
        if "FSM" in sim_name :
            kwargs["FSM"] = True
        else : 
            kwargs["FSM"] = False

        # On récupere les differents descripteurs cf fonction descripteur
        results = descripteurs_canopy(ds_model, ds_snow, [datetime(year_start, 10, 2, 6, 0), datetime(year_start + 1, 6, 30, 5, 0)], **kwargs)

        # Création du Dataset pour cette saison
        season_ds = xr.Dataset(
            {param: (["yy", "xx"], results.data) for i, param in enumerate(params)}, ## Chaque descripteur a comme dimenesion (xx,yy)
            
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
output_path = f'data/canopysnow_metrics_v7.nc'
ds_globale.to_netcdf(output_path)
print(f"NetCDF file saved : {output_path}")



