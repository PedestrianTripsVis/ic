# General Imports
import sys
import geopandas as gpd #pip install geopandas descartes
import pandas as pd #pip install pandas
import matplotlib.pyplot as plt # pip install matplotlib
import numpy as np
from pyproj import Proj, transform

from multiprocessing import Pool
from dbfread import DBF #pip install dbfread
from simpledbf import Dbf5 #pip install simpledbf

# Local imports
sys.path.append('/home/milenacsilva/Área de Trabalho/ic')
import scripts.utils as utils

def process_data(YEAR, DATASETS_DIR, MY_DATASET):
    trips = pd.read_csv(f'{DATASETS_DIR}{MY_DATASET}')


    trips_count = len(trips)

    df = pd.DataFrame({'Year': [YEAR], 'Entries Count': [trips_count]})
    df.plot.bar(x='Year', y='Entries Count', rot=0)
    print("Number of entries in %s dataset: %s", YEAR, trips_count)


    CHECK_ORIGIN_ATTRIBUTE = [x.__contains__('FE_VIA') for x in [trips.columns]]

    if not(trips.columns.__contains__('FE_VIA')):
        raise Exception("Error, the dataset does not uses FE_VIA as attribute name.")

    print("There is missing data 'FE_VIA' attribute?:", not(all(trips[['FE_VIA']].notna().values)))



    incomplete_trips = trips[trips['FE_VIA'].isna()]

    df = pd.DataFrame({'Entries': [trips_count-len(incomplete_trips), len(incomplete_trips)]},
                        index=['Complete', 'Incomplete'])

    df.plot.pie(subplots=True, autopct='%1.1f%%', figsize=(15,10), explode=(0.04,0), fontsize=18)
    print("Number of incomplete entries in the dataset:", len(incomplete_trips))

    trips.dropna(subset=['FE_VIA'], inplace=True)


    expanded_trips_count = trips[['FE_VIA']].sum()[0]

    base_count = expanded_trips_count/1000000

    df = pd.DataFrame({'Year': [YEAR], 'Trips Count': [expanded_trips_count/1000000]})
    ax = df.plot.bar(x='Year', y='Trips Count', rot=0)
    ax.set_ylim(top=55)
    ax.set_ylabel("Trips in millions")
    plt.title("Number of trips per year")

    for p in ax.patches[1:]:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy() 
        ax.annotate('{:.0%}'.format((height/base_count) - 1) + ' increase', (x, y + height + 1.1))

    print("Total of %i million trips in the dataset" % expanded_trips_count)


    rounded_expanded_trips_count = trips[['FE_VIA']].astype(int).sum()[0]

    percentual = 100*(expanded_trips_count - rounded_expanded_trips_count)/expanded_trips_count

    df = pd.DataFrame({'Year': [YEAR],
                       'Trips Count': [expanded_trips_count],
                       'Rounded Count': [rounded_expanded_trips_count]
                      })

    fig, ax1 = plt.subplots()
    ax1.set_ylabel('Number of trips')

    df.plot.bar(x='Year', y='Trips Count', ax=ax1, color='red', rot=0)
    df.plot.bar(x='Year', y='Rounded Count', ax=ax1, rot=0)

    print("""- Dataset:
          \tExpanded Trips (float): %i
          \tRounded Trips (int): %i
          \tLost trips: %i, %f%%""" % 
          (expanded_trips_count, rounded_expanded_trips_count, expanded_trips_count - rounded_expanded_trips_count, percentual))

    trips['FE_VIA'] = trips[['FE_VIA']].astype(int)

    if not(trips.columns.__contains__('ZONA_O') or trips2.columns.__contains__('ORIGIN_ID')):
        raise Exception("Error, the dataset does not uses ZONA_O or ORIGIN_ID as attribute name.")

    trips.rename(columns={"ZONA_O": "ORIGIN_ID"}, inplace=True)

    trips['ORIGIN_ID'] = trips[['ORIGIN_ID']].astype(int)


    trips.dropna(subset=['ORIGIN_ID'], inplace=True)


    if not(trips.columns.__contains__('ZONA_D') or trips.columns.__contains__('DESTINATION_ID')):
        raise Exception("Error, the dataset does not uses ZONA_D or DESTINATION_ID as attribute name.")

    trips.rename(columns={"ZONA_D": "DESTINATION_ID"}, inplace=True)

    trips['DESTINATION_ID'] = trips[['DESTINATION_ID']].astype(int)


    trips.dropna(subset=['DESTINATION_ID'], inplace=True)





    trips.groupby(by=['ORIGIN_ID', 'DESTINATION_ID']).sum()[['FE_VIA']].sort_values(by=['FE_VIA', 'ORIGIN_ID', 'DESTINATION_ID']).describe()



    total_trips = trips[['FE_VIA']].sum()[0]
    print("Total expanded trips: ", total_trips)

    edges_weights = trips.sort_values(by='FE_VIA')
    edges_weights[['PERCENTUAL_VALUE']] = edges_weights[['FE_VIA']]*100/total_trips

    edges_weights = edges_weights.groupby(by='FE_VIA').sum().reset_index()
    edges_weights[['PERCENTUAL_VALUE']] = edges_weights[['PERCENTUAL_VALUE']].cumsum()
    edges_weights.plot.bar(x='FE_VIA',y='PERCENTUAL_VALUE', figsize=(16,8))



    REDUCTION_FACTOR = 10

    reduced_trips = trips.copy().loc[trips['FE_VIA'] >= REDUCTION_FACTOR]
    reduced_trips['FE_VIA'] = reduced_trips['FE_VIA'].div(REDUCTION_FACTOR).astype(int)


    if all(reduced_trips[['CO_O_X', 'CO_O_Y', 'CO_D_X', 'CO_D_Y']].notna()):
        print("There is no missing coordinates data, you can proceeed!")


    geo_trips_origins = gpd.GeoDataFrame(
        reduced_trips[['CO_O_X', 'CO_O_Y']], geometry=gpd.points_from_xy(reduced_trips.CO_O_X, reduced_trips.CO_O_Y))

    geo_trips_origins.crs = {'init': 'epsg:22523'}
    geo_trips_origins.to_crs({'proj': 'longlat', 'ellps': 'WGS84', 'no_defs': True}, inplace=True)

    geo_trips_destinations = gpd.GeoDataFrame(
        reduced_trips[['CO_D_X', 'CO_D_Y']], geometry=gpd.points_from_xy(reduced_trips.CO_D_X, reduced_trips.CO_D_Y))

    geo_trips_destinations.crs = {'init': 'epsg:22523'}
    geo_trips_destinations.to_crs({'proj': 'longlat', 'ellps': 'WGS84', 'no_defs': True}, inplace=True)

    reduced_trips['CO_O_X'] = geo_trips_origins.apply(lambda x: x['geometry'].x, axis=1)
    reduced_trips['CO_O_Y'] = geo_trips_origins.apply(lambda x: x['geometry'].y, axis=1)
    reduced_trips['CO_D_X'] = geo_trips_destinations.apply(lambda x: x['geometry'].x, axis=1)
    reduced_trips['CO_D_Y'] = geo_trips_destinations.apply(lambda x: x['geometry'].y, axis=1)


    reduced_trips['FRACTIONED_MIN_SAIDA']= reduced_trips[['MIN_SAIDA']]/100
    reduced_trips['FRACTIONED_MIN_CHEG']= reduced_trips[['MIN_CHEG']]/100
    reduced_trips['DEPARTURE_TIME']= reduced_trips[['H_SAIDA', 'FRACTIONED_MIN_SAIDA']].sum(axis=1)
    reduced_trips['ARRIVAL_TIME']= reduced_trips[['H_CHEG', 'FRACTIONED_MIN_CHEG']].sum(axis=1)   

    reduced_trips[['ARRIVAL_TIME', 'DEPARTURE_TIME', 'H_SAIDA', 'H_CHEG', 'MIN_SAIDA', 'MIN_CHEG']].describe()


    reduced_trips['FE_VIA_ARRAY'] = reduced_trips.apply(lambda entry: np.arange(entry['FE_VIA']), axis=1)
    expanded_trips = reduced_trips.explode('FE_VIA_ARRAY')

    print("Reduced dataset contains ", len(reduced_trips), " entries.")
    print("Expanded dataset contains: ", len(expanded_trips), " entries")





    cubu_fields = {
        'DEPARTURE_TIME': 'DEPARTURE_TIME',
        'ARRIVAL_TIME': 'ARRIVAL_TIME',
        'LON_O': 'CO_O_X',
        'LAT_O': 'CO_O_Y',
        'LON_D': 'CO_D_X',
        'LAT_D': 'CO_D_Y',
        'MODOPRIN': 'MODOPRIN'
    }

    results_dir = DATASETS_DIR + 'processed'

    import os
    try:
        os.mkdir(results_dir)
    except FileExistsError:
        pass

    file_path = results_dir + '/' + MY_DATASET
    utils.parse_to_cubu(expanded_trips.copy(), file_path, cubu_fields)

areas = [
    'SubCentral-AreaCentral',
    'SubCentral-AreaNoroeste',
    'SubCentral-AreaNorte',
    'SubCentral-AreaNordeste',
    'SubCentral-AreaLeste',
    'SubCentral-AreaSudeste',
    'SubCentral-AreaSul',
    'SubCentral-AreaSudoeste',
    'SubCentral-AreaOeste' 
]

for area in areas:
    process_data(2017, 'data/walks-by-area/', f'{area}.csv')

