import json
import pandas
import os

# Number of sample elements to read at a time
CHUNKSIZE=500

# Set True to read all csv elements
READ_ALL_DATA=False

def load_spec(path):
    datastore = None 
    with open(path) as f:
        datastore = json.load(f) 

    return datastore

def get_entry(year):
    return load_spec("../datasets/od" + year + "/od-spec.json")

def save_spec(entry):
    spec_file = entry['base_path'] + 'od-spec.json'
    with open(spec_file, 'w') as f:
        json.dump(entry, f)

    print("Saved to: ", spec_file)
    return entry

def full_print(dataframe, *args):
    pandas.set_option('display.max_columns', None)
    display(dataframe)
    pandas.reset_option('display.max_columns')
    
def create_results_dir(entry, exp):
    dir_path = entry['base_path'] + 'processed/' + exp
    try:
        os.mkdir(dir_path)
    except FileExistsError:
        pass
    
    print("\nSaving to %s" % dir_path)
    
def parse_to_cubu(trips, path, cubu_fields):
    # Generate index and rename colunms that match Origin and Destination
    trips.reset_index(inplace=True, drop=True)
    trips.insert(1, 'IDX', [str(x) + ':' for x in list(trips.index)])

    fields = ['IDX', 'MODOPRIN', 'DEPARTURE_TIME', 'ARRIVAL_TIME', 'LON_O', 'LAT_O', 'LON_D', 'LAT_D']
    for f in fields[1:]:
        if cubu_fields.keys().__contains__(f):
            trips.rename(columns = {cubu_fields[f]: f}, inplace=True)
        else:
            trips[f] = 1
    
    # Save trips_cubu.csv file to be used by Cubu
    trips.to_csv(path,
              sep=' ',
              index=False,
              header=False,
              columns=fields)
    
    print("\nFile written to ", path)
    print("Cubu colunms: ", fields)
    
def pipe(original):
    class PipeInto(object):
        data = {'function': original}

        def __init__(self, *args, **kwargs):
            self.data['args'] = args
            self.data['kwargs'] = kwargs

        def __rrshift__(self, other):
            return self.data['function'](
                other, 
                *self.data['args'], 
                **self.data['kwargs']
            )

    return PipeInto

def log_info(msg):
    print("[I]: ",msg)

def log_error(msg):
    print("[E]: ",msg)
