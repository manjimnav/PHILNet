GPU=0
LOG_FILE=./experiments${GPU}.out
DATASETS=('data/CIF2016o6' 'data/CIF2016o12' 'data/M3' 'data/M4' 'data/Demand' 'data/ElectricityLoad' 'data/ExchangeRate' 'data/NN5' 'data/Tourism' 'data/SolarEnergy' 'data/Traffic-metr-la' 'data/Traffic-perms-bay' 'data/Traffic' 'data/WikiWebTraffic') 
MODELS=('lstm' 'gru' 'philnet')
PARAMETERS=./parameters.json
OUTPUT=./results
CSV_FILENAME=results.csv

python main.py --datasets ${DATASETS[@]} --models ${MODELS[@]} --gpu ${GPU} --parameters  $PARAMETERS --output $OUTPUT --csv_filename $CSV_FILENAME
