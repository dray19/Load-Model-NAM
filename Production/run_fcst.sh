date_arg1=$1
day_val="${date_arg1}06"

python run_models.py "$day_val" 'mean2'
python run_models.py "$day_val" 'mean3'