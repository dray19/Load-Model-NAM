import configparser
import sys

file_path = str(sys.argv[1])
model =  str(sys.argv[2])
new_test_date  = str(sys.argv[3])


config = configparser.ConfigParser()
config.read(file_path)
config[model]['model_name'] = new_test_date
with open(file_path, 'w') as configfile:
    config.write(configfile)