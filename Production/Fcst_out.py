import os
import configparser
import pandas as pd

config = configparser.ConfigParser()
config.read('/mnt/trade05_data/load_v5_new_code/Production/config.ini')
out_path = config['paths']['out_path']

class Output:
    
    def __init__(self, dts,model_type, model_name, df):
        self.dts = dts
        self.model_type = model_type
        self.model_name = model_name
        self.df = df
        
    def create_path(self):    
        """
        Create output path for a new file.
        Args:
            sdt (datetime): Datetime object.
        Returns:
            str: The generated file name based on the input datetime.
        """
        file_name = f"ewind_pow.{self.dts}_{self.model_type}_lz.NAM"
        return file_name

    def gather_data(self,data,lz):
        """
        Create dataframe of missing datetimes
        param sdate: str
        return: str/dataframe
        """
        dd = data[data['LZ'] == lz]
        dd = dd.drop('LZ', axis = 1)
        return dd.to_string(index = False)

    def Create(self):
        """
        Write forecast data to CSV and create a summary file.

        Args:
        - data (pandas.DataFrame): DataFrame containing forecast data.

        Returns:
        - final_path (str): Path to the final summary file.

        This method writes the forecast data to a CSV file and creates a summary file with detailed information
        for each LZ. The final path to the summary file is returned.

        """
        new_path = f"{out_path}/{self.model_type}/" + self.dts
        lz = ['L01', 'L04','L06','L27','L35', 'L89']
        os.makedirs(new_path, exist_ok=True)
        new_path = new_path + '/'
        self.df.to_csv(new_path + self.dts + '.csv', index = False)
        file_name = self.create_path()
        final_path = new_path + file_name
        print(new_path + file_name)
        f = open(new_path + file_name, "w")
        f.write("{}1.1 {} \n".format(self.model_type,self.dts))
        f.write("\n")
        f.write("{} Forecast Model Number: {} \n".format(self.model_type,self.model_name))
        f.write("\n")
        for i in lz:
            f.write("{} \n".format(('=='*30)))
            f.write("{} \n".format(i))
            f.write("{} \n".format(('=='*30)))
            f.write("\n")
            f.write(self.gather_data(self.df, i))
            f.write("\n")
            f.write("\n")
        f.close()
