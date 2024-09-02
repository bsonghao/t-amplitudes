# system import
import os
import sys
import math
import copy
import itertools as it
import re
import json


# third party import
import numpy as np
from log_conf import log
import pandas as pd

class read_energy(object):
    """This python class readin energy eigenvalues
    from the MCTDH block relax calculation.
    """
    def __init__(self, file_name):
        """
        file_name: name of the MCTDH output file
        """
        self.file_name = file_name
        # Boltzmann constant (eV K-1)
        self.Kb = 8.61733326e-5

    def read_data(self, name, num_of_state=100):
        """
        read energy eigenvalues in MCTDH output
        and store them in csv format

        name: name of the file to be stored
        """
        log.info("Start extracting energy eigenvalue data")
        # read in mctdh output file line by line
        f = open(self.file_name, 'r')
        data = f.readlines()
        f.close()

        # extract the lines that contains converged energy eigenvalues
        energy_data = data[-5-num_of_state : -5]

        # extract energy data from those lines
        energy_data_dic={"Energy(ev)" : []}
        for idx, line in enumerate(energy_data):
            tmp = line.split()
            energy_data_dic["Energy(ev)"].append(float(tmp[5]))
            # print(tmp)

        # store the energy data in to csv
        df = pd.DataFrame(energy_data_dic)
        # print(df.head())
        df.to_csv("{:}_energy_eigenvalues.csv".format(name), index=False)
        log.info("Energy eigenvalue data are extracted and stored successfully!")

        return

    def process_data(self, name, T_grid=np.linspace(1000, 10000, 100000)):
        """
        calculate thermal properties from the MCTDH energy eigenvalues
        name: file name of the energy eigenvalue data
        T_grid: a temperture grid
        """
        log.info("Start calculation thermal properties from the MCTDH data")
        def Cal_partition_function(E, T):
            """ compute partition function """
            # extract GS energy
            # E = E - E[0]
            part = sum(np.exp(-E / (self.Kb * T)))
            return part

        def Cal_thermal_internal_energy(E, T, Z):
            """ compute thermal_internal_energy """
            # extract GS energy
            # E = E - E[0]
            energy = sum(E * np.exp(-E / (self.Kb * T))) / Z
            return energy

        df = pd.read_csv("{:}_energy_eigenvalues.csv".format(name))
        energy = df["Energy(ev)"]

        # calculation the thermal properties and store them in a python dictionary
        thermal_data = {"Temperature": T_grid, "Z": [], "E": []}
        for idx, temp in enumerate(T_grid):
            Z = Cal_partition_function(energy, temp)
            E = Cal_thermal_internal_energy(energy, temp, Z)
            thermal_data["Z"].append(Z)
            thermal_data["E"].append(E)

        # store the thermal properties data into csv format
        df = pd.DataFrame(thermal_data)
        # print(df.head())
        df.to_csv("{:}_mctdh_thermal_data.csv".format(name), index=False)
        log.info("Thermal properties successfully calculated and stored from MCTDH data!")
