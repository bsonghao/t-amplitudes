from read_energy import *
import numpy as np


def main():
    file_name = "output"
    data = read_energy(file_name)
    data.read_data(name="h2o")
    data.process_data(name="h2o", T_grid=np.linspace(700, 2000, 20000))

    return


if __name__ == "__main__":
    main()
