from read_energy import *
import numpy as np


def main():
    file_name = "rlx_info"
    data = read_energy(file_name)
    data.read_data(name="CoF4")
    data.process_data(name="CoF4", T_grid=np.linspace(10, 1000, 20000))

    return


if __name__ == "__main__":
    main()
