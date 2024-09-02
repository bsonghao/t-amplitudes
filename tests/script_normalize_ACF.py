import pandas as pd

file_name = "ACF_low_freq_model_simpler_vibronic_linear_VECC_2_basis_per_mode"
df = pd.read_csv(file_name+".csv")
# normalize the ACF
df["Re"] /= df["Abs"][0]
df["Im"] /= df["Abs"][0]
df["Abs"] /= df["Abs"][0]

with open(file_name+".txt", 'w') as file:
    file.write('#    time[fs]         Re(autocorrel)     Im(autocorrel)     Abs(autocorrel)\n')
    tmp = df.to_records(index=False)
    for t, Re, Im, Abs in tmp:
        x1 = '{:.{}f}'.format(t, 8)
        x2, x3, x4 = ['{:.{}f}'.format(e, 14) for e in (Re, Im, Abs)]
        string = '{:>15} {:>22} {:>18} {:>18}'.format(x1, x2, x3, x4)
        file.write(string+'\n')
