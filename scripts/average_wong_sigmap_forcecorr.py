import numpy as np
import argparse
import pickle
import os

# folder = 'corr_pb+pb_5020gev_su3_pT_0.5_wilson_lines_charm'

parser = argparse.ArgumentParser(description='Compute averages for HQs in Glasma.')
parser.add_argument('-folder', type=str, help="Folder in which results are saved.")
parser.add_argument('-force_corr', help="Compute correlator of Lorentz force.", action='store_true')
args = parser.parse_args()
folder = args.folder
force_corr = args.force_corr

# current_path = os.getcwd() 
# results_folder = 'results'
# results_path = current_path + '/' + results_folder 
results_path = '/home/dana/curraun/results'
wong_path = results_path + '/' + folder + '/'
os.chdir(wong_path)

def average(wong_path):
    average = {}
    x, y = [], []
    px, py = [], []

    taui = 0

    if force_corr:
        tags = ['naive', 'transported']
        all_EformE, all_FformF = {}, {}
        for tag in tags:
            all_EformE[tag] = []
            all_FformF[tag] = []

    for file in os.listdir(wong_path):
        if file.startswith("ev_"):
            print('Averaging file ', file, '...')
            data = pickle.load(open(file, 'rb'))
            x.append(np.array(data['xmu'])[:, 1])
            y.append(np.array(data['xmu'])[:, 2])
            px.append(np.array(data['pmu'])[:, 1])
            py.append(np.array(data['pmu'])[:, 2])

            if taui==0:
                average['tau']=np.array(data['xmu'])[:, 0]
                taui = taui+1

            if force_corr:
                for tag in tags:
                    all_EformE[tag].append(data['correlators']['EformE'][tag])
                    all_FformF[tag].append(data['correlators']['FformF'][tag])


    time_steps = np.array(px).shape[1]

    sigmaxt, sigmapt = [], []

    for i in range(1, time_steps):
        sigmaxt.append(((np.array(x)[:, i]-np.array(x)[:, 0]) ** 2+(np.array(y)[:, i]-np.array(y)[:, 0]) ** 2)/2)
        sigmapt.append(((np.array(px)[:, i]-np.array(px)[:, 0]) ** 2+(np.array(py)[:, i]-np.array(py)[:, 0]) ** 2)/2)

    average['sigmaxt_mean'], average['sigmaxt_std'] = np.mean(sigmaxt, axis=1), np.std(sigmaxt, axis=1)
    average['sigmapt_mean'], average['sigmapt_std'] = np.mean(sigmapt, axis=1), np.std(sigmapt, axis=1)

    if force_corr:
        average['mean_EformE'], average['std_EformE'], average['mean_FformF'], average['std_FformF'] = {}, {}, {}, {}
        for tag in tags:
            average['mean_EformE'][tag], average['std_EformE'][tag] = np.mean(all_EformE[tag], axis=0), np.std(all_EformE[tag], axis=0)
            average['mean_FformF'][tag], average['std_FformF'][tag] = np.mean(all_FformF[tag], axis=0), np.std(all_FformF[tag], axis=0)

    filename = 'avg.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(average, handle)

check_file_avg = os.path.isfile('avg.pickle')
if not check_file_avg:
    average(wong_path)
data = pickle.load(open('avg.pickle', 'rb'))
sigmaxt_mean, sigmaxt_std, sigmapt_mean, sigmapt_std, tau = data['sigmaxt_mean'], data['sigmaxt_std'], data['sigmapt_mean'], data['sigmapt_std'], data['tau']