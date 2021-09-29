import numpy as np
import os
import pickle
import argparse

current_path = os.getcwd() 

parser = argparse.ArgumentParser(description='Compute averages for HQs in Glasma.')
parser.add_argument('-quark', type=str, help="Quark type, charm or beauty.")
parser.add_argument('-fonll', help="FONLL initial pT distribution.", action='store_true')
parser.add_argument('-tau', type=float, help="Final proper time.")
args = parser.parse_args()
quark = args.quark
fonll = args.fonll
tau_s = args.tau

# tau_s = 1.0

if fonll:
    data_tau = pickle.load(open(current_path + '/results/pb+pb_5020gev_su2_fonll_' + quark  + '/ev_1_q_1_n_1.pickle', "rb"))
    tau = np.array(data_tau['xmu'])[:, 0]
    index_tau_s = min(range(len(tau)), key=lambda i: abs(tau[i]-tau_s))

    p = pickle.load(open(current_path + '/results/pb+pb_5020gev_su2_fonll_' + quark  + '/parameters.pickle', "rb"))
    events = p['NEVENTS']

    angles = {}
    pTs_qs = {}
    # pTs_aqs = {}
    i = 0
    folder = 'pb+pb_5020gev_su2_fonll_' + quark
    directory = current_path + '/results/' + folder + '/'

    angles_pT = []
    pTs_q, pTs_aq = [], []

    for ev in range(1, events+1):
        pt, N = p['PTFONLL'], p['NFONLL']
        for ipt in range(len(pt)):
            for ip in range(N[ipt]):
                file_name_q = 'ev_' + str(ev) + '_q_' + str(ipt+1) + '_n_' + str(ip+1) + '.pickle'
                print('Averaging ', file_name_q, '...')
                file_path_q = directory + file_name_q
                data_q = pickle.load(open(file_path_q, "rb"))
                index_q = index_tau_s
                pT_q = [np.array(data_q['pmu'])[index_q, 1], np.array(data_q['pmu'])[index_q, 2]]
                pTs_q.append(np.sqrt(pT_q[0]**2+pT_q[1]**2))

                file_name_aq = 'ev_' + str(ev) + '_aq_' + str(ipt+1) + '_n_' + str(ip+1) + '.pickle'
                print('Averaging ', file_name_aq, '...')
                file_path_aq = directory + file_name_aq
                data_aq = pickle.load(open(file_path_aq, "rb"))
                index_aq = index_tau_s
                pT_aq = [np.array(data_aq['pmu'])[index_aq, 1], np.array(data_aq['pmu'])[index_aq, 2]]
                # pTs_aq.append(np.sqrt(pT_aq[0]**2+pT_aq[1]**2))

                unit_vector_pT_q = pT_q / np.linalg.norm(pT_q)
                unit_vector_pT_aq = pT_aq / np.linalg.norm(pT_aq)
                dot_product = np.dot(unit_vector_pT_q, unit_vector_pT_aq)
                angle = np.arccos(dot_product) * 180 / np.pi
                angles_pT.append(angle)

        angles['fonll'] = angles_pT
        pTs_qs['fonll'] = pTs_q
        # pTs_aqs[str(pT)] = pTs_aq
        i=i+1

    os.chdir(current_path + '/results/')
    filename = 'angles_fonll_' + quark + '_tau_' + str(tau_s) + '.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(angles, handle)
    filename_pTs = 'pT_quarks_fonll_' + quark + '_tau_' + str(tau_s) + '.pickle'
    with open(filename_pTs, 'wb') as handle:
        pickle.dump(pTs_qs, handle)
else:
    pTs = [0.5, 5.0, 10.0]
    events = 20

    data_tau = pickle.load(open(current_path + '/results/pb+pb_5020gev_su2_pT_0.5_' + quark + '/ev_1_q_1_tp_1.pickle', "rb"))
    tau = np.array(data_tau['xmu'])[:, 0]
    # tau_s = 0.4
    index_tau_s = min(range(len(tau)), key=lambda i: abs(tau[i]-tau_s))

    p = pickle.load(open(current_path + '/results/pb+pb_5020gev_su2_pT_0.5_' + quark + '/parameters.pickle', "rb"))
    ntp, nq = p['NTP'], p['NQ']

    angles = {}
    pTs_qs = {}
    # pTs_aqs = {}
    i = 0
    for pT in pTs:
        folder = 'pb+pb_5020gev_su2_pT_' + str(pT) + '_' + quark
        directory = current_path + '/results/' + folder + '/'

        angles_pT = []
        pTs_q, pTs_aq = [], []

        for ev in range(1, events+1):
            for tp in range(1, ntp+1):
                for q in range(1, nq+1):
                    file_name_q = 'ev_' + str(ev) + '_q_' + str(q) + '_tp_' + str(tp) + '.pickle'
                    print('Averaging ', file_name_q, '...')
                    file_path_q = directory + file_name_q
                    data_q = pickle.load(open(file_path_q, "rb"))
                    index_q = index_tau_s
                    pT_q = [np.array(data_q['pmu'])[index_q, 1], np.array(data_q['pmu'])[index_q, 2]]
                    pTs_q.append(np.sqrt(pT_q[0]**2+pT_q[1]**2))

                    file_name_aq = 'ev_' + str(ev) + '_aq_' + str(q) + '_tp_' + str(tp) + '.pickle'
                    print('Averaging ', file_name_aq, '...')
                    file_path_aq = directory + file_name_aq
                    data_aq = pickle.load(open(file_path_aq, "rb"))
                    index_aq = index_tau_s
                    pT_aq = [np.array(data_aq['pmu'])[index_aq, 1], np.array(data_aq['pmu'])[index_aq, 2]]
                    # pTs_aq.append(np.sqrt(pT_aq[0]**2+pT_aq[1]**2))

                    unit_vector_pT_q = pT_q / np.linalg.norm(pT_q)
                    unit_vector_pT_aq = pT_aq / np.linalg.norm(pT_aq)
                    dot_product = np.dot(unit_vector_pT_q, unit_vector_pT_aq)
                    angle = np.arccos(dot_product) * 180 / np.pi
                    angles_pT.append(angle)

        angles[str(pT)] = angles_pT
        pTs_qs[str(pT)] = pTs_q
        # pTs_aqs[str(pT)] = pTs_aq
        i=i+1

    os.chdir(current_path + '/results/')
    filename_angles = 'angles_' + quark + '_tau_' + str(tau_s) + '.pickle'
    with open(filename_angles, 'wb') as handle:
        pickle.dump(angles, handle)
    filename_pTs = 'pT_quarks_' + quark + '_tau_' + str(tau_s) + '.pickle'
    with open(filename_pTs, 'wb') as handle:
        pickle.dump(pTs_qs, handle)