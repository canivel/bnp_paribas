import pandas as pd
import numpy as np
import os

'''*****************************************************************'''
'''***************************predefined****************************'''
'''*****************************************************************'''
# path1 = 'output/param_1'
path2 = '../data/subs'
pathe = 'output/ensemble'
tst_ID = pd.read_csv('data/tst_ID.csv')['ID'].values

def make_submission(proba, filepath, filename, ID=tst_ID):
    df = pd.DataFrame({'ID':ID, 'PredictedProb': proba})
    df.to_csv(os.path.join(filepath, filename), index=False)
'''*****************************************************************'''
'''**************************weak ensemble**************************'''
'''*****************************************************************'''
'''all weak models here'''
'''the best one get about 0.4584 on leaderboard'''
'''but most of the log loss are greater than 0.4610'''
ygbc_1100 = pd.read_csv(os.path.join(path2, 'ygbc_1100.csv')).ix[:, 1]
yrfc_1100 = pd.read_csv(os.path.join(path2, 'yrfc.csv')).ix[:, 1]
yrfr = pd.read_csv(os.path.join(path2, 'yrfr.csv')).ix[:, 1]
ygbr = pd.read_csv(os.path.join(path2, 'ygbr.csv')).ix[:, 1]
yetr = pd.read_csv(os.path.join(path2, 'yetr_3000.csv')).ix[:, 1]
yetcg = pd.read_csv(os.path.join(path2, 'yetc_3000_gini.csv')).ix[:, 1]
yetc = pd.read_csv(os.path.join(path2, 'yetc_3000.csv')).ix[:, 1]


yet_16 = pd.read_csv(os.path.join(path2, 'extra_trees_1_6.csv')).ix[:, 1]
yet_21 = pd.read_csv(os.path.join(path2, 'extratree2_1.csv')).ix[:, 1]
yet_32 = pd.read_csv(os.path.join(path2, 'extra_trees_v3_2.csv')).ix[:, 1]
yet_33 = pd.read_csv(os.path.join(path2, 'extra_trees_v3_3.csv')).ix[:, 1]
yet_331 = pd.read_csv(os.path.join(path2, 'extratree3_3.csv')).ix[:, 1]
yf_xgb = pd.read_csv(os.path.join(path2, 'submission_final_xgb.csv')).ix[:, 1]
yf_xgb_c1 = pd.read_csv(os.path.join(path2, 'submission_xgb_compact_1.csv')).ix[:, 1]
yf_xgb_p = pd.read_csv(os.path.join(path2, 'submission_xgb_poly.csv')).ix[:, 1]
ybnp = pd.read_csv(os.path.join(path2, 'bnp_ensembled.csv')).ix[:, 1]
yetc_3000 = pd.read_csv(os.path.join(path2, 'yetc_3000.csv')).ix[:, 1]
ytop_predicition = pd.read_csv(os.path.join(path2, 'top_prediction.csv')).ix[:, 1]
yet_14 = pd.read_csv(os.path.join(path2, 'extra_trees_1_4.csv')).ix[:, 1]
ytop_predicition = pd.read_csv(os.path.join(path2, 'top_prediction.csv')).ix[:, 1]
yet_14 = pd.read_csv(os.path.join(path2, 'extra_trees_1_4.csv')).ix[:, 1]
yet_15 = pd.read_csv(os.path.join(path2, 'extra_trees_1_5.csv')).ix[:, 1]
yet_45= pd.read_csv(os.path.join(path2, 'extra_trees_045.csv')).ix[:, 1]
yet_17 = pd.read_csv(os.path.join(path2, 'extra_trees_1_7.csv')).ix[:, 1]
yet_18 = pd.read_csv(os.path.join(path2, 'extra_trees_1_8.csv')).ix[:, 1]

#gather weak models
weak_models = pd.DataFrame(index=yet_16.index)
weak_models['ygbc_1100'] = ygbc_1100
weak_models['yrfc_1100'] = yrfc_1100
weak_models['yrfr'] = yrfr
weak_models['ygbr'] = ygbr
weak_models['yetr'] = yetr
weak_models['yetcg'] = yetcg
weak_models['yetc'] = yetc
weak_models['yet_16'] = yet_16
weak_models['yet_21'] = yet_21
weak_models['yet_32'] = yet_32
weak_models['yet_33'] = yet_33
weak_models['yet_331'] = yet_331
weak_models['yf_xgb'] = yf_xgb
weak_models['yf_xgb_c1'] = yf_xgb_c1
weak_models['yf_xgb_p'] = yf_xgb_p
weak_models['ybnp'] = ybnp
weak_models['yetc_3000'] = yetc_3000
weak_models['ytop_predicition'] = ytop_predicition
weak_models['yet_14'] = yet_14
weak_models['yet_15'] = yet_15
weak_models['yet_45'] = yet_45
weak_models['yet_18'] = yet_18
#compute some info
weak_models_info = pd.DataFrame(index=weak_models.index)
weak_models_info['avg'] = weak_models.mean(axis=1)
weak_models_info['mini'] = weak_models.min(axis=1)
weak_models_info['small2mean'] = np.sort(weak_models)[:, :2].mean(axis=1)
weak_models_info['large8mean'] = np.sort(weak_models)[:, -8:].mean(axis=1)

'''if weak_mean>0.95:       mean of the greatest 8 of the 10'''
'''if 0.15<weak_mean<0.5:   mean of the lowest 2 of the 10'''
'''if weak_mean<0.15:       the minimun'''
'''if 0.5<weak_mean<0.95:   weak_mean'''
weak_ensemble = weak_models_info['avg'].copy()
weak_ensemble[weak_ensemble>0.95] = weak_models_info.loc[weak_ensemble>0.95, 'large8mean']
weak_ensemble[weak_ensemble<0.5] = weak_models_info.loc[weak_ensemble<0.5, 'small2mean']
weak_ensemble[weak_models_info['avg']<0.15] = \
    weak_models_info.loc[weak_models_info['avg']<0.15, 'mini']
#to csv
weak_models[weak_models_info.columns] = weak_models_info
weak_models.to_csv(os.path.join(pathe, 'weak_models.csv'))
weak_ensemble.to_csv(os.path.join(pathe, 'weak_ensemble.csv'))


'''*****************************************************************'''
'''*************************strong ensemble*************************'''
'''*****************************************************************'''
'''load strong models'''
'''yetc can reach 0.4528 on leadboard'''
'''yet4 & weak_ensemble can get 0.4550'''


yet_st_2017 = pd.read_csv(os.path.join(path2, 'extra_trees_strongs_2017.csv')).ix[:, 1]
yet_st_3249 = pd.read_csv(os.path.join(path2, 'extra_trees_strongs_3249.csv')).ix[:, 1]
#gather strong models
strong_models = pd.DataFrame(index=yet_st_2017.index)

strong_models['yet_st_3249'] = yet_st_3249
strong_models['yet_st_2017'] = yet_st_2017
strong_models['weak_ensemble'] = weak_ensemble
#compute some info
strong_models_info = pd.DataFrame(index=strong_models.index)
strong_models_info['avg'] = strong_models.mean(axis=1)
strong_models_info['mini'] = strong_models.min(axis=1)
strong_models_info['medi'] = np.sort(strong_models)[:, 1]
strong_models_info['maxi'] = strong_models.max(axis=1)
strong_models_info['0.6min0.4med'] = 0.6*strong_models_info['mini'] + 0.4*strong_models_info['medi']
strong_models_info['0.4med0.6max'] = 0.4*strong_models_info['medi'] + 0.6*strong_models_info['maxi']

'''if strong_mean>0.97:       0.4*median + 0.6*maximun'''
'''if 0.15<strong_mean<0.5:   0.6*minimun + 0.4*median'''
'''if strong_mean<0.15:       the minimun'''
'''if 0.5<strong_mean<0.97:   strong_mean'''
strong_ensemble = strong_models_info['avg'].copy()
strong_ensemble[strong_ensemble>0.97] = strong_models_info.loc[strong_ensemble>0.97, '0.4med0.6max']
strong_ensemble[strong_ensemble<0.5] = strong_models_info.loc[strong_ensemble<0.5, '0.6min0.4med']
strong_ensemble[strong_models_info['avg']<0.15] = \
    strong_models_info.loc[strong_models_info['avg']<0.15, 'mini']
#to csv
strong_models[strong_models_info.columns] = strong_models_info
strong_models.to_csv(os.path.join(pathe, 'strong_models.csv'))
strong_ensemble.to_csv(os.path.join(pathe, 'strong_ensemble.csv'))


'''at last, the strong ensemble get logloss=0.45072 on the leaderboard'''
'''and this score can be improved if squeeze more'''
''''''
make_submission(strong_ensemble, pathe, 'sub_models6.csv')