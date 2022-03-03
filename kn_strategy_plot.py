from __future__ import division
from __future__ import print_function
import matplotlib
matplotlib.use('agg')
#from cbomcode.image.image import *
from cbomcode.tools import fits_cat as fc
import os
import numpy as np
import seaborn as sns
from cbomcode.tools.photo_lib import *
import pandas as pd
from scipy.interpolate import interpn
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import healpy as hp
from mpl_toolkits.mplot3d import Axes3D

"""
Pipeline to ....
"""

def plt_style():
    plt.rcParams.update({
                        'lines.linewidth':1.0,
                        'lines.linestyle':'-',
                        'lines.color':'black',
                        'font.family':'serif',
                        'font.weight':'normal',
                        'font.size':13.0,
                        'text.color':'black',
                        'text.usetex':True,
                        'axes.edgecolor':'black',
                        'axes.linewidth':1.0,
                        'axes.grid':False,
                        'axes.titlesize':'x-large',
                        'axes.labelsize':'x-large',
                        'axes.labelweight':'normal',
                        'axes.labelcolor':'black',
                        'axes.formatter.limits':[-4,4],
                        'xtick.major.size':7,
                        'xtick.minor.size':4,
                        'xtick.major.pad':8,
                        'xtick.minor.pad':8,
                        'xtick.labelsize':'medium',
                        'xtick.minor.width':1.0,
                        'xtick.major.width':1.0,
                        'ytick.major.size':7,
                        'ytick.minor.size':4,
                        'ytick.major.pad':8,
                        'ytick.minor.pad':8,
                        'ytick.labelsize':'medium',
                        'ytick.minor.width':1.0,
                        'ytick.major.width':1.0,
                        'legend.numpoints':1,
                        #'legend.fontsize':'x-large',
                        'legend.shadow':False,
                        'legend.frameon':False})

    return 0
def get_map_distance(map_,savedir=''):

    pb,distmu,distsigma,distnorm = hp.read_map(map_,field=range(4),verbose=False, dtype=[np.float64,np.float64,np.float64,np.float64])#clecio dtype='numpy.float64'
    #pb,distmu,distsigma = hp.read_map(map_,field=range(3),verbose=False, dtype=[np.float64,np.float64,np.float64])#clecio dtype='numpy.float64'
    NSIDE=hp.npix2nside(pb.shape[0])
    pb_check=pb[np.logical_not(np.isinf(distmu))]
    distsigma_check=distsigma[np.logical_not(np.isinf(distmu))]
    distmu_check=distmu[np.logical_not(np.isinf(distmu))]
            
    pb_check=pb_check[np.logical_not(np.isinf(distsigma_check))]
    distmu_check=distmu_check[np.logical_not(np.isinf(distsigma_check))]
    distsigma_check=distsigma_check[np.logical_not(np.isinf(distsigma_check))]


    
    distmu_check_average= np.average(distmu_check,weights=pb_check)
    distsigma_check_average= np.average(distsigma_check,weights=pb_check)

    idx_sort = np.argsort(pb)
    idx_sort_up = list(reversed(idx_sort))
            
    
    resolution=hp.nside2pixarea(NSIDE,degrees=True)
    #sum_ = 0
    id_c = 0
    sum_full=0
    id_full=0
    area_max=sum(pb)
            
    while (sum_full<0.9) and (id_full <len(idx_sort_up)) :
        this_idx = idx_sort_up[id_full]
        sum_full = sum_full+pb[this_idx]
        id_full = id_full+1
        total_area=id_full*resolution
    print("Total event area (deg)="+str(id_full*resolution)+" max area="+str(area_max))


    if savedir!='':
        np.save(savedir,[distmu_check_average,distsigma_check_average])
    return distmu_check_average,distsigma_check_average


def create_heatmap(x,y,nCut,outname,feature_name, x_ticks,y_ticks):

    data = np.transpose([x,y])#(size = (nSamples, 2))
    data = pd.DataFrame(data)
    data['n'] = np.ones(len(x))
    print(data.info())
    cuts = pd.DataFrame({str(feature_name[feature]): pd.cut(data[feature], nCut[feature]) for feature in [0, 1]})
    #print('at first cuts are pandas intervalindex.')
    #print(cuts.head())
    #print(cuts.info())

    print(data.join(cuts).head())

    sums = data.join(cuts).groupby( list(cuts) ).sum()
    sums = sums.unstack(level = 0) # Use level 0 to put 0Bin as columns.

    # Reverse the order of the rows as the heatmap will print from top to bottom.
    sums = sums.iloc[::-1]
    y_ticks=np.flip(y_ticks)
    print('The sum of events')
    print(sums['n'])
    sums['n'] = sums['n'].replace(np.nan, 0)
    print('The sum of events after nan replace')
    print(sum(sum(sums['n'].values)))
    sums['n']=sums['n']/ (sum(sum(sums['n'].values)))
    plt.clf()
    sns.heatmap(sums['n'], linewidths=.5, xticklabels=x_ticks, yticklabels=y_ticks) 
    #plt.title('Means of z vs Features 0 and 1')
    plt.tight_layout()
    plt.savefig(outname)
    return 0

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]


markers20=["*","D",".","v","^","<",">","1","2","3","4","8","s","p","P",",","o"]


for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)


#================ START 
#===================================================================

import glob
#=============== the following paths are required in case you need to build a catalog of the main strategies. If the catalog is ready it will not use the GW maps
map_path='/share/storage1/SLcosmology/darksirens/sims/O4_BNS/gw_sims04/flatten/'
map_path_info='/share/storage1/SLcosmology/knstrategy_runs/O4_BN_SIMS_INFO/'
strategy_dir='/share/storage1/SLcosmology/kn_strategy_paper/sblue_nmo/'
event_list_all=glob.glob(strategy_dir+'*_allconfig.csv')
#===================== end of paths required to build your strategy catalog.

# bins and ticks used for the heatmap of coverage area vs exposure time
area_depth_bins=[np.array([0.65,0.7,0.75,0.8,0.85,0.9])+0.025,[45.0,75.0,105.0,135.0,265.0,335.0,865.0,1535.0,3225.0,3975.0]]
xdvsw=[0.7,0.75,0.8,0.85,0.9]
ydvsw=[60.0,90.0,120.0,200.0,300.0,600.0,1200.0,2400.0,3600.0]
#====== end of bins and ticks definitions for area vs exposure time heatmaps



constrain_time=True#True # in case you want that the observations have at least 1.0 apart.
sufix='sbnmo_time_'# sufix used in the output file names


use_ref_mult=True #false if you dont want the multiple exposure catalog to include the reference strategy, not used if not in multiple exposure mode.

evaluate_strategies=False#False # in case you dont need to build a catalog for the strategies, just plot them
m_exp=False


#========= in case you are running multiple exposure times and want to plot the reference strategy together
sufix_nomulti_strategy='_notmoony_blue__allconfig.csv'
path_nomulti_Strategy='/share/storage1/SLcosmology/kn_strategy_paper/sblue_nmo/'
#========================================================== end of multiple exposure time definitions regarding reference strategy



#=========================== Important: Please change when you need.
strategy_csv_cat=strategy_dir+sufix+'.csv' # strategy output catalog. It will save the caralog here, or in case you are not evaluating the strategies (evaluate_strategies=False) it will try to read the preexisting catalog from here.
#===================================================================

# Reference strategy definition, only used if (evaluate_strategies=True and m_exp=False) or  if m_exp=True with evaluate_strategies=True and use_ref_mult=True
old_strategy_arr=[[90.0,90.0,'zz',0.5,1.5,0.90],[90.0,90.0,'zi',0.5,1.5,0.90],[90.0,90.0,'iz',0.5,1.5,0.90], [90.0,90.0,'ii',0.5,1.5,0.90]]
#{"Detection Probability": probs_all,second_loop_legend: area_all ,"Filter_comb": filters_all,"Exposure01 ": exposure01_all,"Exposure02": exposure02_all,"Observation01": time_delays01_all,"Observation02": time_delays02_all, "Telescope_time01": telescope_time1_all, "Telescope_time02": telescope_time2_all,"Area_deg": areas }


# from now on you just want the plots if probably dont need to change anything from here.

import pandas as pd


fig = plt.figure()
ax = fig.add_subplot(111)




ltt_config=[0.05,0.1]
TT_max=[]
TT_low=[]
obs1=[]
obs2=[]
exp1=[]
exp2=[]
TT=[]
TT1=[]
TT2=[]
probdet1=[]
probdet2=[]
strategy_type=[]
bands=[]
distance_all=[]
distance_all_err=[]
allstrategy_prob=[]
prob_area=[]
prob_area_deg=[]



event_names=[]
probs_low_tt=[]
probs_top_=[]
probs_old=[]
#TT_old=[]

if evaluate_strategies==False:
    event_list_all=[]

for i in range(0,len(event_list_all)):
    map_file_aux=event_list_all[i].split('/')[-1]
    
    map_file=map_file_aux.split('_')[0]+'.fits.gz'
    print (map_file)
    print("this is event "+str(i+1))
    try: 
        distance,distance_err=np.load(map_path_info+map_file_aux.split('_')[0]+'.npy')
    except:
        print("Did not find npy with event info. So, I am Calculating Map distance for "+map_file) 
        distance,distance_err=get_map_distance(map_path+map_file,savedir=map_path_info+map_file_aux.split('_')[0]+'.npy')


    

    all_df=pd.read_csv(event_list_all[i],comment='#')
    total_telescope_time=np.add(all_df["Telescope_time01"].values,all_df["Telescope_time02"].values) # Telescope_time01,Telescope_time02


    if constrain_time==True:
        strategy_time_delays=np.add(-1*np.array(all_df["Observation01"].values),all_df["Observation02"].values) 
        all_df=all_df[np.logical_or(strategy_time_delays > 0.6,strategy_time_delays < 0.4) ]



    if (m_exp==True) and (use_ref_mult==True):
        file_old=map_file_aux.split('_')[0]+sufix_nomulti_strategy
        file_old=path_nomulti_Strategy+file_old
        try:
        
            df_all_old=pd.read_csv(file_old,comment='#')
        except:
            print("Single exposure file not found: ",file_old)
            continue   
        if constrain_time==True:
            strategy_time_delays_nomulti=np.add(-1*np.array(df_all_old["Observation01"].values),df_all_old["Observation02"].values) 
            df_all_old=df_all_old[np.logical_or(strategy_time_delays_nomulti > 0.6,strategy_time_delays_nomulti < 0.4) ]
        #continue
    else:
        df_all_old=all_df







    prob_all=all_df["Detection Probability"].values
    observation1_all=all_df["Observation01"].values
    observation2_all=all_df["Observation02"].values
    exposure1_all=all_df["Exposure01"].values
    exposure2_all=all_df["Exposure02"].values
    region_all=all_df["Region Coverage"].values
    region_all_deg=all_df["Region_coverage_deg"].values
    #Deprob1 , Deprob2
    probdet1_all=all_df["Deprob1"].values#top["Detection Probability"].values
    probdet2_all=all_df["Detprob2"].values
    telescope_time1_all=all_df["Telescope_time01"].values
    telescope_time2_all=all_df["Telescope_time02"].values
    bands_all=all_df["Filter_comb"].values
    #prob_top=prob_top_arr[0]
    prob_top=max(all_df["Detection Probability"].values)
    if prob_top==0:
        print("undetected event for any strategy -- skipping")
        continue


    tt_maxprob=total_telescope_time[np.where(prob_all==max(prob_all))][0]
    obs1_maxprob=observation1_all[np.where(prob_all==max(prob_all))][0]
    obs2_maxprob=observation2_all[np.where(prob_all==max(prob_all))][0]
    exposure1_maxprob=exposure1_all[np.where(prob_all==max(prob_all))][0]
    exposure2_maxprob=exposure2_all[np.where(prob_all==max(prob_all))][0]
    area_maxprob=region_all[np.where(prob_all==max(prob_all))][0]
    area_maxprob_deg=region_all_deg[np.where(prob_all==max(prob_all))][0]
    bands_maxprob=bands_all[np.where(prob_all==max(prob_all))][0]
    probdet1_maxprob=probdet1_all[np.where(prob_all==max(prob_all))][0]
    probdet2_maxprob=probdet2_all[np.where(prob_all==max(prob_all))][0]
    tt1_maxprob=telescope_time1_all[np.where(prob_all==max(prob_all))][0]
    tt2_maxprob=telescope_time1_all[np.where(prob_all==max(prob_all))][0]
    
    allstrategy_prob.append(prob_top)
    
    #TT_max.append(tt_maxprob)
    TT.append(tt_maxprob)
    TT1.append(tt1_maxprob)
    TT2.append(tt2_maxprob)

    obs1.append(obs1_maxprob)
    obs2.append(obs2_maxprob)
    exp1.append(exposure1_maxprob)
    exp2.append(exposure2_maxprob)
    bands.append(bands_maxprob)
    prob_area.append(area_maxprob)
    prob_area_deg.append(area_maxprob_deg)
    distance_all.append(distance)
    distance_all_err.append(distance_err)
    probdet1.append(probdet1_maxprob)
    probdet2.append(probdet2_maxprob)
    strategy_type.append("Top")
    event_names.append(map_file_aux) 

    print("===== top probability")
    print (prob_top)
    #print(prob_top2)
    print(tt_maxprob)
    

    df_ltt=all_df[all_df["Detection Probability"].values > (prob_top-(ltt_config[1]*prob_top))]

    total_telescope_timeltt=np.add(df_ltt["Telescope_time01"].values,df_ltt["Telescope_time02"].values)

    prob_ltt=df_ltt["Detection Probability"].values
    prob_ltt_sel=prob_ltt[np.where(total_telescope_timeltt==min(total_telescope_timeltt))]
    ltt=min(total_telescope_timeltt)

    observation1_ltt=df_ltt["Observation01"].values
    observation2_ltt=df_ltt["Observation02"].values
    exposure1_ltt=df_ltt["Exposure01"].values
    exposure2_ltt=df_ltt["Exposure02"].values
    region_ltt=df_ltt["Region Coverage"].values
    region_ltt_deg=df_ltt["Region_coverage_deg"].values
    filters_ltt=df_ltt["Filter_comb"].values
    probdet1_ltt=df_ltt["Deprob1"].values#top["Detection Probability"].values
    probdet2_ltt=df_ltt["Detprob2"].values
    telescope_time1_ltt=df_ltt["Telescope_time01"].values
    telescope_time2_ltt=df_ltt["Telescope_time02"].values



    obs1_ltt=observation1_ltt[np.where(total_telescope_timeltt==min(total_telescope_timeltt))][0]
    obs2_ltt=observation2_ltt[np.where(total_telescope_timeltt==min(total_telescope_timeltt))][0]
    exp1_ltt=exposure1_ltt[np.where(total_telescope_timeltt==min(total_telescope_timeltt))][0]
    exp2_ltt=exposure2_ltt[np.where(total_telescope_timeltt==min(total_telescope_timeltt))][0]
    bands_ltt=filters_ltt[np.where(total_telescope_timeltt==min(total_telescope_timeltt))][0]
    area_ltt=region_ltt[np.where(total_telescope_timeltt==min(total_telescope_timeltt))][0]
    area_ltt_deg=region_ltt_deg[np.where(total_telescope_timeltt==min(total_telescope_timeltt))][0]

    pdet1_ltt=probdet1_ltt[np.where(total_telescope_timeltt==min(total_telescope_timeltt))][0]
    pdet2_ltt=probdet2_ltt[np.where(total_telescope_timeltt==min(total_telescope_timeltt))][0]
    tt1_ltt=telescope_time1_ltt[np.where(total_telescope_timeltt==min(total_telescope_timeltt))][0]
    tt2_ltt=telescope_time2_ltt[np.where(total_telescope_timeltt==min(total_telescope_timeltt))][0]


    TT.append(ltt)
    TT1.append(tt1_ltt)
    TT2.append(tt2_ltt)
    obs1.append(obs1_ltt)
    obs2.append(obs2_ltt)
    exp1.append(exp1_ltt)
    exp2.append(exp2_ltt)
    bands.append(bands_ltt)
    prob_area.append(area_ltt)
    prob_area_deg.append(area_ltt_deg)
    distance_all.append(distance)
    distance_all_err.append(distance_err)
    strategy_type.append("Telescope Time 10%")
    TT_low.append(ltt)
    allstrategy_prob.append(prob_ltt_sel[0])
    event_names.append(map_file_aux)
    probdet1.append(pdet1_ltt)
    probdet2.append(pdet2_ltt)


    df_ltt=all_df[all_df["Detection Probability"].values > (prob_top-(ltt_config[1]*prob_top))]

    total_time_discovery=df_ltt["Observation02"].values
    total_telescope_timeltt=np.add(df_ltt["Telescope_time01"].values,df_ltt["Telescope_time02"].values)

    prob_ltt=df_ltt["Detection Probability"].values
    prob_ltt_sel_d=prob_ltt[np.where(total_time_discovery==min(total_time_discovery))]


    observation1_ltt=df_ltt["Observation01"].values
    observation2_ltt=df_ltt["Observation02"].values
    exposure1_ltt=df_ltt["Exposure01"].values
    exposure2_ltt=df_ltt["Exposure02"].values
    region_ltt=df_ltt["Region Coverage"].values
    region_ltt_deg=df_ltt["Region_coverage_deg"].values
    filters_ltt=df_ltt["Filter_comb"].values
    probdet1_ltt=df_ltt["Deprob1"].values#top["Detection Probability"].values
    probdet2_ltt=df_ltt["Detprob2"].values
    telescope_time1_ltt=df_ltt["Telescope_time01"].values
    telescope_time2_ltt=df_ltt["Telescope_time02"].values

    obs1_ltt=observation1_ltt[np.where(total_time_discovery==min(total_time_discovery))][0]
    obs2_ltt=observation2_ltt[np.where(total_time_discovery==min(total_time_discovery))][0]
    exp1_ltt=exposure1_ltt[np.where(total_time_discovery==min(total_time_discovery))][0]
    exp2_ltt=exposure2_ltt[np.where(total_time_discovery==min(total_time_discovery))][0]
    bands_ltt=filters_ltt[np.where(total_time_discovery==min(total_time_discovery))][0]
    area_ltt=region_ltt[np.where(total_time_discovery==min(total_time_discovery))][0]
    area_ltt_deg=region_ltt_deg[np.where(total_time_discovery==min(total_time_discovery))][0]
    ltt=total_telescope_timeltt[np.where(total_time_discovery==min(total_time_discovery))][0]

    pdet1_ltt=probdet1_ltt[np.where(total_time_discovery==min(total_time_discovery))][0]
    pdet2_ltt=probdet2_ltt[np.where(total_time_discovery==min(total_time_discovery))][0]
    tt1_ltt=telescope_time1_ltt[np.where(total_time_discovery==min(total_time_discovery))][0]
    tt2_ltt=telescope_time2_ltt[np.where(total_time_discovery==min(total_time_discovery))][0]



    TT.append(ltt)
    TT1.append(tt1_ltt)
    TT2.append(tt2_ltt)
    obs1.append(obs1_ltt)
    obs2.append(obs2_ltt)
    exp1.append(exp1_ltt)
    exp2.append(exp2_ltt)
    bands.append(bands_ltt)
    prob_area.append(area_ltt)
    prob_area_deg.append(area_ltt_deg)
    distance_all.append(distance)
    distance_all_err.append(distance_err)
    strategy_type.append("Early discovery")
    allstrategy_prob.append(prob_ltt_sel_d[0])
    event_names.append(map_file_aux)
    probdet1.append(pdet1_ltt)
    probdet2.append(pdet2_ltt)


    prob_ltt_sel_d=prob_ltt[np.where(total_time_discovery==max(total_time_discovery))]
    obs1_ltt=observation1_ltt[np.where(total_time_discovery==max(total_time_discovery))][0]
    obs2_ltt=observation2_ltt[np.where(total_time_discovery==max(total_time_discovery))][0]
    exp1_ltt=exposure1_ltt[np.where(total_time_discovery==max(total_time_discovery))][0]
    exp2_ltt=exposure2_ltt[np.where(total_time_discovery==max(total_time_discovery))][0]
    bands_ltt=filters_ltt[np.where(total_time_discovery==max(total_time_discovery))][0]
    area_ltt=region_ltt[np.where(total_time_discovery==max(total_time_discovery))][0]
    area_ltt_deg=region_ltt_deg[np.where(total_time_discovery==max(total_time_discovery))][0]
    ltt=total_telescope_timeltt[np.where(total_time_discovery==max(total_time_discovery))][0]


    probdet1_ltt=probdet1_ltt[np.where(total_time_discovery==max(total_time_discovery))][0]
    probdet2_ltt=probdet2_ltt[np.where(total_time_discovery==max(total_time_discovery))][0]
    tt1_ltt=telescope_time1_ltt[np.where(total_time_discovery==max(total_time_discovery))][0]
    tt2_ltt=telescope_time2_ltt[np.where(total_time_discovery==max(total_time_discovery))][0]


    TT.append(ltt)
    TT1.append(tt1_ltt)
    TT2.append(tt2_ltt)
    obs1.append(obs1_ltt)
    obs2.append(obs2_ltt)
    exp1.append(exp1_ltt)
    exp2.append(exp2_ltt)
    bands.append(bands_ltt)
    prob_area.append(area_ltt)
    prob_area_deg.append(area_ltt_deg)
    distance_all.append(distance)
    distance_all_err.append(distance_err)
    strategy_type.append("Late discovery")
    allstrategy_prob.append(prob_ltt_sel_d[0])
    event_names.append(map_file_aux)
    probdet1.append(probdet1_ltt)
    probdet2.append(probdet2_ltt)

    


    prob_ostrategy=-1.0
    for j in range(0,len(old_strategy_arr)):
        old_strategy=old_strategy_arr[j]

         

        df_ostrategy=df_all_old[(df_all_old["Exposure01"].values==old_strategy[0]) & (df_all_old["Exposure02"].values ==old_strategy[1]) & (df_all_old["Filter_comb"].values==old_strategy[2]) & (df_all_old["Observation01"].values ==old_strategy[3]) & (df_all_old["Observation02"].values==old_strategy[4]) & (df_all_old["Region Coverage"].values==old_strategy[5])].copy().reset_index(drop=True)
    #filter_top=np.loadtxt(event_list_top[i],delimiter=',',skiprows=2,usecols=(0,3),unpack=True)
        if df_ostrategy["Detection Probability"].values[0] > prob_ostrategy:
            prob_ostrategy=df_ostrategy["Detection Probability"].values
            #TT_old.append(np.add(df_ostrategy["Telescope_time01"].values,df_ostrategy["Telescope_time02"].values)[0])
            print("Telescope time for old strategy")
            print(df_ostrategy["Telescope_time01"]+df_ostrategy["Telescope_time02"])
            print("====")
            print(np.add(df_ostrategy["Telescope_time01"].values,df_ostrategy["Telescope_time02"].values)[0])
            #print("2 ====") 
            TT_old_sel=np.add(df_ostrategy["Telescope_time01"].values,df_ostrategy["Telescope_time02"].values)[0]
            TT1_old_sel=df_ostrategy["Telescope_time01"].values[0]
            TT2_old_sel=df_ostrategy["Telescope_time02"].values[0]
            obs1_old_sel=old_strategy[3]
            obs2_old_sel=old_strategy[4]
            exp1_old_sel=old_strategy[0]
            exp2_old_sel=old_strategy[1]
            bands_old_sel=old_strategy[2]
            prob_area_old_sel=old_strategy[5]
            prob_area_deg_old_sel=df_ostrategy["Region_coverage_deg"].values[0]
            probdet1_old_sel=df_ostrategy["Deprob1"].values[0]
            probdet2_old_sel=df_ostrategy["Detprob2"].values[0]
    TT.append(TT_old_sel)
    TT1.append(TT1_old_sel)
    TT2.append(TT2_old_sel)
    obs1.append(obs1_old_sel)
    obs2.append(obs2_old_sel)
    exp1.append(exp1_old_sel)
    exp2.append(exp2_old_sel)
    bands.append(bands_old_sel)
    prob_area.append(prob_area_old_sel)
    prob_area_deg.append(prob_area_deg_old_sel)
    allstrategy_prob.append(prob_ostrategy[0])
    distance_all.append(distance) 
    distance_all_err.append(distance_err)
    strategy_type.append("Reference")
    event_names.append(map_file_aux)

    probdet1.append(probdet1_old_sel)
    probdet2.append(probdet2_old_sel)



    color_plot_opt='indianred'
    color_plot_os=tableau20[0]
           
    
    #print(distance)
    probs_top_.append(prob_top)
    probs_low_tt.append(prob_ltt_sel[0])
    probs_old.append(prob_ostrategy[0])
    print('TOP filter and probability   ', prob_top ) 
    print('Old Strategy filter and probability   ',prob_ostrategy[0] )
 


if evaluate_strategies==True:   

    TT=np.array(TT)#/(60.0*60.0)
    


    exp1=np.array(exp1)
    exp2=np.array(exp1)
    obs1=np.array(obs1)
    obs2=np.array(obs2)
    strategy_type=np.array(strategy_type)
    allstrategy_prob=np.array(allstrategy_prob)

    print (TT)
    print (type(TT))
    print(TT.shape)
    print(exp1.shape)
    print(exp2.shape)
    print(obs1.shape)
    print(obs2.shape)
    print(np.array(bands).shape)
    print(np.array(prob_area).shape)
    print(np.array(allstrategy_prob).shape)
  
    print(strategy_type.shape) #'Exposure1': np.array(exp1), 'Exposure2': np.array(exp2)
    #print (TT)
    #print (prob_area_deg)
    
    strategy_dict={'Telescope Time': TT,'Observation1': np.array(obs1), 'Observation2': np.array(obs2), 'Strategy': np.array(strategy_type), 'Detection Prob': allstrategy_prob, 'Integrated Prob Area': np.array(prob_area), 'Filters': np.array(bands),'Exposure1': np.array(exp1), 'Exposure2': np.array(exp2), 'Distance':   np.array(distance_all), 'Coverage_deg': prob_area_deg,'TT/area': np.divide(TT,prob_area_deg), 'Prob01': probdet1  , 'Prob02': probdet2, 'TTime1': TT1, 'TTime2': TT2}

    #if m_exp==True:
    #     strategy_dict['Exposure1_deep']=np.array(exp1_deep)    
    #     strategy_dict['Exposure2_deep']=np.array(exp2_deep)
    #     strategy_dict['Coverage_deg_deep']=prob_area_deg_deep

    strategy_df=pd.DataFrame.from_dict(strategy_dict)
    strategy_df.to_csv(strategy_csv_cat, index=False)  
else:
    print('Loading strategy catalog file')
    strategy_df=pd.read_csv(strategy_csv_cat)

#strategy_df['TT/area']=strategy_df['TT/area'].values*60

#plot_vars=['Detection Prob','Telescope Time','Observation1','Observation2']
#g = sns.PairGrid(strategy_df,  corner=True, hue='Strategy', vars = plot_vars)
#g.map_lower(sns.scatterplot, size=strategy_df['Filters'], style=strategy_df['Filters'])
#g.map_diag(sns.histplot, color=".3")
#g.add_legend(title="", adjust_subtitles=True)

#g.map_upper(sns.kdeplot)

# deep (exp1,exp2) vs area (deg or intprob) vs distance
# early or later obs1 vs ob2 (as function of distance?)
# prob vs distance for moony, not moony, red or blue (one plot per strategy)
# telescope time vs distance
# filters vs distance
# area coverage vs volume coverage 50%, 95%

#sns.pairplot(strategy_df, hue='Strategy',  diag_kind="hist") #corner=True
#plt.savefig('strategy_pair_.png',dpi=200)

#plt.clf()
print('Strategy after pair')
#g = sns.PairGrid(strategy_df,  corner=True, hue='Strategy', vars = ['Exposure1','Exposure2','Distance','TT/area'])
#g.map_lower(sns.scatterplot, size=strategy_df['Filters'], style=strategy_df['Filters'])
#g.map_diag(sns.histplot, color=".3")
#g.add_legend(title="", adjust_subtitles=True)
#plt.savefig('strategy_pair_v2.png',dpi=200)

print('Strategy after second pair')
#plt.clf()
#h=sns.jointplot(data=strategy_df, x='Distance', y="TT/area", hue='Strategy', kind="kde")
#plt.savefig('ALLstrategy_joinkde_ttdist.png',dpi=200)
plt.clf()


#h=sns.histplot(data=strategy_df, x='Exposure1', y='Exposure2', hue="Strategy")
#plt.savefig('ALLstrategy_allexp.png',dpi=200)
#plt.clf()


#h=sns.histplot(data=strategy_df, x='Distance', y='TT/area', hue="Strategy")
#plt.savefig('ALLstrategy_alldisttttarea.png',dpi=200)
#plt.clf()

#h=sns.histplot(data=strategy_df, x='Distance', y='Telescope Time', hue="Strategy")
#plt.savefig('ALLstrategy_alldistttt.png',dpi=200)
#plt.clf()

#h=sns.histplot(data=strategy_df, x='Exposure1', y='Integrated Prob Area', hue="Strategy")
#plt.savefig('ALLstrategy_allexp_intarea.png',dpi=200)
#plt.clf()

#h=sns.histplot(data=strategy_df, x='Exposure1', y='Coverage_deg', hue="Strategy")
#plt.savefig('ALLstrategy_allexp_covarea.png',dpi=200)
#plt.clf()

####### ================ Here the plotting starts! ====================================#######

plt.clf()
if m_exp==False or use_ref_mult==True:
    types_of_strategy_name=["R","ED","TT", "Top", "LD"] #"Telescope Time 5%"
    types_of_strategy=["Reference","Early discovery","Telescope Time 10%", "Top", "Late discovery"] #"Telescope Time 5%"
else:

    types_of_strategy_name=["ED","TT", "Top", "LD"] #"Telescope Time 5%"
    types_of_strategy=["Early discovery","Telescope Time 10%", "Top", "Late discovery"] #"Telescope Time 5%"


fig = plt.figure()
ax = fig.add_subplot(111)

print('Strategy 01')
markers_=['*','D','.']
bin_edges_ref=np.arange(0,340,8)
for i in range(0,len(types_of_strategy)):
    print('Strategy '+str(types_of_strategy[i]))
    strategy_plt_dict=strategy_df[(strategy_df['Strategy'].values==types_of_strategy[i])].copy().reset_index(drop=True)
    print('Strategy after selection')
    strategy_plt_dict=strategy_plt_dict[(strategy_plt_dict['Coverage_deg'].values<100.0)].copy().reset_index(drop=True)
    total_prob=strategy_plt_dict['Detection Prob'].values
    first_prob=strategy_plt_dict['Prob01'].values
    second_prob=strategy_plt_dict['Prob02'].values
    dist=strategy_plt_dict['Distance'].values
    
    bin_center,bin_mean,bin_std=fc.make_bin(dist,total_prob,bin_edges_ref)
    plt.plot(bin_center, bin_mean, lw=2, ls='solid', label=types_of_strategy_name[i], color=tableau20[i])
    plt.fill_between(bin_center, bin_mean+bin_std, bin_mean-bin_std, facecolor=tableau20[i], alpha=0.3)



plt.legend()
plt.yticks(np.arange(0, 100+1, 10))
plt.xlabel(r'Distance (MPC)')
plt.axvline(x=120, ls='dashed', c='r')
plt.text(125, 10, "NSNS", fontsize=10, rotation='vertical', color='r', alpha=0.7)
plt.axvline(x=220, ls='-.', c='r')
plt.text(225, 10, "NSBH", fontsize=10, rotation='vertical', color='r', alpha=0.7)
plt.axvline(x=240, ls='-.', c='gray')
plt.text(245, 20, "GW190814", fontsize=10, rotation='vertical', color='gray', alpha=0.7)
plt.axhline(y=68, ls='-.', alpha=0.3)
plt.axhline(y=50, ls='-.', alpha=0.3)
plt.axhline(y=20, ls='-.', alpha=0.3)
plt.ylabel(r'Detection Probability (2 X)')
plt.savefig('strategy_distance_all_'+sufix+'.png',dpi=200)


for i in range(0,len(types_of_strategy)):
    #print('Strategy '+str(types_of_strategy[i]))
    strategy_plt_dict=strategy_df[(strategy_df['Strategy'].values==types_of_strategy[i])].copy().reset_index(drop=True)
    #print('Strategy after selection')
    strategy_plt_dict=strategy_plt_dict[(strategy_plt_dict['Coverage_deg'].values<100.0)].copy().reset_index(drop=True)
    total_prob_area=strategy_plt_dict['Integrated Prob Area'].values
    depth2=strategy_plt_dict['Exposure2'].values
    depth1=strategy_plt_dict['Exposure1'].values
    #second_prob=strategy_plt_dict['Prob02'].values
    #dist=strategy_plt_dict['Distance'].values
    
    bin_center,bin_mean,bin_std=fc.make_bin(dist,total_prob,bin_edges_ref)
    plt.plot(bin_center, bin_mean, lw=2, ls='solid', label=types_of_strategy_name[i], color=tableau20[i])
    plt.fill_between(bin_center, bin_mean+bin_std, bin_mean-bin_std, facecolor=tableau20[i], alpha=0.3)

    #x='Exposure1', y='Integrated Prob Area'
    create_heatmap(x=total_prob_area,y=depth2,nCut=area_depth_bins,outname=types_of_strategy[i]+sufix+'deepvswide.png',feature_name=['Integrated Prob Area','Exposure2'], x_ticks=xdvsw,y_ticks=ydvsw)
    create_heatmap(x=total_prob_area,y=depth1,nCut=area_depth_bins,outname=types_of_strategy[i]+sufix+'deepvswideexp01.png',feature_name=['Integrated Prob Area','Exposure1'], x_ticks=xdvsw,y_ticks=ydvsw)


#for i in range(0,len(types_of_strategy)):
#    strategy_plt_dict=strategy_df[(strategy_df['Strategy'].values==types_of_strategy[i])].copy().reset_index(drop=True)
#    total_prob=strategy_plt_dict['TT/area'].values
    #first_prob=strategy_plt_dict['TT/area'].values
    #second_prob=strategy_plt_dict['Prob02'].values
    #dist=strategy_plt_dict['Distance'].values

    #bin_center,bin_mean,bin_std=fc.make_bin(dist,total_prob,bin_edges_ref)
    #ax1.plot(bin_center, bin_mean, lw=2, ls='solid', label=types_of_strategy_name[i], color=tableau20[i])
    #ax1.fill_between(bin_center, bin_mean+bin_std, bin_mean-bin_std, facecolor=tableau20[i], alpha=0.3)
plt.clf()
for i in range(0,len(types_of_strategy)):
    strategy_plt_dict=strategy_df[(strategy_df['Strategy'].values==types_of_strategy[i])].copy().reset_index(drop=True)
    strategy_plt_dict=strategy_plt_dict[(strategy_plt_dict['Coverage_deg'].values<100.0)].copy().reset_index(drop=True)
    total_prob=strategy_plt_dict['TT/area'].values
    #first_prob=strategy_plt_dict['Prob01'].values
    #second_prob=strategy_plt_dict['Prob02'].values
    dist=strategy_plt_dict['Distance'].values
    
    bin_center,bin_mean,bin_std=fc.make_bin(dist,total_prob,bin_edges_ref)
    plt.plot(bin_center, bin_mean, lw=2, ls='solid', label=types_of_strategy_name[i], color=tableau20[i])
    plt.fill_between(bin_center, bin_mean+bin_std, bin_mean-bin_std, facecolor=tableau20[i], alpha=0.3)


plt.legend()
#plt.yticks(np.arange(0, 100+1, 10))
plt.xlabel(r'Distance (MPC)')
plt.axvline(x=120, ls='dashed', c='r')
plt.axvline(x=220, ls='-.', c='r')
plt.axvline(x=250, ls='-.', c='r')
#plt.axhline(y=68, ls='-.', alpha=0.3)
#plt.axhline(y=50, ls='-.', alpha=0.3)
#plt.axhline(y=20, ls='-.', alpha=0.3)
plt.ylabel(r'Telescope Time/Area')
plt.savefig('strategy_distance_tt_'+sufix+'.png',dpi=200)
plt.clf()


fig = plt.figure()
ax1 = fig.add_subplot(111)
tt_edges=np.arange(0,2600,200)
shift_bin=30
for i in range(0,len(types_of_strategy)):
    strategy_plt_dict=strategy_df[(strategy_df['Strategy'].values==types_of_strategy[i])].copy().reset_index(drop=True)
    strategy_plt_dict=strategy_plt_dict[(strategy_plt_dict['Coverage_deg'].values<100.0)].copy().reset_index(drop=True)
    strategy_plt_dict=strategy_plt_dict[(strategy_plt_dict['Distance'].values<120.0)].copy().reset_index(drop=True)
    total_prob=strategy_plt_dict['TT/area'].values
    #first_prob=strategy_plt_dict['Prob01'].values
    #second_prob=strategy_plt_dict['Prob02'].values
    dist=strategy_plt_dict['Distance'].values
    
    #bin_center,bin_mean,bin_std=fc.make_bin(dist,total_prob,bin_edges_ref)
    #plt.plot(bin_center, bin_mean, lw=2, ls='solid', label=types_of_strategy_name[i], color=tableau20[i])
    #plt.fill_between(bin_center, bin_mean+bin_std, bin_mean-bin_std, facecolor=tableau20[i], alpha=0.3)

    tt_hist, bin_edges_tt = np.histogram(total_prob,bins=tt_edges,density=False)
    tt_center=bin_edges_tt[:-1]+(float(tt_edges[1]-tt_edges[0])/2.0)

    ax1.step(tt_center+(i*shift_bin),tt_hist,alpha=0.8,where='mid', c=tableau20[i], label=types_of_strategy_name[i]) #hatch='\\'#width=tt_edges[1]-tt_edges[0]  fill=False

plt.xlabel(r'Telescope Time/Area (s/$deg^2$)')
plt.legend()
plt.savefig('tt_hist_lowD'+sufix+'.png')
#plt.close('all')

plt.clf()
fig = plt.figure()
ax1 = fig.add_subplot(111)
tt_edges=np.arange(0,2600,200)
shift_bin=30
for i in range(0,len(types_of_strategy)):
    strategy_plt_dict=strategy_df[(strategy_df['Strategy'].values==types_of_strategy[i])].copy().reset_index(drop=True)
    strategy_plt_dict=strategy_plt_dict[(strategy_plt_dict['Coverage_deg'].values<100.0)].copy().reset_index(drop=True)
    strategy_plt_dict=strategy_plt_dict[(strategy_plt_dict['Distance'].values>120.0)].copy().reset_index(drop=True)
    total_prob=strategy_plt_dict['TT/area'].values
    #first_prob=strategy_plt_dict['Prob01'].values
    #second_prob=strategy_plt_dict['Prob02'].values
    dist=strategy_plt_dict['Distance'].values
    
    #bin_center,bin_mean,bin_std=fc.make_bin(dist,total_prob,bin_edges_ref)
    #plt.plot(bin_center, bin_mean, lw=2, ls='solid', label=types_of_strategy_name[i], color=tableau20[i])
    #plt.fill_between(bin_center, bin_mean+bin_std, bin_mean-bin_std, facecolor=tableau20[i], alpha=0.3)

    tt_hist, bin_edges_tt = np.histogram(total_prob,bins=tt_edges,density=False)
    tt_center=bin_edges_tt[:-1]+(float(tt_edges[1]-tt_edges[0])/2.0)

    ax1.step(tt_center+(i*shift_bin),tt_hist,alpha=0.8,where='mid', c=tableau20[i], label=types_of_strategy_name[i]) #hatch='\\'#width=tt_edges[1]-tt_edges[0] 

plt.xlabel(r'Telescope Time/Area (s/$deg^2$)')
plt.legend()
plt.savefig('tt_hist_highD'+sufix+'.png')




plt.clf()
fig = plt.figure()
ax1 = fig.add_subplot(111)
tt_edges=np.arange(0,2600,200)
shift_bin=30
for i in range(0,len(types_of_strategy)):
    strategy_plt_dict=strategy_df[(strategy_df['Strategy'].values==types_of_strategy[i])].copy().reset_index(drop=True)
    strategy_plt_dict=strategy_plt_dict[(strategy_plt_dict['Coverage_deg'].values<100.0)].copy().reset_index(drop=True)
    #strategy_plt_dict=strategy_plt_dict[(strategy_plt_dict['Distance'].values>120.0)].copy().reset_index(drop=True)
    total_prob=strategy_plt_dict['TT/area'].values
    #first_prob=strategy_plt_dict['Prob01'].values
    #second_prob=strategy_plt_dict['Prob02'].values
    dist=strategy_plt_dict['Distance'].values
    
    #bin_center,bin_mean,bin_std=fc.make_bin(dist,total_prob,bin_edges_ref)
    #plt.plot(bin_center, bin_mean, lw=2, ls='solid', label=types_of_strategy_name[i], color=tableau20[i])
    #plt.fill_between(bin_center, bin_mean+bin_std, bin_mean-bin_std, facecolor=tableau20[i], alpha=0.3)

    tt_hist, bin_edges_tt = np.histogram(total_prob,bins=tt_edges,density=False)
    tt_center=bin_edges_tt[:-1]+(float(tt_edges[1]-tt_edges[0])/2.0)

    ax1.step(tt_center+(i*shift_bin),tt_hist,alpha=0.8,where='mid', c=tableau20[i], label=types_of_strategy_name[i]) #hatch='\\'#width=tt_edges[1]-tt_edges[0] 

plt.xlabel(r'Telescope Time/Area (s/$deg^2$)')
plt.legend()
plt.savefig('tt_hist_allD'+sufix+'.png')





plt.clf()
fig = plt.figure()
ax1 = fig.add_subplot(111)
tt_edges=np.arange(0,12.5,0.5)
shift_bin=0.1
for i in range(0,len(types_of_strategy)):
    strategy_plt_dict=strategy_df[(strategy_df['Strategy'].values==types_of_strategy[i])].copy().reset_index(drop=True)
    strategy_plt_dict=strategy_plt_dict[(strategy_plt_dict['Coverage_deg'].values<100.0)].copy().reset_index(drop=True)
    strategy_plt_dict=strategy_plt_dict[(strategy_plt_dict['Distance'].values>120.0)].copy().reset_index(drop=True)
    total_prob=strategy_plt_dict['Telescope Time'].values
    total_prob=np.divide(strategy_plt_dict['Telescope Time'].values,60*60)
    #first_prob=strategy_plt_dict['Prob01'].values
    #second_prob=strategy_plt_dict['Prob02'].values
    dist=strategy_plt_dict['Distance'].values
    
    #bin_center,bin_mean,bin_std=fc.make_bin(dist,total_prob,bin_edges_ref)
    #plt.plot(bin_center, bin_mean, lw=2, ls='solid', label=types_of_strategy_name[i], color=tableau20[i])
    #plt.fill_between(bin_center, bin_mean+bin_std, bin_mean-bin_std, facecolor=tableau20[i], alpha=0.3)

    tt_hist, bin_edges_tt = np.histogram(total_prob,bins=tt_edges,density=False)
    tt_center=bin_edges_tt[:-1]+(float(tt_edges[1]-tt_edges[0])/2.0)

    ax1.step(tt_center+(i*shift_bin),tt_hist,alpha=0.8,where='mid', c=tableau20[i], label=types_of_strategy_name[i]) #hatch='\\'#width=tt_edges[1]-tt_edges[0] 
plt.xlabel(r'Telescope Time (hours)')
plt.legend()
plt.savefig('tt_full_hist_highD'+sufix+'.png')



plt.clf()
fig = plt.figure()
ax1 = fig.add_subplot(111)
tt_edges=np.arange(0,12.5,0.5)
shift_bin=0.1
for i in range(0,len(types_of_strategy)):
    strategy_plt_dict=strategy_df[(strategy_df['Strategy'].values==types_of_strategy[i])].copy().reset_index(drop=True)
    strategy_plt_dict=strategy_plt_dict[(strategy_plt_dict['Coverage_deg'].values<100.0)].copy().reset_index(drop=True)
    strategy_plt_dict=strategy_plt_dict[(strategy_plt_dict['Distance'].values<120.0)].copy().reset_index(drop=True)
    total_prob=strategy_plt_dict['Telescope Time'].values
    total_prob=np.divide(strategy_plt_dict['Telescope Time'].values,60*60)
    #first_prob=strategy_plt_dict['Prob01'].values
    #second_prob=strategy_plt_dict['Prob02'].values
    dist=strategy_plt_dict['Distance'].values
    
    #bin_center,bin_mean,bin_std=fc.make_bin(dist,total_prob,bin_edges_ref)
    #plt.plot(bin_center, bin_mean, lw=2, ls='solid', label=types_of_strategy_name[i], color=tableau20[i])
    #plt.fill_between(bin_center, bin_mean+bin_std, bin_mean-bin_std, facecolor=tableau20[i], alpha=0.3)

    tt_hist, bin_edges_tt = np.histogram(total_prob,bins=tt_edges,density=False)
    tt_center=bin_edges_tt[:-1]+(float(tt_edges[1]-tt_edges[0])/2.0)

    ax1.step(tt_center+(i*shift_bin),tt_hist,alpha=0.8,where='mid', c=tableau20[i], label=types_of_strategy_name[i]) #hatch='\\'#width=tt_edges[1]-tt_edges[0] 
plt.xlabel(r'Telescope Time (hours)')
plt.legend()
plt.savefig('tt_full_hist_lowD'+sufix+'.png')



plt.clf()
fig = plt.figure()
ax1 = fig.add_subplot(111)
tt_edges=np.arange(0,12.5,0.5)
shift_bin=0.1
for i in range(0,len(types_of_strategy)):
    strategy_plt_dict=strategy_df[(strategy_df['Strategy'].values==types_of_strategy[i])].copy().reset_index(drop=True)
    strategy_plt_dict=strategy_plt_dict[(strategy_plt_dict['Coverage_deg'].values<100.0)].copy().reset_index(drop=True)
    #strategy_plt_dict=strategy_plt_dict[(strategy_plt_dict['Distance'].values<120.0)].copy().reset_index(drop=True)
    total_prob=strategy_plt_dict['Telescope Time'].values
    total_prob=np.divide(strategy_plt_dict['Telescope Time'].values,60*60)
    #first_prob=strategy_plt_dict['Prob01'].values
    #second_prob=strategy_plt_dict['Prob02'].values
    dist=strategy_plt_dict['Distance'].values
    
    #bin_center,bin_mean,bin_std=fc.make_bin(dist,total_prob,bin_edges_ref)
    #plt.plot(bin_center, bin_mean, lw=2, ls='solid', label=types_of_strategy_name[i], color=tableau20[i])
    #plt.fill_between(bin_center, bin_mean+bin_std, bin_mean-bin_std, facecolor=tableau20[i], alpha=0.3)

    tt_hist, bin_edges_tt = np.histogram(total_prob,bins=tt_edges,density=False)
    tt_center=bin_edges_tt[:-1]+(float(tt_edges[1]-tt_edges[0])/2.0)

    ax1.step(tt_center+(i*shift_bin),tt_hist,alpha=0.8,where='mid', c=tableau20[i], label=types_of_strategy_name[i]) #hatch='\\'#width=tt_edges[1]-tt_edges[0] 
plt.xlabel(r'Telescope Time (hours)')
plt.legend()
plt.savefig('tt_full_hist_allD'+sufix+'.png')





plt.clf()
fig = plt.figure()
ax1 = fig.add_subplot(111)
tt_edges=np.arange(0,5.5,0.5)
shift_bin=0.1
for i in range(0,len(types_of_strategy)):
    strategy_plt_dict=strategy_df[(strategy_df['Strategy'].values==types_of_strategy[i])].copy().reset_index(drop=True)
    #strategy_plt_dict=strategy_plt_dict[(strategy_plt_dict['Coverage_deg'].values<100.0)].copy().reset_index(drop=True)
    strategy_plt_dict=strategy_plt_dict[(strategy_plt_dict['Distance'].values<120.0)].copy().reset_index(drop=True)
    total_prob=strategy_plt_dict['Observation2'].values
    #total_prob=np.divide(strategy_plt_dict['Telescope Time'].values,60*60)
    #first_prob=strategy_plt_dict['Prob01'].values
    #second_prob=strategy_plt_dict['Prob02'].values
    dist=strategy_plt_dict['Distance'].values
    
    #bin_center,bin_mean,bin_std=fc.make_bin(dist,total_prob,bin_edges_ref)
    #plt.plot(bin_center, bin_mean, lw=2, ls='solid', label=types_of_strategy_name[i], color=tableau20[i])
    #plt.fill_between(bin_center, bin_mean+bin_std, bin_mean-bin_std, facecolor=tableau20[i], alpha=0.3)

    tt_hist, bin_edges_tt = np.histogram(total_prob,bins=tt_edges,density=False)
    tt_center=bin_edges_tt[:-1]+(float(tt_edges[1]-tt_edges[0])/2.0)

    ax1.step(tt_center+(i*shift_bin),tt_hist,alpha=0.8,where='mid', c=tableau20[i], label=types_of_strategy_name[i]) #hatch='\\'#width=tt_edges[1]-tt_edges[0] 
plt.xlabel(r'Telescope Time (hours)')
plt.legend()
plt.savefig('obs2_full_hist_lowD'+sufix+'.png')



plt.clf()
fig = plt.figure()
ax1 = fig.add_subplot(111)
tt_edges=np.arange(0,4.5,0.5)
shift_bin=0.1
for i in range(0,len(types_of_strategy)):
    strategy_plt_dict=strategy_df[(strategy_df['Strategy'].values==types_of_strategy[i])].copy().reset_index(drop=True)
    strategy_plt_dict=strategy_plt_dict[(strategy_plt_dict['Coverage_deg'].values<100.0)].copy().reset_index(drop=True)
    strategy_plt_dict=strategy_plt_dict[(strategy_plt_dict['Distance'].values>120.0)].copy().reset_index(drop=True)
    total_prob=strategy_plt_dict['Observation2'].values
    #total_prob=np.divide(strategy_plt_dict['Telescope Time'].values,60*60)
    #first_prob=strategy_plt_dict['Prob01'].values
    #second_prob=strategy_plt_dict['Prob02'].values
    dist=strategy_plt_dict['Distance'].values
    
    #bin_center,bin_mean,bin_std=fc.make_bin(dist,total_prob,bin_edges_ref)
    #plt.plot(bin_center, bin_mean, lw=2, ls='solid', label=types_of_strategy_name[i], color=tableau20[i])
    #plt.fill_between(bin_center, bin_mean+bin_std, bin_mean-bin_std, facecolor=tableau20[i], alpha=0.3)

    tt_hist, bin_edges_tt = np.histogram(total_prob,bins=tt_edges,density=False)
    tt_center=bin_edges_tt[:-1]+(float(tt_edges[1]-tt_edges[0])/2.0)

    ax1.step(tt_center+(i*shift_bin),tt_hist,alpha=0.8,where='mid', c=tableau20[i], label=types_of_strategy_name[i]) #hatch='\\'#width=tt_edges[1]-tt_edges[0] 
plt.xlabel(r'Days After Burst')
plt.legend()
plt.savefig('obs2_full_hist_highD'+sufix+'.png')




plt.clf()
fig = plt.figure()
ax1 = fig.add_subplot(111)
tt_edges=np.arange(0,4.5,0.5)
shift_bin=0.1
for i in range(0,len(types_of_strategy)):
    strategy_plt_dict=strategy_df[(strategy_df['Strategy'].values==types_of_strategy[i])].copy().reset_index(drop=True)
    strategy_plt_dict=strategy_plt_dict[(strategy_plt_dict['Coverage_deg'].values<100.0)].copy().reset_index(drop=True)
    #strategy_plt_dict=strategy_plt_dict[(strategy_plt_dict['Distance'].values>120.0)].copy().reset_index(drop=True)
    total_prob=strategy_plt_dict['Observation2'].values
    #total_prob=np.divide(strategy_plt_dict['Telescope Time'].values,60*60)
    #first_prob=strategy_plt_dict['Prob01'].values
    #second_prob=strategy_plt_dict['Prob02'].values
    dist=strategy_plt_dict['Distance'].values
    
    #bin_center,bin_mean,bin_std=fc.make_bin(dist,total_prob,bin_edges_ref)
    #plt.plot(bin_center, bin_mean, lw=2, ls='solid', label=types_of_strategy_name[i], color=tableau20[i])
    #plt.fill_between(bin_center, bin_mean+bin_std, bin_mean-bin_std, facecolor=tableau20[i], alpha=0.3)

    tt_hist, bin_edges_tt = np.histogram(total_prob,bins=tt_edges,density=False)
    tt_center=bin_edges_tt[:-1]+(float(tt_edges[1]-tt_edges[0])/2.0)

    ax1.step(tt_center+(i*shift_bin),tt_hist,alpha=0.8,where='mid', c=tableau20[i], label=types_of_strategy_name[i]) #hatch='\\'#width=tt_edges[1]-tt_edges[0] 
plt.xlabel(r'Days After Burst')
plt.legend()
plt.savefig('obs2_full_hist_allD'+sufix+'.png')











plt.clf()
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = fig.add_subplot(111)
strategy_reducecov=strategy_df[(strategy_df['Coverage_deg'].values<50.0)].copy().reset_index(drop=True)
strategy_reducecov=strategy_reducecov[(strategy_reducecov['Distance'].values<220)].copy().reset_index(drop=True)
#'Coverage_deg'
h=sns.histplot(data=strategy_reducecov, x='TT/area', element="poly", hue='Strategy', bins=np.arange(0,40,1), shrink=.8)
#plt.yticks(np.arange(0, 100+1, 10))
#ax1.set_xlabel(r'Distance (MPC)')
#ax1.axvline(x=120, ls='dashed', c='r')
#ax1.axvline(x=220, ls='-.', c='r')
#ax1.axvline(x=220, ls='-.', c='r')
#ax1.axhline(y=68, ls='-.', alpha=0.3)
#ax1.axhline(y=50, ls='-.', alpha=0.3)
#ax1.axhline(y=20, ls='-.', alpha=0.3)
ax1.set_ylabel(r'Telescope Time/Area')
plt.savefig('strategy_distance_TTA_.png',dpi=200)


 



