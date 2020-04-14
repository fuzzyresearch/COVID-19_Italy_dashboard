# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 09:57:11 2020

@author: ranglani
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from ast import literal_eval
import math
from numpy import inf

def num2str0(x):
    if x > 9:
        return(str(x))
    else:
        return(str(0)+str(x))

def date2str(x):
    return(str(x.year)+num2str0(x.month)+num2str0(x.day))

def date2strWHO(x):
    return(str(x.month)+'/'+str(x.day)+'/'+str(x.year)[-2:])

def str2date(x):
    if "-" in x:    
        return datetime.date(int(x[0:4]),int(x[5:7]), int(x[8:10])) 
    else:
        return datetime.date(int(x[0:4]),int(x[4:6]), int(x[6:8])) 

def doublingtime(x):
    ratio = x.pct_change()+1
    lnratio = np.log(ratio)
    return  np.log(2)/lnratio
    
def merc(Coords):
	Coords = literal_eval(Coords)
	lat = Coords[0]
	lon = Coords[1]
	r_major = 6378137.000
	x = r_major * math.radians(lon)
	scale = x/lon
	y = 180.0/math.pi * math.log(math.tan(math.pi/4.0 + lat * (math.pi/180.0)/2.0)) * scale
	return x,y

mercx = lambda x: merc(x)[0]
mercy = lambda x: merc(x)[1]

def zerolog(x):
    if x == 0:
        return 0
    else:
        return np.log(x)
    
def contain_numbers(inputString):
    return any(char.isdigit() for char in inputString)


def raw2dataframe(url):
    db = pd.read_csv(url, sep = ",", error_bad_lines = False)
    db["data"] = db["data"].map(str2date)
    db["date_string"] = db["data"].map(date2str)
    return db

def raw2startenddate(url):
    db = pd.read_csv(url, sep = ",", error_bad_lines = False)
    db["data"] = db["data"].map(str2date)
    db["date_string"] = db["data"].map(date2str)
    lod_italy = list(db["data"])
    st = lod_italy[0]
    en = lod_italy[-1]
    return lod_italy, st, en


#%% Costruzione del DB nazionale

url_national = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv"
url_province = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-province/dpc-covid19-ita-province.csv"
url_regional = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv"
list_of_date_italy = raw2startenddate(url_national)[0]
start = raw2startenddate(url_national)[1]
end = raw2startenddate(url_national)[2]
dbnat = raw2dataframe(url_national)
dbreg = raw2dataframe(url_regional)
dbpro = raw2dataframe(url_province)

db_national = dbnat.drop(columns = ['note_it', 'note_en'])
db_regional = dbreg.loc[(dbreg.lat != 0) & (dbreg.long != 0)]
db_regional = db_regional.drop(columns = ['note_it', 'note_en'])
db_province = dbpro.loc[(dbpro.lat != 0) & (dbpro.long != 0)]
db_province = db_province.drop(columns = ['note_it', 'note_en'])

col_national = ['data', 'stato', 'ricoverati_con_sintomi', 'terapia_intensiva',
       'totale_ospedalizzati', 'isolamento_domiciliare',
       'totale_positivi', 'variazione_totale_positivi', 'nuovi_positivi',
       'dimessi_guariti', 'deceduti', 'totale_casi', 'tamponi', 'date_string']

col_national_mod = ['date', 'state', 'n_hosp_not_ic', 'n_ic', 'n_tot_hosp', 
                    'n_pos_at_home', 'n_tot_pos', 'n_var_tot_pos', 'n_new_pos',      
                    'n_recovered', 'n_dead', 'n_tot_case', 'n_swab', 'date_string']

col_national_num = ['n_hosp_not_ic', 'n_ic', 'n_tot_hosp', 'n_pos_at_home', 
                    'n_tot_pos', 'n_var_tot_pos', 'n_new_pos', 'n_recovered', 'n_dead', 
                    'n_tot_case', 'n_swab']

col_regional = ['data', 'stato', 'codice_regione', 'denominazione_regione', 'lat',
                'long', 'ricoverati_con_sintomi', 'terapia_intensiva',
                'totale_ospedalizzati', 'isolamento_domiciliare',
                'totale_positivi', 'variazione_totale_positivi', 'nuovi_positivi',
                'dimessi_guariti', 'deceduti', 'totale_casi', 'tamponi', 'date_string']

col_regional_mod = ['date', 'state', 'code_reg', 'name_reg', 'lat', 'long', 
                    'n_hosp_not_ic', 'n_ic', 'n_tot_hosp', 'n_pos_at_home', 
                    'n_tot_pos', 'n_var_tot_pos', 'n_new_pos', 'n_recovered', 'n_dead', 
                    'n_tot_case', 'n_swab', 'date_string']

col_province = ['data', 'stato', 'codice_regione', 'denominazione_regione',
                'codice_provincia', 'denominazione_provincia', 'sigla_provincia', 'lat',
                'long', 'totale_casi', 'date_string']

col_province_mod = ['date', 'state', 'code_reg', 'name_reg',
                'code_prov', 'name_prov', 'acronym_prov', 'lat',
                'long', 'n_tot_case', 'date_string']

X_national = db_national.copy()
X_national.columns = col_national_mod

X_regional = db_regional.copy()
X_regional.columns = col_regional_mod
X_regional["eq_coord"] = "("+X_regional.lat.map(str)+","+X_regional.long.map(str)+")"
X_regional["utm_coord_x"] = X_regional["eq_coord"].map(mercx)
X_regional["utm_coord_y"] = X_regional["eq_coord"].map(mercy)

X_province = db_province.copy()
X_province.columns = col_province_mod
X_province["eq_coord"] = "("+X_province.lat.map(str)+","+X_province.long.map(str)+")"
X_province["utm_coord_x"] = X_province["eq_coord"].map(mercx)
X_province["utm_coord_y"] = X_province["eq_coord"].map(mercy)

#%%

X_national_pct = X_national[col_national_num].pct_change().dropna()
col_national_num_pct = [x+"_pct" for x in col_national_num]
X_national_pct.columns = col_national_num_pct
X_national_pct["date_string"] = np.array(X_national["date_string"])[1:]

X_national_diff = X_national[col_national_num].diff().dropna()
col_national_num_diff = [x+"_diff" for x in col_national_num]
X_national_diff.columns = col_national_num_diff
X_national_diff["date_string"] = np.array(X_national["date_string"])[1:]

pos_over_swab = X_national.n_tot_pos/X_national.n_swab
hosp_over_pos = X_national.n_tot_hosp/X_national.n_tot_pos
rec_over_pos = X_national.n_recovered/X_national.n_tot_pos
ic_over_pos = X_national.n_ic/X_national.n_tot_pos
dead_over_pos = X_national.n_dead/X_national.n_tot_pos

X_national_frac = pd.DataFrame(dict(pos_over_swab = pos_over_swab,
                  hosp_over_pos = hosp_over_pos,
                  rec_over_pos = rec_over_pos,
                  ic_over_pos = ic_over_pos, 
                  dead_over_pos = dead_over_pos))

X_national_frac["date_string"] = np.array(X_national["date_string"])
X_national_2 = pd.merge(X_national, X_national_frac, how = 'left', on = 'date_string')
X_national_3 = pd.merge(X_national_2, X_national_pct, how = 'left', on = 'date_string')
X_national_4 = pd.merge(X_national_3, X_national_diff, how = 'left', on = 'date_string')
#%%
X_regional_last = X_regional.loc[X_regional.date == end,:].reset_index(drop = True)
X_regional_last["n_tot_pos_map"] = (X_regional_last["n_tot_pos"])**0.45
X_regional_last["n_tot_case_map"] = (X_regional_last["n_tot_case"])**0.45
X_regional_last["n_tot_rec_map"] = (X_regional_last["n_recovered"])**0.45
X_regional_last["n_ic_map"] = (X_regional_last["n_ic"])**0.45
X_regional_last["n_dead_map"] = (X_regional_last["n_dead"])**0.45
X_regional_info = X_regional_last.drop(columns = col_national_num)
X_regional_before_last = X_regional.loc[X_regional.date == end-datetime.timedelta(days = 1),:].reset_index(drop = True)
X_regional_var = X_regional_last[col_national_num]-X_regional_before_last[col_national_num]
X_regional_var = pd.concat([X_regional_info, X_regional_var], axis = 1)
X_regional_var.loc[X_regional_var["n_tot_case"] <=0, "n_var_case_map"] = 0.5
X_regional_var.loc[X_regional_var["n_tot_case"] >0, "n_var_case_map"] = (np.abs(X_regional_var.loc[X_regional_var["n_tot_case"] >0,"n_tot_case"]))**0.6
X_regional_var.loc[X_regional_var["n_tot_pos"] <=0, "n_var_pos_map"] = 0.5
X_regional_var.loc[X_regional_var["n_tot_pos"] >0, "n_var_pos_map"] = (np.abs(X_regional_var.loc[X_regional_var["n_tot_pos"] >0,"n_tot_pos"]))**0.6
X_regional_var.loc[X_regional_var["n_recovered"] <=0, "n_var_rec_map"] = 0.5
X_regional_var.loc[X_regional_var["n_recovered"] >0, "n_var_rec_map"] = (np.abs(X_regional_var.loc[X_regional_var["n_recovered"] >0,"n_recovered"]))**0.6
X_regional_var.loc[X_regional_var["n_ic"] <=0, "n_var_ic_map"] = 0.5
X_regional_var.loc[X_regional_var["n_ic"] >0, "n_var_ic_map"] = (np.abs(X_regional_var.loc[X_regional_var["n_ic"] >0,"n_ic"]))**0.6
X_regional_var.loc[X_regional_var["n_dead"] <=0, "n_var_dead_map"] = 0.5
X_regional_var.loc[X_regional_var["n_dead"] >0, "n_var_dead_map"] = (np.abs(X_regional_var.loc[X_regional_var["n_dead"] >0,"n_dead"]))**0.6

X_regional_puglia = X_regional.loc[X_regional.name_reg == 'Puglia',:]
X_regional_puglia_diff = X_regional_puglia[col_national_num].diff().dropna()
col_national_num_diff = [x+"_diff" for x in col_national_num]
X_regional_puglia_diff.columns = col_national_num_diff
X_regional_puglia_diff["date_string"] = np.array(X_regional_puglia["date_string"])[1:]
X_regional_puglia_2 = pd.merge(X_regional_puglia, X_regional_puglia_diff, how = 'left', on = 'date_string')

pos_over_swab_puglia = X_regional_puglia_2.n_tot_pos/X_regional_puglia_2.n_swab
hosp_over_pos_puglia = X_regional_puglia_2.n_tot_hosp/X_regional_puglia_2.n_tot_pos
rec_over_pos_puglia = X_regional_puglia_2.n_recovered/X_regional_puglia_2.n_tot_pos
ic_over_pos_puglia = X_regional_puglia_2.n_ic/X_regional_puglia_2.n_tot_pos
dead_over_pos_puglia = X_regional_puglia_2.n_dead/X_regional_puglia_2.n_tot_pos

X_regional_puglia_2_frac = pd.DataFrame(dict(pos_over_swab = pos_over_swab_puglia,
                  hosp_over_pos = hosp_over_pos_puglia,
                  rec_over_pos = rec_over_pos_puglia,
                  ic_over_pos = ic_over_pos_puglia, 
                  dead_over_pos = dead_over_pos_puglia))

X_regional_puglia_2_frac["date_string"] = np.array(X_national["date_string"])
X_regional_puglia_3 = pd.merge(X_regional_puglia_2, X_regional_puglia_2_frac, how = 'left', on = 'date_string')
X_regional_puglia_3 = X_regional_puglia_3.replace(inf,0)


X_province_last = X_province.loc[X_province.date == end,:].reset_index(drop = True)
X_province_last["n_tot_case_map"] = (X_province_last["n_tot_case"])**0.4
X_province_info = X_province_last.drop(columns = ['n_tot_case'])
X_province_before_last = X_province.loc[X_province.date == end-datetime.timedelta(days = 1),:].reset_index(drop = True)
X_province_var = X_province_last['n_tot_case']-X_province_before_last['n_tot_case']
X_province_var = pd.concat([X_province_info, X_province_var], axis = 1)

pop_prov_bari = 1252557
pop_prov_bat = 390063
pop_prov_brindisi = 392975
pop_prov_foggia = 622532
pop_prov_lecce = 795134
pop_prov_taranto = 573225
X_province_puglia = X_province.loc[X_province.name_reg == 'Puglia',:].reset_index(drop = True)
X_province_puglia_bari = X_province_puglia.loc[X_province_puglia.name_prov == 'Bari',:].reset_index(drop = True)
X_province_puglia_bari['n_tot_case_unit'] = X_province_puglia_bari.n_tot_case/pop_prov_bari
X_province_puglia_bari['n_tot_case_diff'] = X_province_puglia_bari.n_tot_case.diff()
X_province_puglia_bat = X_province_puglia.loc[X_province_puglia.name_prov == 'Barletta-Andria-Trani',:].reset_index(drop = True)
X_province_puglia_bat['n_tot_case_unit'] = X_province_puglia_bat.n_tot_case/pop_prov_bat
X_province_puglia_bat['n_tot_case_diff'] = X_province_puglia_bat.n_tot_case.diff()
X_province_puglia_brindisi = X_province_puglia.loc[X_province_puglia.name_prov == 'Brindisi',:].reset_index(drop = True)
X_province_puglia_brindisi['n_tot_case_unit'] = X_province_puglia_brindisi.n_tot_case/pop_prov_brindisi
X_province_puglia_brindisi['n_tot_case_diff'] = X_province_puglia_brindisi.n_tot_case.diff()
X_province_puglia_foggia = X_province_puglia.loc[X_province_puglia.name_prov == 'Foggia',:].reset_index(drop = True)
X_province_puglia_foggia['n_tot_case_unit'] = X_province_puglia_foggia.n_tot_case/pop_prov_foggia
X_province_puglia_foggia['n_tot_case_diff'] = X_province_puglia_foggia.n_tot_case.diff()
X_province_puglia_lecce = X_province_puglia.loc[X_province_puglia.name_prov == 'Lecce',:].reset_index(drop = True)
X_province_puglia_lecce['n_tot_case_unit'] = X_province_puglia_lecce.n_tot_case/pop_prov_lecce
X_province_puglia_lecce['n_tot_case_diff'] = X_province_puglia_lecce.n_tot_case.diff()
X_province_puglia_taranto = X_province_puglia.loc[X_province_puglia.name_prov == 'Taranto',:].reset_index(drop = True)
X_province_puglia_taranto['n_tot_case_unit'] = X_province_puglia_taranto.n_tot_case/pop_prov_taranto
X_province_puglia_taranto['n_tot_case_diff'] = X_province_puglia_taranto.n_tot_case.diff()

#%%


world_conf = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
world_rec = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"
world_death = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"

dbw = pd.read_csv(world_conf, sep = ',', error_bad_lines = False)
colsw = list(dbw.columns)
time_window = [datetime.datetime.strptime(x, '%m/%d/%y').date() for x in colsw if contain_numbers(x)]
start_world = min(time_window)
end_world = max(time_window)
date_cols = [x for x in colsw if contain_numbers(x)]


db_conf_world = pd.read_csv(world_conf, sep = ',', error_bad_lines = False)
world_last_cols = ['Province/State', 'Country/Region', 'Lat', 'Long', date2strWHO(end_world)]
db_conf_world_last = db_conf_world.reindex(columns = world_last_cols)
db_conf_world_last.columns = ['State', 'Country', 'Lat', 'Long', 'conf']
db_conf_world_last = db_conf_world_last.loc[(db_conf_world_last.Lat != 0) & (db_conf_world_last.Long != 0)]
db_conf_world_last["eq_coord"] = "("+db_conf_world_last.Lat.map(str)+","+db_conf_world_last.Long.map(str)+")"


db_rec_world = pd.read_csv(world_rec, sep = ',', error_bad_lines = False)
db_rec_world_last = db_rec_world.reindex(columns = world_last_cols)
db_rec_world_last.columns = ['State', 'Country', 'Lat', 'Long', 'reco']
db_rec_world_last = db_rec_world_last.loc[(db_rec_world_last.Lat != 0) & (db_rec_world_last.Long != 0)]
db_rec_world_last["eq_coord"] = "("+db_rec_world_last.Lat.map(str)+","+db_rec_world_last.Long.map(str)+")"


db_death_world = pd.read_csv(world_death, sep = ',', error_bad_lines = False)
db_death_world_last = db_death_world.reindex(columns = world_last_cols)
db_death_world_last.columns = ['State', 'Country', 'Lat', 'Long', 'dead']
db_death_world_last = db_death_world_last.loc[(db_death_world_last.Lat != 0) & (db_death_world_last.Long != 0)]
db_death_world_last["eq_coord"] = "("+db_death_world_last.Lat.map(str)+","+db_death_world_last.Long.map(str)+")"


X_world_last = pd.merge(db_conf_world_last, db_rec_world_last, how = "left", on = "eq_coord")
X_world_last = pd.merge(X_world_last, db_death_world_last, how = "left", on = "eq_coord")
X_world_last = X_world_last.reindex(columns = ['State_x', 'Country_x', 'Lat_x', 'Long_x', "eq_coord", 'conf', 'reco', 'dead'])
X_world_last = X_world_last.loc[(X_world_last.Lat_x != 0) & (X_world_last.Long_x != 0)]
X_world_last.columns = ['State', 'Country', 'Lat', 'Long', "eq_coord", 'conf', 'reco', 'dead']
X_world_last['State'] = X_world_last['State'].fillna('-')
X_world_last = X_world_last.fillna(0.0)
#X_world_last["eq_coord"] = "("+X_world_last.Lat.map(str)+","+X_world_last.Long.map(str)+")"
X_world_last["utm_coord_x"] = X_world_last["eq_coord"].map(mercx)
X_world_last["utm_coord_y"] = X_world_last["eq_coord"].map(mercy)

X_world_last["n_conf_map"] = 3*(X_world_last['conf'].map(zerolog))
X_world_last["n_reco_map"] = 3*(X_world_last['reco'].map(zerolog))
X_world_last["n_dead_map"] = 3*(X_world_last['dead'].map(zerolog))
X_world_last["date_string"] = [str(end_world)]*len(X_world_last)

X_world_last_h = X_world_last.groupby(['Country'])[['conf', 'reco', 'dead']].sum().reset_index(drop = False)
X_world_last_hist = X_world_last_h.sort_values(by = 'conf', ascending = True)
X_world_last_hist = X_world_last_hist.reset_index()
X_world_last_hist["date_string"] = [str(end_world)]*len(X_world_last_hist)
X_world_last_hist = X_world_last_hist.iloc[-50:]
N_MAIN_COUNTRY = 10
X_world_main = X_world_last_h.sort_values(by = 'conf', ascending = False).reset_index(drop = True)
X_world_main_list = list(X_world_main.iloc[0:N_MAIN_COUNTRY]["Country"])

db_conf_world_country = db_conf_world.groupby(['Country/Region']).sum().reset_index()
X_conf_world_time_series = db_conf_world_country.loc[db_conf_world_country["Country/Region"].isin(X_world_main_list),:]
X_conf_world_time_series = X_conf_world_time_series.drop(columns = ['Lat', 'Long'])
X_conf_world_time_series = X_conf_world_time_series.transpose().reset_index()
X_conf_world_time_series = X_conf_world_time_series.rename(columns=X_conf_world_time_series.iloc[0]).drop(index = 0)
X_conf_world_time_series = X_conf_world_time_series.rename(columns={'Country/Region': 'Date'})
X_conf_world_time_series.Date = time_window
X_conf_world_time_series["date_string"] = X_conf_world_time_series.Date.map(date2str)

db_death_world_country = db_death_world.groupby(['Country/Region']).sum().reset_index()
X_death_world_time_series = db_death_world_country.loc[db_death_world_country["Country/Region"].isin(X_world_main_list),:]
X_death_world_time_series = X_death_world_time_series.drop(columns = ['Lat', 'Long'])
X_death_world_time_series = X_death_world_time_series.transpose().reset_index()
X_death_world_time_series = X_death_world_time_series.rename(columns=X_death_world_time_series.iloc[0]).drop(index = 0)
X_death_world_time_series = X_death_world_time_series.rename(columns={'Country/Region': 'Date'})
X_death_world_time_series.Date = time_window
X_death_world_time_series["date_string"] = X_death_world_time_series.Date.map(date2str)

X_conf_world_doubling = X_conf_world_time_series[X_world_main_list].apply(doublingtime)
ROLL_WINDOW = 1
X_conf_world_doubling = X_conf_world_doubling.rolling(ROLL_WINDOW).sum()/ROLL_WINDOW
X_conf_world_doubling["Date"] = time_window
X_conf_world_doubling["date_string"] = X_conf_world_doubling.Date.map(date2str)

#%%
from scipy.optimize import Bounds, minimize, curve_fit
from scipy.integrate import odeint


def si(y, future_time_window):
    yn = (y-y.min())/(y.max()-y.min())
    x = np.array(range(len(y)))
    def f(x, a, b, c, d):
        return a / (1. + np.exp(-c * (x - d))) + b
    (aest, best, cest, dest) = curve_fit(f, x, yn)[0]
    yest = f(x, aest, best, cest, dest)
    Iest = yest*(y.max()-y.min())+y.min()
    xnew = np.array(range(future_time_window))
    ynew = f(xnew, aest, best, cest, dest)
    Inew = ynew*(y.max()-y.min())+y.min()
    return Iest, Inew


ROLLING_WINDOW_SI = 25
FUTURE_TIME_WINDOW = 40

remaining_list_of_date = []
for i in range(1, FUTURE_TIME_WINDOW):
    remaining_list_of_date.append(list_of_date_italy[-1] + datetime.timedelta(days=i))

all_dates = sum([list_of_date_italy, remaining_list_of_date], [])

yori = X_national.loc[:, "n_tot_pos"]

si_result = pd.DataFrame(data = all_dates, columns = ["Date"])
si_result["Date"] = all_dates
for j in range(0,len(list_of_date_italy)-ROLLING_WINDOW_SI+1):
    sel_date = list_of_date_italy[j:j+ROLLING_WINDOW_SI]
    end_date = sel_date[-1]
    for i in range(1, FUTURE_TIME_WINDOW+1):
        sel_date = np.append(sel_date, end_date + datetime.timedelta(days=i))
    sel_yori = yori.iloc[j:j+ROLLING_WINDOW_SI]
    yforec = si(sel_yori, ROLLING_WINDOW_SI+FUTURE_TIME_WINDOW)[1]
    sel_df = pd.DataFrame(data = yforec, index = sel_date, columns = [str(sel_date[0])])
    sel_df["Date"] = sel_date
    si_result = pd.merge(si_result, sel_df, how = "left", on = "Date")
si_scenario_columns = si_result.drop(columns = ["Date"]).columns
si_result.loc[:, "date_string"] = si_result.Date.map(date2str)

si_result_diff = si_result.copy()
si_result_diff = si_result_diff.reindex(columns = si_scenario_columns)
si_result_diff = si_result_diff.diff()
si_result_diff["Date"] = si_result.Date
si_result_diff.loc[:, "date_string"] = si_result_diff.Date.map(date2str)



TIME_WINDOW_SI = 60
yori = X_national.loc[:, "n_tot_pos"]
SCENARIO_RANGE = np.array(range(len(yori)-10,len(yori)))
newdate = []
for i in range(TIME_WINDOW_SI):
    newdate.append(start + datetime.timedelta(days=i))
newdate = np.array(newdate)

ynew = []
spike_x_list = []
spike_y_list = []
  
for m in SCENARIO_RANGE:
    i2 = si(yori.iloc[0:m], TIME_WINDOW_SI)[1]
    pos_spike = np.where(np.diff(np.sign(np.gradient(np.gradient(i2)))))
    spike_x_list.append(pos_spike[0][0])
    spike_y_list.append(round(i2[pos_spike][0],0))
spikedf = pd.DataFrame({'Date': newdate[spike_x_list], 'x_spike': spike_x_list, 'y_spike': spike_y_list})
spikedf.loc[:, "date_string"] = spikedf.Date.map(date2str)
spikedf["Scenario"] = SCENARIO_RANGE

##############################################################################

TIME_WINDOW_SIR = 54    

Iobs = X_national.loc[:, "n_tot_pos"]
Robs = X_national.loc[:, "n_recovered"] + X_national.loc[:, "n_dead"]
Iobs = Iobs-Robs
min_scenario = Iobs.iloc[-1]*1.01
max_scenario = Iobs.iloc[-1]*2
list_of_scenarios0 = np.arange(min_scenario, max_scenario, (max_scenario-min_scenario)/15)
list_of_scenarios0 = np.array(list(map(int, list_of_scenarios0)))
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I/N
    dIdt = beta * S * I/N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

Ilist = []
for n in list_of_scenarios0:
    I0, R0 = Iobs.iloc[0], Robs.iloc[0]
    S0 = n - I0 - R0
    t = range(len(Iobs))
    bound_values = Bounds((0,0), (1,1))
    def distC(x):
        y0 = S0, I0, R0
        ret = odeint(deriv, y0, t, args=(n, x[0], x[1]))
        S, I, R = ret.T
        return np.sqrt(np.mean((np.array(Iobs)-I)**2))
    algores = minimize(distC, x0 = (0, 0), method='L-BFGS-B', options={'ftol': 1e-9, 'disp': False},  bounds=bound_values)
    I0, R0 = Iobs[0], Robs[0]
    S0 = n - I0 - R0
    y0 = S0, I0, R0
    ret = odeint(deriv, y0, range(TIME_WINDOW_SIR), args=(n, algores.x[0], algores.x[1]))
    S, I, R = ret.T
    Ilist.append(I)
    
newdate_SIR = []
for i in range(TIME_WINDOW_SIR):
    newdate_SIR.append(start + datetime.timedelta(days=i))
newdate_SIR = np.array(newdate_SIR)

sirdf0 = pd.DataFrame(data = np.transpose(Ilist), columns = map(str, list_of_scenarios0))
sirdf0["Date"] = newdate_SIR
sirdf0.loc[:, "date_string"] = sirdf0.Date.map(date2str)
Iox = pd.DataFrame(data = np.array(Iobs), columns = ["n_tot_pos"])
sirdf0 = pd.concat([sirdf0, Iox], axis = 1)

min_scenario = Iobs.iloc[-1]*1.01
max_scenario = 3*10**7
list_of_scenarios1 = np.arange(np.max(list_of_scenarios0), max_scenario, (max_scenario-min_scenario)/10)
list_of_scenarios1 = np.array(list(map(int, list_of_scenarios1)))
Ilist = []
for n in list_of_scenarios1:
    I0, R0 = Iobs.iloc[0], Robs.iloc[0]
    S0 = n - I0 - R0
    t = range(len(Iobs))
    bound_values = Bounds((0,0), (1,1))
    def distC(x):
        y0 = S0, I0, R0
        ret = odeint(deriv, y0, t, args=(n, x[0], x[1]))
        S, I, R = ret.T
        return np.sqrt(np.mean((np.array(Iobs)-I)**2))
    algores = minimize(distC, x0 = (0, 0), method='L-BFGS-B', options={'ftol': 1e-9, 'disp': False},  bounds=bound_values)
    I0, R0 = Iobs[0], Robs[0]
    S0 = n - I0 - R0
    y0 = S0, I0, R0
    ret = odeint(deriv, y0, range(TIME_WINDOW_SIR), args=(n, algores.x[0], algores.x[1]))
    S, I, R = ret.T
    Ilist.append(I)
    
newdate_SIR = []
for i in range(TIME_WINDOW_SIR):
    newdate_SIR.append(start + datetime.timedelta(days=i))
newdate_SIR = np.array(newdate_SIR)

sirdf1 = pd.DataFrame(data = np.transpose(Ilist), columns = map(str, list_of_scenarios1))
sirdf1["Date"] = newdate_SIR
sirdf1.loc[:, "date_string"] = sirdf0.Date.map(date2str)
Iox = pd.DataFrame(data = np.array(Iobs), columns = ["n_tot_pos"])
sirdf1 = pd.concat([sirdf1, Iox], axis = 1)
    
#%% REPORT BOKEH

import bokeh.models as bkh_mod
import bokeh.models.widgets as bkh_mod_w
import bokeh.palettes as bkh_pal
import bokeh.io as bkh_io
import bokeh.plotting as bkh_plt
from bokeh.io import curdoc, show
from bokeh.plotting import reset_output
from bokeh.tile_providers import get_provider, Vendors
from bokeh.transform import linear_cmap

### OUTPUT BOKEH
   
reset_output() ### Importante per evitare il messaggio d'errore
               ### "Models must be owned by only a single document" 
"""
Importante! Riaggiornare le sorgenti Bokeh ad ogni aggiornamento del grafico
altrimenti restituisce errore
"""

TOOLS_NEW = "crosshair, pan, wheel_zoom, zoom_in, zoom_out, box_zoom, undo, \
             redo, reset, tap, save, box_select, poly_select, lasso_select,"

abs_tooltips = [('Date', '@date_string'), ('Swabs', '@n_swab'), ('Cases', '@n_tot_case'), 
                    ('Positives', '@n_tot_pos'), ('IC', '@n_ic'),
                    ('Recovered', '@n_recovered'), ('Deaths', '@n_dead')]
diff_tooltips = [('Date', '@date_string'), ('Var. Swabs', '@n_swab_diff'), ('Var. Cases', '@n_tot_case_diff'), 
                    ('Var. Infected', '@n_tot_pos_diff'), ('Var. IC', '@n_ic_diff'),
                    ('Var. Recovered', '@n_recovered_diff'), ('Var. Deaths', '@n_dead_diff')]
pct_tooltips = [('Date', '@date_string'), ('Var.% Swabs', '@n_swab_pct{%0.2%}'), ('Var.% Case', '@n_tot_case_pct{%0.2%}'), 
                    ('Var.% Infected', '@n_tot_pos_pct{%0.2%}'), ('Var.% IC', '@n_ic_pct{%0.2%}'),
                    ('Var. Recovered', '@n_recovered_pct{%0.2%}'), ('Var.% Deaths', '@n_dead_pct{%0.2%}')]
frac_tooltips = [('Date', '@date_string'), ('Infected/Swabs', '@pos_over_swab{%0.2%}'), 
                    ('Hosp/Infected', '@hosp_over_pos{%0.2%}'), ('Recovered/Infected', '@rec_over_pos{%0.2%}'),
                    ('IC/Infected', '@ic_over_pos{%0.2%}'), ('Deaths/Infected', '@dead_over_pos{%0.2%}')]
reg_tooltips = [('Date', '@date_string'), ('Region', '@name_reg'), 
                ('Swabs', '@n_swab'),
                    ('Cases', '@n_tot_case'), 
                    ('Positives', '@n_tot_pos'), ('IC', '@n_ic'),
                    ('Recovered', '@n_recovered'), ('Deaths', '@n_dead')]
reg_var_tooltips = [('Date', '@date_string'), ('Regione', '@name_reg'), 
                    ('Var. Swabs', '@n_swab'),
                    ('Var. Cases', '@n_tot_case'), 
                    ('Var. Infected', '@n_tot_pos'), ('Var. IC', '@n_ic'),
                    ('Var. Recovered', '@n_recovered'), ('Var. Deaths', '@n_dead')]
prov_tooltips = [('Date', '@date_string'), ('Province', '@name_prov'), 
                    ('Cases', '@n_tot_case')]
puglia_tooltips = [('Date', '@date_string'), ('Province', '@name_prov'), 
                    ('Cases', '@n_tot_case'), ('New Positives:', '@n_tot_case_diff'),
                    ('Fraction per population:', '@n_tot_case_unit{%0.3%}')]

prov_var_tooltips = [('Date', '@date_string'), ('Provincia', '@name_prov'), 
                    ('Var. Cases', '@n_tot_case')]

prov_var_tooltips = [('Date', '@date_string'), ('Province', '@name_prov'), 
                    ('Var. Cases', '@n_tot_case')]
world_map_tooltips = [('Date', '@date_string'), ('Country', '@Country'),
                     ('State', '@State'),
                     ('Confirmed', '@conf'),
                     ('Recovered', '@reco'),
                     ('Deaths', '@dead')]
world_hist_tooltips = [('Date', '@date_string'), ('Country', '@Country'),
                     ('Confirmed', '@conf'),
                     ('Recovered', '@reco'),
                     ('Deaths', '@dead')]
world_ts_tooltips = [('Date', '@Date'),
                     ('Confirmed', '@y')]




### Aggiornamento sorgenti

source_db = bkh_mod.ColumnDataSource(data = X_national_4)
source_db_pct = bkh_mod.ColumnDataSource(data = X_national_pct)
source_db_frac = bkh_mod.ColumnDataSource(data = X_national_frac)
source_db_reg = bkh_mod.ColumnDataSource(data = X_regional_last)
source_db_reg_var = bkh_mod.ColumnDataSource(data = X_regional_var)
source_db_reg_puglia = bkh_mod.ColumnDataSource(data = X_regional_puglia_3)
source_db_prov = bkh_mod.ColumnDataSource(data = X_province_last)
source_db_prov_var = bkh_mod.ColumnDataSource(data = X_province_var)
source_db_prov_bari = bkh_mod.ColumnDataSource(data = X_province_puglia_bari)
source_db_prov_bat = bkh_mod.ColumnDataSource(data = X_province_puglia_bat)
source_db_prov_brindisi = bkh_mod.ColumnDataSource(data = X_province_puglia_brindisi)
source_db_prov_foggia = bkh_mod.ColumnDataSource(data = X_province_puglia_foggia)
source_db_prov_lecce = bkh_mod.ColumnDataSource(data = X_province_puglia_lecce)
source_db_prov_taranto = bkh_mod.ColumnDataSource(data = X_province_puglia_taranto)

source_db_world = bkh_mod.ColumnDataSource(data = X_world_last)
source_db_world_hist = bkh_mod.ColumnDataSource(data = X_world_last_hist)
source_db_world_conf_ts = bkh_mod.ColumnDataSource(data = X_conf_world_time_series)
source_db_world_death_ts = bkh_mod.ColumnDataSource(data = X_death_world_time_series)
source_db_world_doubling = bkh_mod.ColumnDataSource(data = X_conf_world_doubling)

source_si = bkh_mod.ColumnDataSource(data = si_result)
source_si_spike = bkh_mod.ColumnDataSource(data = spikedf)
source_si_diff = bkh_mod.ColumnDataSource(data = si_result_diff)
source_sir0 = bkh_mod.ColumnDataSource(data = sirdf0)
source_sir1 = bkh_mod.ColumnDataSource(data = sirdf1)


progr = str(np.random.rand(1))[3:7]
#bkh_io.output_file("t"+progr+".html", title='FuzzyResearch-Covid19')
bkh_io.output_file("index.html", title='FuzzyResearch-Covid19')

##############################################################################
### TAB. 1  - Statistiche generali su azienda
##############################################################################
cover_1 = bkh_mod_w.Div(text =
"""
<font size="5"> Situation as of <b> %s</b></font><br>
<font size="5"> <b> %s</b></font> Cases&nbsp;
<font size="4"> <b> %s</b></font> Rec&nbsp;
<font size="4"> <b> %s</b></font> Pos&nbsp;
<font size="4"> <b> %s</b></font> Deaths
"""% (X_national.date.iloc[-1],
'{:,}'.format(X_national.n_tot_case.iloc[-1]),
'{:,}'.format(X_national.n_recovered.iloc[-1]),
'{:,}'.format(X_national.n_tot_pos.iloc[-1]),
'{:,}'.format(X_national.n_dead.iloc[-1])), #width=250, height=75
)
    
Sign = bkh_mod_w.Div(text =
"""
Licensed under Creative Commons <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons Licence" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/80x15.png" /></a><br>
<span xmlns:dct="http://purl.org/dc/terms/" property="dct:title"> Developed </span> 
by <a xmlns:cc="http://creativecommons.org/ns#" href="https://www.linkedin.com/in/anglani/" property="cc:attributionName" rel="cc:attributionURL"> Roberto Anglani </a> and
<a xmlns:cc="http://creativecommons.org/ns#" href="https://www.linkedin.com/in/stefanonicotri/" property="cc:attributionName" rel="cc:attributionURL"> Stefano Nicotri </a> <br>
""", 
#width=1500, height=75
)
Sign.sizing_mode = 'scale_width'

LegClick = bkh_mod_w.Div(text =
"""
<font color = "red"> <i><b> Click on legend items to activate/remove curves </b></i></font> <br>
<font color = "red"> <i> Hover on points to see further details </i></font>
""", 
#width=1500, height=75
)
LegClick.sizing_mode = 'scale_width'

LegMeaning = bkh_mod_w.Div(text =
"""
Cases = Positives + Recovered + Deaths <br> IC = Intensive Care <br>
Hospitalized = Hospitalized patients without IC and with IC
""", 
#width=1500, height=75
)
LegMeaning.sizing_mode = 'scale_width'
    
    

p01 = bkh_plt.figure(tools = TOOLS_NEW, width=500, #height=350,
                    title="Trend of cumulative values for Italy",
                    #x_axis_label='x', #y_axis_label='y',
                    x_axis_type='datetime', y_axis_type="log")
p01.line(x = 'date', y = 'n_swab', source = source_db, legend_label="Swabs",
           color="blue", line_color = 'blue', alpha = 1, line_width = 2)
p01.line(x = 'date', y = 'n_tot_case', source = source_db, legend_label="Cases",
          color="magenta", line_color = 'magenta', alpha = 1, line_width = 2)
p01.line(x = 'date', y = 'n_tot_pos', source = source_db, legend_label="Positives",
          color="red", line_color = 'red', alpha = 1, line_width = 2)
p01.line(x = 'date', y = 'n_tot_hosp', source = source_db, legend_label="Hospitalized",
          color="orange", line_color = 'orange', alpha = 1, line_width = 2)
p01.line(x = 'date', y = 'n_recovered', source = source_db, legend_label="Recovered",
          color="green", line_color = 'green', alpha = 1, line_width = 2)
p01.line(x = 'date', y = 'n_ic', source = source_db, legend_label="IC",
          color="purple", line_color = 'purple', alpha = 1, line_width = 2)
p01.line(x = 'date', y = 'n_dead', source = source_db, legend_label="Deaths",
          color="black", line_color = 'black', alpha = 1, line_width = 2)
p01.circle(x = 'date', y = 'n_swab', source = source_db, legend_label="Swabs",
          color="blue", size = 10, fill_alpha = 0.5, line_color = 'blue')
p01.circle(x = 'date', y = 'n_tot_case', source = source_db, legend_label="Cases",
          color="magenta", size = 10, line_color = 'magenta', alpha = 0.5, line_width = 2)
p01.circle(x = 'date', y = 'n_tot_pos', source = source_db, legend_label="Positives",
          color="red", size = 10, line_color = 'red', alpha = 0.5, line_width = 2)
p01.circle(x = 'date', y = 'n_tot_hosp', source = source_db, legend_label="Hospitalized",
          color="orange", size = 10, line_color = 'orange', alpha = 0.5, line_width = 2)
p01.circle(x = 'date', y = 'n_recovered', source = source_db, legend_label="Recovered",
          color="green", size = 10, line_color = 'green', alpha = 0.5, line_width = 2)
p01.circle(x = 'date', y = 'n_ic', source = source_db, legend_label="IC",
          color="purple", size = 10, line_color = 'purple', alpha = 0.5, line_width = 2)
p01.circle(x = 'date', y = 'n_dead', source = source_db, legend_label="Deaths",
          color="black", size = 10, line_color = 'black', alpha = 0.5, line_width = 2)


p01.background_fill_color ="gainsboro"
p01.legend.label_text_font_size = "9pt"
p01.sizing_mode = 'scale_width'
p01.legend.location = "top_left"
p01.legend.background_fill_alpha = 0.0
p01.legend.click_policy="hide"
p01.yaxis[0].formatter = bkh_mod.NumeralTickFormatter(format="0,000")
p01.add_tools(bkh_mod.HoverTool(tooltips = abs_tooltips))


p02 = bkh_plt.figure(tools = TOOLS_NEW, width=500, #height=350,
                    title="Trend of proportions for Italy",
                    #x_axis_label='x', #y_axis_label='y',
                    x_axis_type='datetime')
p02.circle(x = 'date', y = 'pos_over_swab', source = source_db, legend_label="Positive/Swabs",
          color="red", size = 10, fill_alpha = 0.5, line_color = 'red', line_width = 2)
p02.line(x = 'date', y = 'pos_over_swab', source = source_db, legend_label="Positive/Swabs",
          color="red", line_color = 'red', line_width = 2)
p02.circle(x = 'date', y = 'hosp_over_pos', source = source_db, legend_label="Hospit/Positives",
          color="orange", size = 10, fill_alpha = 0.5, line_color = 'orange', line_width = 2)
p02.line(x = 'date', y = 'hosp_over_pos', source = source_db, legend_label="Hospit/Positives",
          color="orange", line_color = 'orange', line_width = 2)
p02.circle(x = 'date', y = 'rec_over_pos', source = source_db, legend_label="Recovered/Positives",
          color="green", size = 10, fill_alpha = 0.5, line_color = 'green', line_width = 2)
p02.line(x = 'date', y = 'rec_over_pos', source = source_db, legend_label="Recovered/Positives",
          color="green", line_color = 'green', line_width = 2)
p02.circle(x = 'date', y = 'ic_over_pos', source = source_db, legend_label="IC/Positives",
          color="purple", size = 10, fill_alpha = 0.5, line_color = 'purple', line_width = 2)
p02.line(x = 'date', y = 'ic_over_pos', source = source_db, legend_label="IC/Positives",
          color="purple", line_color = 'purple', line_width = 2)
p02.circle(x = 'date', y = 'dead_over_pos', source = source_db, legend_label="Deaths/Positives",
          color="black", size = 10, fill_alpha = 0.5, line_color = 'black', line_width = 2)
p02.line(x = 'date', y = 'dead_over_pos', source = source_db, legend_label="Deaths/Positives",
          color="black", line_color = 'black', line_width = 2)

p02.yaxis[0].formatter = bkh_mod.NumeralTickFormatter(format="0.0%")
p02.legend.label_text_font_size = "9pt"
p02.background_fill_color ="gainsboro"
p02.sizing_mode = 'scale_width'
p02.legend.location = "top_left"
p02.legend.background_fill_alpha = 0.0
p02.legend.click_policy="hide"
p02.add_tools(bkh_mod.HoverTool(tooltips = frac_tooltips))

tile_provider = get_provider(Vendors.CARTODBPOSITRON)
p03 = bkh_plt.figure(x_range=(+600000, 2200000), y_range=(+4900000, 5500000),
                                        title="Spread Map for Italy",
           x_axis_type="mercator", y_axis_type="mercator",  
                    width=500)
p03.add_tile(tile_provider)
p03.circle(x='utm_coord_x', y='utm_coord_y', size="n_tot_case_map", color = 'magenta', fill_color="magenta", fill_alpha=0.1, source = source_db_reg, legend_label="Cases")
p03.circle(x='utm_coord_x', y='utm_coord_y', size="n_tot_pos_map", color = 'red', fill_color="red", fill_alpha=0.1, source = source_db_reg, legend_label="Positives")
p03.circle(x='utm_coord_x', y='utm_coord_y', size="n_ic_map", color = 'purple', fill_color="purple", fill_alpha=0.1, source = source_db_reg, legend_label="IC")
p03.circle(x='utm_coord_x', y='utm_coord_y', size="n_tot_rec_map", color = 'green', fill_color="green", fill_alpha=0.1, source = source_db_reg, legend_label="Recovered")
p03.circle(x='utm_coord_x', y='utm_coord_y', size="n_dead_map", color = 'black', fill_color="black", fill_alpha=0.1, source = source_db_reg, legend_label="Deaths")
glyph = bkh_mod.Text(x='utm_coord_x', y='utm_coord_y', text="name_reg", angle=0.0, text_color="black", text_font_size = "8pt")
p03.add_glyph(source_db_reg, glyph)
p03.sizing_mode = 'scale_width'
p03.legend.label_text_font_size = "9pt"
p03.legend.location = "top_right"
p03.legend.background_fill_alpha = 0.0
p03.legend.click_policy="hide"
p03.add_tools(bkh_mod.HoverTool(tooltips = reg_tooltips))

##############################################################################

p04 = bkh_plt.figure(tools = TOOLS_NEW, width=500, #height=350,
                    title="Daily Absolute Variations for Italy",
                    #x_axis_label='x', #y_axis_label='y',
                    x_axis_type='datetime', #y_axis_type="log"
                    )
#p04.y_range=bkh_mod.Range1d(bottom, top)
cir = {}; lin = {}
#cir["Swabs"] = p04.circle(x = 'date', y = 'n_swab_diff', source = source_db, legend_label="Swabs",
#          color="blue", size = 10, fill_alpha = 0.5, line_color = 'blue', line_width = 2)
#lin["Swabs"] = p04.line(x = 'date', y = 'n_swab_diff', source = source_db, legend_label="Swabs",
#          color="blue", line_color = 'blue', line_width = 2)
cir["Cases"] = p04.circle(x = 'date', y = 'n_tot_case_diff', source = source_db, legend_label="Cases",
          color="magenta", size = 10, fill_alpha = 0.5, line_color = 'magenta', line_width = 2)
lin["Cases"] = p04.line(x = 'date', y = 'n_tot_case_diff', source = source_db, legend_label="Cases",
          color="magenta", line_color = 'magenta', line_width = 2)
p04.circle(x = 'date', y = 'n_tot_pos_diff', source = source_db, legend_label="Positives",
          color="red", size = 10, fill_alpha = 0.5, line_color = 'red', line_width = 2)
p04.line(x = 'date', y = 'n_tot_pos_diff', source = source_db, legend_label="Positives",
          color="red", line_color = 'red', line_width = 2)
cir["Hospitalized"] = p04.circle(x = 'date', y = 'n_tot_hosp_diff', source = source_db, legend_label="Hospitalized",
          color="orange", size = 10, fill_alpha = 0.5, line_color = 'orange', line_width = 2)
lin["Hospitalized"] = p04.line(x = 'date', y = 'n_tot_hosp_diff', source = source_db, legend_label="Hospitalized",
          color="orange", line_color = 'orange', line_width = 2)
cir["Recovered"] = p04.circle(x = 'date', y = 'n_recovered_diff', source = source_db, legend_label="Recovered",
          color="green", size = 10, fill_alpha = 0.5, line_color = 'green', line_width = 2)
lin["Recovered"] = p04.line(x = 'date', y = 'n_recovered_diff', source = source_db, legend_label="Recovered",
          color="green", line_color = 'green', line_width = 2)
p04.circle(x = 'date', y = 'n_ic_diff', source = source_db, legend_label="IC",
          color="purple", size = 10, fill_alpha = 0.5, line_color = 'purple', line_width = 2)
p04.line(x = 'date', y = 'n_ic_diff', source = source_db, legend_label="IC",
          color="purple", line_color = 'purple', line_width = 2)
p04.circle(x = 'date', y = 'n_dead_diff', source = source_db, legend_label="Deaths",
          color="black", size = 10, fill_alpha = 0.5, line_color = 'black', line_width = 2)
p04.line(x = 'date', y = 'n_dead_diff', source = source_db, legend_label="Deaths",
          color="black", line_color = 'black', line_width = 2)
#lin["Swabs"].visible = False
lin["Cases"].visible = False
lin["Hospitalized"].visible = False
lin["Recovered"].visible = False
#cir["Swabs"].visible = False
cir["Cases"].visible = False
cir["Hospitalized"].visible = False
cir["Recovered"].visible = False

p04.background_fill_color ="gainsboro"
p04.sizing_mode = 'scale_width'
p04.legend.location = "top_left"
p04.legend.label_text_font_size = "9pt"
p04.legend.background_fill_alpha = 0.0
p04.legend.click_policy="hide"
p04.yaxis[0].formatter = bkh_mod.NumeralTickFormatter(format="0,000")
p04.add_tools(bkh_mod.HoverTool(tooltips = diff_tooltips))

p05 = bkh_plt.figure(tools = TOOLS_NEW, width=500, #height=350,
                    title="Daily Relative Variations for Italy",
                    #x_axis_label='x', #y_axis_label='y',
                    x_axis_type='datetime')
cir = {}; lin = {}
bottom, top = -0.2, 1
p05.y_range=bkh_mod.Range1d(bottom, top)
cir["Swabs"] = p05.circle(x = 'date', y = 'n_swab_pct', source = source_db, legend_label="Swabs",
          color="blue", size = 10, fill_alpha = 0.5, line_color = 'blue', line_width = 2)
lin["Swabs"] = p05.line(x = 'date', y = 'n_swab_pct', source = source_db, legend_label="Swabs",
          color="blue", line_color = 'blue', line_width = 2)
cir["Cases"] = p05.circle(x = 'date', y = 'n_tot_case_pct', source = source_db, legend_label="Cases",
          color="magenta", size = 10, fill_alpha = 0.5, line_color = 'magenta', line_width = 2)
lin["Cases"] = p05.line(x = 'date', y = 'n_tot_case_pct', source = source_db, legend_label="Cases",
          color="magenta", line_color = 'magenta', line_width = 2)
p05.circle(x = 'date', y = 'n_tot_pos_pct', source = source_db, legend_label="Positives",
          color="red", size = 10, fill_alpha = 0.5, line_color = 'red', line_width = 2)
p05.line(x = 'date', y = 'n_tot_pos_pct', source = source_db, legend_label="Positives",
          color="red", line_color = 'red', line_width = 2)
cir["Hospitalized"] = p05.circle(x = 'date', y = 'n_tot_hosp_pct', source = source_db, legend_label="Hospitalized",
          color="orange", size = 10, fill_alpha = 0.5, line_color = 'orange', line_width = 2)
lin["Hospitalized"] = p05.line(x = 'date', y = 'n_tot_hosp_pct', source = source_db, legend_label="Hospitalized",
          color="orange", line_color = 'orange', line_width = 2)
cir["Recovered"] = p05.circle(x = 'date', y = 'n_recovered_pct', source = source_db, legend_label="Recovered",
          color="green", size = 10, fill_alpha = 0.5, line_color = 'green', line_width = 2)
lin["Recovered"] = p05.line(x = 'date', y = 'n_recovered_pct', source = source_db, legend_label="Recovered",
          color="green", line_color = 'green', line_width = 2)
p05.circle(x = 'date', y = 'n_ic_pct', source = source_db, legend_label="IC",
          color="purple", size = 10, fill_alpha = 0.5, line_color = 'purple', line_width = 2)
p05.line(x = 'date', y = 'n_ic_pct', source = source_db, legend_label="IC",
          color="purple", line_color = 'purple', line_width = 2)
p05.circle(x = 'date', y = 'n_dead_pct', source = source_db, legend_label="Deaths",
          color="black", size = 10, fill_alpha = 0.5, line_color = 'black', line_width = 2)
lin["Deaths"] = p05.line(x = 'date', y = 'n_dead_pct', source = source_db, legend_label="Deaths",
          color="black", line_color = 'black', line_width = 2)
lin["Swabs"].visible = False
lin["Hospitalized"].visible = False
lin["Recovered"].visible = False
cir["Swabs"].visible = False
cir["Hospitalized"].visible = False
cir["Recovered"].visible = False
cir["Cases"].visible = False
lin["Cases"].visible = False

p05.background_fill_color ="gainsboro"
p05.sizing_mode = 'scale_width'
p05.legend.label_text_font_size = "9pt"
p05.legend.location = "top_right"
p05.legend.background_fill_alpha = 0.0
p05.legend.click_policy="hide"
p05.yaxis[0].formatter = bkh_mod.NumeralTickFormatter(format="0.0%")
p05.add_tools(bkh_mod.HoverTool(tooltips = pct_tooltips))

p06 = bkh_plt.figure(x_range=(+600000, 2200000), y_range=(+4900000, 5500000),
                                        title="Variations Spread Map",
           x_axis_type="mercator", y_axis_type="mercator",  
                    width=500)
p06.add_tile(tile_provider)
p06.circle(x='utm_coord_x', y='utm_coord_y', size="n_var_case_map", color = 'magenta', fill_color="magenta", fill_alpha=0.1, source = source_db_reg_var, legend_label="Cases")
p06.circle(x='utm_coord_x', y='utm_coord_y', size="n_var_pos_map", color = 'red', fill_color="red", fill_alpha=0.1, source = source_db_reg_var, legend_label="Positives")
p06.circle(x='utm_coord_x', y='utm_coord_y', size="n_var_rec_map", color = 'green',fill_color="green", fill_alpha=0.1, source = source_db_reg_var, legend_label="Recovered")
p06.circle(x='utm_coord_x', y='utm_coord_y', size="n_var_ic_map", color = 'purple',fill_color="purple", fill_alpha=0.1, source = source_db_reg_var, legend_label="IC")
p06.circle(x='utm_coord_x', y='utm_coord_y', size="n_var_dead_map", color = 'black',fill_color="black", fill_alpha=0.1, source = source_db_reg_var, legend_label="Deaths")
glyph = bkh_mod.Text(x='utm_coord_x', y='utm_coord_y', text="name_reg", angle=0.0, text_color="black", text_font_size = "8pt")
p06.add_glyph(source_db_reg_var, glyph)
p06.sizing_mode = 'scale_width'
p06.legend.label_text_font_size = "9pt"
p06.legend.background_fill_alpha = 0.0
p06.legend.location = "top_right"
p06.legend.click_policy="hide"
p06.add_tools(bkh_mod.HoverTool(tooltips = reg_var_tooltips))
##############################################################################
    
p07 = bkh_plt.figure(tools = TOOLS_NEW, width=500, #height=350,
                    title="Trend of cumulative values for Apulia Region",
                    #x_axis_label='x', #y_axis_label='y',
                    x_axis_type='datetime', y_axis_type = "log")
p07.circle(x = 'date', y = 'n_swab', source = source_db_reg_puglia, legend_label="Swabs",
          color="blue", size = 10, fill_alpha = 0.5, line_color = 'blue', line_width = 2)
p07.line(x = 'date', y = 'n_swab', source = source_db_reg_puglia, legend_label="Swabs",
           color="blue", line_color = 'blue', alpha = 1, line_width = 2)
p07.circle(x = 'date', y = 'n_tot_case', source = source_db_reg_puglia, legend_label="Cases",
          color="magenta", size = 10, line_color = 'magenta', fill_alpha = 0.5, line_width = 2)
p07.line(x = 'date', y = 'n_tot_case', source = source_db_reg_puglia, legend_label="Cases",
          color="magenta", line_color = 'magenta', alpha = 1, line_width = 2)
p07.circle(x = 'date', y = 'n_tot_pos', source = source_db_reg_puglia, legend_label="Positives",
          color="red", size = 10, line_color = 'red', fill_alpha = 0.5, line_width = 2)
p07.line(x = 'date', y = 'n_tot_pos', source = source_db_reg_puglia, legend_label="Positives",
          color="red", line_color = 'red', alpha = 1, line_width = 2)
p07.circle(x = 'date', y = 'n_tot_hosp', source = source_db_reg_puglia, legend_label="Hospitalized",
          color="orange", size = 10, line_color = 'orange', fill_alpha = 0.5, line_width = 2)
p07.line(x = 'date', y = 'n_tot_hosp', source = source_db_reg_puglia, legend_label="Hospitalized",
          color="orange", line_color = 'orange', alpha = 1, line_width = 2)
p07.circle(x = 'date', y = 'n_recovered', source = source_db_reg_puglia, legend_label="Recovered",
          color="green", size = 10, line_color = 'green', fill_alpha = 0.5, line_width = 2)
p07.line(x = 'date', y = 'n_recovered', source = source_db_reg_puglia, legend_label="Recovered",
          color="green", line_color = 'green', alpha = 1, line_width = 2)
p07.circle(x = 'date', y = 'n_ic', source = source_db_reg_puglia, legend_label="IC",
          color="purple", size = 10, line_color = 'purple', fill_alpha = 0.5, line_width = 2)
p07.line(x = 'date', y = 'n_ic', source = source_db_reg_puglia, legend_label="IC",
          color="purple", line_color = 'purple', alpha = 1, line_width = 2)
p07.circle(x = 'date', y = 'n_dead', source = source_db_reg_puglia, legend_label="Deaths",
          color="black", size = 10, line_color = 'black', fill_alpha = 0.5, line_width = 2)
p07.line(x = 'date', y = 'n_dead', source = source_db_reg_puglia, legend_label="Deaths",
          color="black", line_color = 'black', alpha = 1, line_width = 2)


p07.background_fill_color ="gainsboro"
p07.sizing_mode = 'scale_width'
p07.legend.location = "top_left"
p07.legend.label_text_font_size = "9pt"
p07.legend.background_fill_alpha = 0.0
p07.legend.click_policy="hide"
p07.yaxis[0].formatter = bkh_mod.NumeralTickFormatter(format="0,000")
p07.add_tools(bkh_mod.HoverTool(tooltips = abs_tooltips))    
    
p08 = bkh_plt.figure(tools = TOOLS_NEW, width=500, #height=350,
                    title="Daily variations for Apulia Region",
                    #x_axis_label='x', #y_axis_label='y',
                    x_axis_type='datetime', 
                    #y_axis_type = "log"
                    )
# p08.circle(x = 'date', y = 'n_swab_diff', source = source_db_reg_puglia, legend_label="Swabs",
#           color="blue", size = 10, fill_alpha = 0.5, line_color = 'blue', line_width = 2)
# p08.line(x = 'date', y = 'n_swab_diff', source = source_db_reg_puglia, legend_label="Swabs",
#           color="blue", line_color = 'blue', line_width = 2)
p08.circle(x = 'date', y = 'n_tot_case_diff', source = source_db_reg_puglia, legend_label="Cases",
          color="magenta", size = 10, fill_alpha = 0.5, line_color = 'magenta', line_width = 2)
p08.line(x = 'date', y = 'n_tot_case_diff', source = source_db_reg_puglia, legend_label="Cases",
          color="magenta", line_color = 'magenta', line_width = 2)
p08.circle(x = 'date', y = 'n_tot_pos_diff', source = source_db_reg_puglia, legend_label="Positives",
          color="red", size = 10, fill_alpha = 0.5, line_color = 'red', line_width = 2)
p08.line(x = 'date', y = 'n_tot_pos_diff', source = source_db_reg_puglia, legend_label="Positives",
          color="red", line_color = 'red', line_width = 2)
p08.circle(x = 'date', y = 'n_tot_hosp_diff', source = source_db_reg_puglia, legend_label="Hospitalized",
          color="orange", size = 10, fill_alpha = 0.5, line_color = 'orange', line_width = 2)
p08.line(x = 'date', y = 'n_tot_hosp_diff', source = source_db_reg_puglia, legend_label="Hospitalized",
          color="orange", line_color = 'orange', line_width = 2)
p08.circle(x = 'date', y = 'n_recovered_diff', source = source_db_reg_puglia, legend_label="Recovered",
          color="green", size = 10, fill_alpha = 0.5, line_color = 'green', line_width = 2)
p08.line(x = 'date', y = 'n_recovered_diff', source = source_db_reg_puglia, legend_label="Recovered",
          color="green", line_color = 'green', line_width = 2)
p08.circle(x = 'date', y = 'n_ic_diff', source = source_db_reg_puglia, legend_label="IC",
          color="purple", size = 10, fill_alpha = 0.5, line_color = 'purple', line_width = 2)
p08.line(x = 'date', y = 'n_ic_diff', source = source_db_reg_puglia, legend_label="IC",
          color="purple", line_color = 'purple', line_width = 2)
p08.circle(x = 'date', y = 'n_dead_diff', source = source_db_reg_puglia, legend_label="Deaths",
          color="black", size = 10, fill_alpha = 0.5, line_color = 'black', line_width = 2)
p08.line(x = 'date', y = 'n_dead_diff', source = source_db_reg_puglia, legend_label="Deaths",
          color="black", line_color = 'black', line_width = 2)
p08.background_fill_color ="gainsboro"
p08.sizing_mode = 'scale_width'
p08.legend.location = "top_left"
p08.legend.label_text_font_size = "9pt"
p08.legend.background_fill_alpha = 0.0
p08.legend.click_policy="hide"
p08.yaxis[0].formatter = bkh_mod.NumeralTickFormatter(format="0,000")
p08.add_tools(bkh_mod.HoverTool(tooltips = diff_tooltips))

p09 = bkh_plt.figure(tools = TOOLS_NEW, width=500, #height=350,
                    title="Trend of proportions for Apulia Region",
                    #x_axis_label='x', #y_axis_label='y',
                    x_axis_type='datetime')
p09.circle(x = 'date', y = 'pos_over_swab', source = source_db_reg_puglia, legend_label="Positive/Swabs",
          color="red", size = 10, fill_alpha = 0.5, line_color = 'red', line_width = 2)
p09.line(x = 'date', y = 'pos_over_swab', source = source_db_reg_puglia, legend_label="Positive/Swabs",
          color="red", line_color = 'red', line_width = 2)
p09.circle(x = 'date', y = 'hosp_over_pos', source = source_db_reg_puglia, legend_label="Hospit/Positives",
          color="orange", size = 10, fill_alpha = 0.5, line_color = 'orange', line_width = 2)
p09.line(x = 'date', y = 'hosp_over_pos', source = source_db_reg_puglia, legend_label="Hospit/Positives",
          color="orange", line_color = 'orange', line_width = 2)
p09.circle(x = 'date', y = 'rec_over_pos', source = source_db_reg_puglia, legend_label="Recovered/Positives",
          color="green", size = 10, fill_alpha = 0.5, line_color = 'green', line_width = 2)
p09.line(x = 'date', y = 'rec_over_pos', source = source_db_reg_puglia, legend_label="Recovered/Positives",
          color="green", line_color = 'green', line_width = 2)
p09.circle(x = 'date', y = 'ic_over_pos', source = source_db_reg_puglia, legend_label="IC/Positives",
          color="purple", size = 10, fill_alpha = 0.5, line_color = 'purple', line_width = 2)
p09.line(x = 'date', y = 'ic_over_pos', source = source_db_reg_puglia, legend_label="IC/Positives",
          color="purple", line_color = 'purple', line_width = 2)
p09.circle(x = 'date', y = 'dead_over_pos', source = source_db_reg_puglia, legend_label="Deaths/Positives",
          color="black", size = 10, fill_alpha = 0.5, line_color = 'black', line_width = 2)
p09.line(x = 'date', y = 'dead_over_pos', source = source_db_reg_puglia, legend_label="Deaths/Positives",
          color="black", line_color = 'black', line_width = 2)

p09.yaxis[0].formatter = bkh_mod.NumeralTickFormatter(format="0.0%")
p09.legend.label_text_font_size = "8pt"
p09.background_fill_color ="gainsboro"
p09.sizing_mode = 'scale_width'
p09.legend.label_text_font_size = "9pt"
p09.legend.background_fill_alpha = 0.0
p09.legend.location = "top_right"
p09.legend.click_policy="hide"
p09.add_tools(bkh_mod.HoverTool(tooltips = frac_tooltips))



##############################################################################

p10 = bkh_plt.figure(x_range=(+600000, 2200000), y_range=(+4900000, 5500000),
                                        title="Spread Map for Italian Provinces",
            x_axis_type="mercator", y_axis_type="mercator",  
                    width=500, #height = 650
                    )
p10.add_tile(tile_provider) #"n_tot_case_map"
y = X_province_last['n_tot_case_map'].values
mapper = linear_cmap(field_name='n_tot_case_map', palette=bkh_pal.RdBu10 ,low=min(y) ,high=max(y))
p10.circle(x='utm_coord_x', y='utm_coord_y', size='n_tot_case_map', color = mapper, fill_color=mapper, fill_alpha=0.7, source = source_db_prov, legend_label="Cases")
#glyph = bkh_mod.Text(x='utm_coord_x', y='utm_coord_y', text="name_prov", angle=0.0, text_color="black", text_font_size = "8pt")
#p10.add_glyph(source_db_prov, glyph)
p10.sizing_mode = 'scale_width'
p10.legend.location = "top_right"
p10.legend.label_text_font_size = "9pt"
p10.legend.background_fill_alpha = 0.0
p10.legend.click_policy="hide"
p10.add_tools(bkh_mod.HoverTool(tooltips = prov_tooltips))

p11 = bkh_plt.figure(tools = TOOLS_NEW, width=500, #height = 650,
                    title="Confirmed cases for Apulia provinces",
                    #x_axis_label='x', #y_axis_label='y',
                    x_axis_type='datetime', y_axis_type = "log")
p11.line(x = 'date', y = 'n_tot_case', source = source_db_prov_bari, legend_label="Bari",
          color="blue", line_color = 'blue', line_width = 2)
p11.circle(x = 'date', y = 'n_tot_case', source = source_db_prov_bari, legend_label="Bari",
          color="blue", size = 10, fill_alpha = 0.5, line_color = 'blue', line_width = 2)
p11.line(x = 'date', y = 'n_tot_case', source = source_db_prov_bat, legend_label="B-A-T",
          color="red", line_color = 'red', line_width = 2)
p11.circle(x = 'date', y = 'n_tot_case', source = source_db_prov_bat, legend_label="B-A-T",
          color="red", size = 10, fill_alpha = 0.5, line_color = 'red', line_width = 2)
p11.line(x = 'date', y = 'n_tot_case', source = source_db_prov_brindisi, legend_label="Brindisi",
          color="green", line_color = 'green', line_width = 2)
p11.circle(x = 'date', y = 'n_tot_case', source = source_db_prov_brindisi, legend_label="Brindisi",
          color="green", size = 10, fill_alpha = 0.5, line_color = 'green', line_width = 2)
p11.line(x = 'date', y = 'n_tot_case', source = source_db_prov_foggia, legend_label="Foggia",
          color="magenta", line_color = 'magenta', line_width = 2)
p11.circle(x = 'date', y = 'n_tot_case', source = source_db_prov_foggia, legend_label="Foggia",
          color="magenta", size = 10, fill_alpha = 0.5, line_color = 'magenta', line_width = 2)
p11.line(x = 'date', y = 'n_tot_case', source = source_db_prov_lecce, legend_label="Lecce",
          color="orange", line_color = 'orange', line_width = 2)
p11.circle(x = 'date', y = 'n_tot_case', source = source_db_prov_lecce, legend_label="Lecce",
          color="orange", size = 10, fill_alpha = 0.5, line_color = 'orange', line_width = 2)
p11.line(x = 'date', y = 'n_tot_case', source = source_db_prov_taranto, legend_label="Taranto",
          color="purple", line_color = 'purple', line_width = 2)
p11.circle(x = 'date', y = 'n_tot_case', source = source_db_prov_taranto, legend_label="Taranto",
          color="purple", size = 10, fill_alpha = 0.5, line_color = 'purple', line_width = 2)


p11.yaxis[0].formatter = bkh_mod.NumeralTickFormatter(format="0,000")
p11.legend.label_text_font_size = "9pt"
p11.background_fill_color ="gainsboro"
p11.sizing_mode = 'scale_width'
p11.legend.location = "top_left"
p11.legend.background_fill_alpha = 0.0
p11.legend.click_policy="hide"
p11.add_tools(bkh_mod.HoverTool(tooltips = puglia_tooltips))

p12 = bkh_plt.figure(tools = TOOLS_NEW, width=500, #height = 650,
                    title="Confirmed cases over province population",
                    #x_axis_label='x', #y_axis_label='y',
                    x_axis_type='datetime', #y_axis_type = "log"
                    )
p12.line(x = 'date', y = 'n_tot_case_unit', source = source_db_prov_bari, legend_label="Bari",
          color="blue", line_color = 'blue', line_width = 2)
p12.circle(x = 'date', y = 'n_tot_case_unit', source = source_db_prov_bari, legend_label="Bari",
          color="blue", size = 10, fill_alpha = 0.5, line_color = 'blue', line_width = 2)
p12.line(x = 'date', y = 'n_tot_case_unit', source = source_db_prov_bat, legend_label="B-A-T",
          color="red", line_color = 'red', line_width = 2)
p12.circle(x = 'date', y = 'n_tot_case_unit', source = source_db_prov_bat, legend_label="B-A-T",
          color="red", size = 10, fill_alpha = 0.5, line_color = 'red', line_width = 2)
p12.line(x = 'date', y = 'n_tot_case_unit', source = source_db_prov_brindisi, legend_label="Brindisi",
          color="green", line_color = 'green', line_width = 2)
p12.circle(x = 'date', y = 'n_tot_case_unit', source = source_db_prov_brindisi, legend_label="Brindisi",
          color="green", size = 10, fill_alpha = 0.5, line_color = 'green', line_width = 2)
p12.line(x = 'date', y = 'n_tot_case_unit', source = source_db_prov_foggia, legend_label="Foggia",
          color="magenta", line_color = 'magenta', line_width = 2)
p12.circle(x = 'date', y = 'n_tot_case_unit', source = source_db_prov_foggia, legend_label="Foggia",
          color="magenta", size = 10, fill_alpha = 0.5, line_color = 'magenta', line_width = 2)
p12.line(x = 'date', y = 'n_tot_case_unit', source = source_db_prov_lecce, legend_label="Lecce",
          color="orange", line_color = 'orange', line_width = 2)
p12.circle(x = 'date', y = 'n_tot_case_unit', source = source_db_prov_lecce, legend_label="Lecce",
          color="orange", size = 10, fill_alpha = 0.5, line_color = 'orange', line_width = 2)
p12.line(x = 'date', y = 'n_tot_case_unit', source = source_db_prov_taranto, legend_label="Taranto",
          color="purple", line_color = 'purple', line_width = 2)
p12.circle(x = 'date', y = 'n_tot_case_unit', source = source_db_prov_taranto, legend_label="Taranto",
          color="purple", size = 10, fill_alpha = 0.5, line_color = 'purple', line_width = 2)


p12.yaxis[0].formatter = bkh_mod.NumeralTickFormatter(format="0.00%")
p12.legend.label_text_font_size = "9pt"
p12.background_fill_color ="gainsboro"
p12.sizing_mode = 'scale_width'
p12.legend.location = "top_left"
p12.legend.background_fill_alpha = 0.0
p12.legend.click_policy="hide"
p12.add_tools(bkh_mod.HoverTool(tooltips = puglia_tooltips))






###############################################################################
cover_world = bkh_mod_w.Div(text =
"""
<font size="5"> Situation as of <b> %s</b></font><br>
<font size="5"> <b> %s</b></font> Cases&nbsp;
<font size="4"> <b> %s</b></font> Recovered&nbsp;
<font size="4"> <b> %s</b></font> Deaths
"""% (str(end_world),
'{:,}'.format((X_world_last.conf.sum())),
'{:,}'.format(int(X_world_last.reco.sum())),
'{:,}'.format(int(X_world_last.dead.sum()))), #width=250, height=75
)
cover_world.sizing_mode = 'scale_width'

p13 = bkh_plt.figure(x_range=(source_db_world.data['utm_coord_x'].min()/1.2, source_db_world.data['utm_coord_x'].max()/1.2), 
                     y_range=(source_db_world.data['utm_coord_y'].min(), source_db_world.data['utm_coord_y'].max()),
                     title="World map of outbreak spread (hover on points to see further details)",
                     x_axis_type="mercator", y_axis_type="mercator",  
                     width=1500, 
                     #height = 650
                     )
p13.add_tile(tile_provider) #"n_tot_case_map"
p13.circle(x='utm_coord_x', y='utm_coord_y', size='n_conf_map', color = 'magenta', fill_color='magenta', fill_alpha=0.2, source = source_db_world, legend_label="Confirmed")
p13.circle(x='utm_coord_x', y='utm_coord_y', size='n_reco_map', color = 'green', fill_color='green', fill_alpha=0.2, source = source_db_world, legend_label="Recovered")
p13.circle(x='utm_coord_x', y='utm_coord_y', size='n_dead_map', color = 'black', fill_color='black', fill_alpha=0.2, source = source_db_world, legend_label="Deaths")
#glyph = bkh_mod.Text(x='utm_coord_x', y='utm_coord_y', text="name_prov", angle=0.0, text_color="black", text_font_size = "8pt")
#p10.add_glyph(source_db_prov, glyph)
p13.sizing_mode = 'scale_width'
p13.legend.location = "top_right"
p13.legend.label_text_font_size = "9pt"
p13.legend.background_fill_alpha = 0.0
p13.legend.click_policy="hide"
p13.add_tools(bkh_mod.HoverTool(tooltips = world_map_tooltips))

p14 = bkh_plt.figure(tools = TOOLS_NEW, width=500, #height=650,
                    title="World figures",
                    y_range=source_db_world_hist.data["Country"],
                    x_range=(0,source_db_world_hist.data["conf"].max()*1.1),
                    #x_axis_type = 'log'
                    #x_axis_label='x', #y_axis_label='y',
                    #x_axis_type='datetime'
                    )
p14.hbar(y = 'Country', right = 'conf', source = source_db_world_hist, legend_label="Confirmed",
           color="red", line_color = 'gainsboro', height = 1, alpha = 0.75)
p14.hbar(y = 'Country', right = 'reco', source = source_db_world_hist, legend_label="Recovered",
          color="green", line_color = 'gainsboro', height = 1,alpha = 0.75)
p14.hbar(y = 'Country', right = 'dead', source = source_db_world_hist, legend_label="Deaths",
          color="black", line_color = 'gainsboro', height = 1,alpha = 0.75)

p14.background_fill_color ="gainsboro"
p14.legend.label_text_font_size = "9pt"
p14.sizing_mode = 'scale_width'
p14.legend.location = "top_right"
p14.legend.background_fill_alpha = 0.0
p14.legend.click_policy="hide"
p14.xaxis[0].formatter = bkh_mod.NumeralTickFormatter(format="0,000")
#p14.xaxis.major_label_orientation = "vertical"
p14.add_tools(bkh_mod.HoverTool(tooltips = world_hist_tooltips))

p15 = bkh_plt.figure(tools = TOOLS_NEW, width=500, #height = 650,
                    title="Most hit country confirmed trends",
                    #x_axis_label='x', #y_axis_label='y',
                    x_axis_type='datetime', y_axis_type = "log")

cmap = bkh_pal.Spectral[N_MAIN_COUNTRY] 
k = 0
for country in source_db_world_conf_ts.data.keys():
    if (country !='index')&(country !='Date')&(country !='date_string'):
        lx = p15.line(x = 'Date', y = country, source = source_db_world_conf_ts, color = cmap[k], legend_label=country, line_width = 2)
        p15.circle(x = 'Date', y = country, source = source_db_world_conf_ts, legend_label= country,
          color=cmap[k], size = 8, line_color = cmap[k], alpha = 0.75, line_width = 2)
        #p15.add_tools(bkh_mod.HoverTool(tooltips="This is %s %s" % (country, 'y'), renderers=[lx]))
        p15.add_tools(bkh_mod.HoverTool( tooltips=[("Date", "@date_string"),  ("Country", country), ("Confirmed", "$y{0,000f}")], renderers=[lx]))
        #p15.add_tools(bkh_mod.HoverTool(tooltips = world_ts_tooltips))
        k = k + 1

p15.yaxis[0].formatter = bkh_mod.NumeralTickFormatter(format="0,000")
p15.legend.label_text_font_size = "9pt"
p15.background_fill_color ="gainsboro"
p15.sizing_mode = 'scale_width'
p15.legend.location = "top_left"
p15.legend.background_fill_alpha = 0.0
p15.legend.click_policy="hide"

p15b = bkh_plt.figure(tools = TOOLS_NEW, width=500, #height = 650,
                    title="Most hit country death trends",
                    #x_axis_label='x', #y_axis_label='y',
                    x_axis_type='datetime', y_axis_type = "log"
                    )

cmap = bkh_pal.Spectral[N_MAIN_COUNTRY] 
k = 0
for country in source_db_world_death_ts.data.keys():
    if (country !='index')&(country !='Date')&(country !='date_string'):
        p15b.line(x = 'Date', y = country, source = source_db_world_death_ts, color = cmap[k], legend_label=country, line_width = 2)
        lxb = p15b.circle(x = 'Date', y = country, source = source_db_world_death_ts, legend_label= country,
          color=cmap[k], size = 8, line_color = cmap[k], alpha = 0.75, line_width = 2)
        #p15.add_tools(bkh_mod.HoverTool(tooltips="This is %s %s" % (country, 'y'), renderers=[lx]))
        p15b.add_tools(bkh_mod.HoverTool( tooltips=[("Date", "@date_string"),  ("Country", country), ("Deaths", "$y{0,000f}")], renderers=[lxb]))
        #p15.add_tools(bkh_mod.HoverTool(tooltips = world_ts_tooltips))
        k = k + 1

p15b.yaxis[0].formatter = bkh_mod.NumeralTickFormatter(format="0,000")
p15b.legend.label_text_font_size = "9pt"
p15b.background_fill_color ="gainsboro"
p15b.sizing_mode = 'scale_width'
p15b.legend.location = "top_left"
p15b.legend.background_fill_alpha = 0.0
p15b.legend.click_policy="hide"

##############################################################################

p16 = bkh_plt.figure(tools = TOOLS_NEW, width=500, #height = 650,
                    title="SI Model Forecast (for different start forecast dates)",
                    #x_axis_label='x', 
                    y_axis_label='Number of infected',
                    x_axis_type='datetime', y_axis_type = "log")

cmap = bkh_pal.inferno(len(si_scenario_columns)) 
for i in range(len(si_scenario_columns)):
    lsi1 = p16.line(x = 'Date', y = str(si_scenario_columns[i]), source = source_si, color = cmap[i], #legend_label=str(si_scenario_columns[i]), 
                    line_width = 2)
    p16.add_tools(bkh_mod.HoverTool( tooltips=[("Date", "@date_string"),  ("Start forecast date", str(si_scenario_columns[i])), ("Infected", "$y{0,000f}")], renderers=[lsi1]))
lsi16b = p16.circle(x = 'date', y = "n_tot_pos", source = source_db, legend_label= "observed",
           color="red", size = 8, line_color = "red", alpha = 0.5, line_width = 2)
p16.add_tools(bkh_mod.HoverTool(tooltips = abs_tooltips, renderers=[lsi16b]))

p16.yaxis[0].formatter = bkh_mod.NumeralTickFormatter(format="0,000")
p16.legend.label_text_font_size = "9pt"
p16.background_fill_color ="gainsboro"
p16.sizing_mode = 'scale_width'
p16.legend.location = "bottom_right"
p16.legend.background_fill_alpha = 0.0
p16.legend.click_policy="hide"

p17 = bkh_plt.figure(tools = TOOLS_NEW, width=500, #height = 650,
                    title="SI Model Forecast  (for different start forecast dates)",
                    #x_axis_label='x', #y_axis_label='y',
                    x_axis_type='datetime', #y_axis_type = "log"
                    )

cmap = bkh_pal.inferno(len(si_scenario_columns))
k = 0
for i in range(len(si_scenario_columns)):
    lsi2 = p17.line(x = 'Date', y = str(si_scenario_columns[i]), source = source_si, color = cmap[i], #legend_label=str(si_scenario_columns[i])+" observations", 
                    line_width = 2)
    p17.add_tools(bkh_mod.HoverTool( tooltips=[("Date", "@date_string"),  ("Start forecast date", str(si_scenario_columns[i])), ("Infected", "$y{0,000f}")], renderers=[lsi2]))
    #p15.add_tools(bkh_mod.HoverTool(tooltips="This is %s %s" % (country, 'y'), renderers=[lx]))
    #p15.add_tools(bkh_mod.HoverTool( tooltips=[("Date", "@date_string"),  ("Country", country), ("Confirmed", "$y{0,000f}")], renderers=[lx]))
    #p15.add_tools(bkh_mod.HoverTool(tooltips = world_ts_tooltips))
    #k = k + 1
#lsi2c = p17.cross(x = 'Date', y = str(SCENARIO_RANGE[-1]), source = source_si, color = 'gray', 
#                   legend_label='forecast', size = 8, line_color = "gray", alpha = 0.5, line_width = 2)
#p17.add_tools(bkh_mod.HoverTool( tooltips=[("Date", "@date_string"),  ("n. observations", str(SCENARIO_RANGE[-1])), ("Infected", "$y{0,000f}")], renderers=[lsi2c]))
lsi17b = p17.circle(x = 'date', y = "n_tot_pos", source = source_db, legend_label= "observed",
           color="red", size = 8, line_color = "red", alpha = 0.5, line_width = 2)
p17.add_tools(bkh_mod.HoverTool(tooltips = abs_tooltips, renderers=[lsi17b]))

p17.yaxis[0].formatter = bkh_mod.NumeralTickFormatter(format="0,000")
p17.legend.label_text_font_size = "9pt"
p17.background_fill_color ="gainsboro"
p17.sizing_mode = 'scale_width'
p17.legend.location = "top_left"
p17.legend.background_fill_alpha = 0.0
p17.legend.click_policy="hide"

p18 = bkh_plt.figure(tools = TOOLS_NEW, width=500, #height = 650,
                    title=" Expected inflexion points for different size of observations",
                    #x_axis_label='x', #y_axis_label='y',
                    #y_range=(source_si_spike.data["y_spike"].min(), source_si_spike.data["y_spike"].max()),
                    x_axis_type='datetime', #y_axis_type = "log"
                    )

cmap = bkh_pal.Category20[len(SCENARIO_RANGE)] 
k = 0
p18.circle(x = 'date', y = "n_tot_pos", source = source_db, legend_label= "observed",
           color="magenta", size = 8, line_color = "magenta", alpha = 0.5, line_width = 2)
lsi3 = p18.cross(x = 'Date', y = "y_spike", source = source_si_spike, legend_label= "inflexion",
           color="black", size = 25, line_color = "black", alpha = 0.5, line_width = 2)
p18.add_tools(bkh_mod.HoverTool( tooltips=[("Exp. Spike date", "@date_string"),  ("Spike value:", "@y_spike{0,000f}"), ("n. observations", "@Scenario")], renderers=[lsi3]))
p18.diamond(x = 'Date', y = str(si_scenario_columns[-1]), source = source_si, color = "blue", legend_label=str(si_scenario_columns[-1])+ " observations", size = 8, alpha = 0.75, line_width = 2)
p18.yaxis[0].formatter = bkh_mod.NumeralTickFormatter(format="0,000")
p18.legend.label_text_font_size = "9pt"
p18.background_fill_color ="gainsboro"
p18.sizing_mode = 'scale_width'
p18.legend.location = "top_left"
p18.legend.background_fill_alpha = 0.0
p18.legend.click_policy="hide"

p18b = bkh_plt.figure(tools = TOOLS_NEW, width=500, #height = 650,
                    title="SI Model Forecast  (for different start forecast dates)",
                    #x_axis_label='x', #y_axis_label='y',
                    x_axis_type='datetime', #y_axis_type = "log"
                    )

cmap = bkh_pal.inferno(len(si_scenario_columns))
k = 0
for i in range(len(si_scenario_columns)):
    lsi2 = p18b.line(x = 'Date', y = str(si_scenario_columns[i]), source = source_si_diff, color = cmap[i], #legend_label=str(si_scenario_columns[i])+" observations", 
                    line_width = 2)
    p18b.add_tools(bkh_mod.HoverTool( tooltips=[("Date", "@date_string"),  ("Start forecast date", str(si_scenario_columns[i])), ("Infected", "$y{0,000f}")], renderers=[lsi2]))
    #p15.add_tools(bkh_mod.HoverTool(tooltips="This is %s %s" % (country, 'y'), renderers=[lx]))
    #p15.add_tools(bkh_mod.HoverTool( tooltips=[("Date", "@date_string"),  ("Country", country), ("Confirmed", "$y{0,000f}")], renderers=[lx]))
    #p15.add_tools(bkh_mod.HoverTool(tooltips = world_ts_tooltips))
    #k = k + 1
#lsi2c = p18b.cross(x = 'Date', y = str(SCENARIO_RANGE[-1]), source = source_si, color = 'gray', 
#                   legend_label='forecast', size = 8, line_color = "gray", alpha = 0.5, line_width = 2)
#p18b.add_tools(bkh_mod.HoverTool( tooltips=[("Date", "@date_string"),  ("n. observations", str(SCENARIO_RANGE[-1])), ("Infected", "$y{0,000f}")], renderers=[lsi2c]))
lsi2b = p18b.circle(x = 'date', y = "n_tot_pos_diff", source = source_db, legend_label= "observed",
           color="red", size = 8, line_color = "red", alpha = 0.5, line_width = 2)
p18b.add_tools(bkh_mod.HoverTool(tooltips = diff_tooltips, renderers=[lsi2b]))

p18b.yaxis[0].formatter = bkh_mod.NumeralTickFormatter(format="0,000")
p18b.legend.label_text_font_size = "9pt"
p18b.background_fill_color ="gainsboro"
p18b.sizing_mode = 'scale_width'
p18b.legend.location = "top_left"
p18b.legend.background_fill_alpha = 0.0
p18b.legend.click_policy="hide"



##############################################################################
p19 = bkh_plt.figure(tools = TOOLS_NEW, width=500, #height = 650,
                    title="SIR Model Forecast (for different sizes of susceptible population)",
                    #x_axis_label='x', 
                    y_axis_label='Number of infected',
                    x_axis_type='datetime', 
                    #y_axis_type = "log"
                    )

cmap = bkh_pal.inferno(len(list_of_scenarios0))
for i in range(len(list_of_scenarios0)):
    lsi4 = p19.line(x = 'Date', y = str(list_of_scenarios0[i]), source = source_sir0, color = cmap[i], legend_label='{:,}'.format((round(list_of_scenarios0[i],0))), line_width = 2)
    p19.add_tools(bkh_mod.HoverTool( tooltips=[("Date", "@date_string"),  ("Population Size", str(list_of_scenarios0[i])), ("Infected", "$y{0,000f}")], renderers=[lsi4]))
lsi4b = p19.circle(x = 'Date', y = "n_tot_pos", source = source_sir0, legend_label= "observed",
           color="red", size = 8, line_color = "red", alpha = 0.5, line_width = 2)
p19.add_tools(bkh_mod.HoverTool( tooltips=[("Date", "@date_string"), ("Positives without removed", "$y{0,000f}")], renderers=[lsi4b]))
p19.yaxis[0].formatter = bkh_mod.NumeralTickFormatter(format="0,000")
p19.legend.label_text_font_size = "9pt"
p19.background_fill_color ="gainsboro"
p19.sizing_mode = 'scale_width'
p19.legend.location = "bottom_right"
p19.legend.background_fill_alpha = 0.0
p19.legend.click_policy="hide"

p20 = bkh_plt.figure(tools = TOOLS_NEW, width=500, #àheight = 650,
                    title="SIR Model Forecast (for different sizes of susceptible population)",
                    #x_axis_label='x', 
                    #y_axis_label='Number of infected',
                    x_axis_type='datetime', y_axis_type = "log"
                    )

cmap = bkh_pal.inferno(len(list_of_scenarios0))
for i in range(len(list_of_scenarios0)):
    lsi5 = p20.line(x = 'Date', y = str(list_of_scenarios0[i]), source = source_sir0, color = cmap[i], legend_label='{:,}'.format((round(list_of_scenarios0[i],0))), line_width = 2)
    p20.add_tools(bkh_mod.HoverTool( tooltips=[("Date", "@date_string"),  ("Population Size", str(list_of_scenarios0[i])), ("Infected", "$y{0,000f}")], renderers=[lsi5]))
lsi5b = p20.circle(x = 'Date', y = "n_tot_pos", source = source_sir0, legend_label= "observed",
           color="red", size = 8, line_color = "red", alpha = 0.5, line_width = 2)
p20.add_tools(bkh_mod.HoverTool( tooltips=[("Date", "@date_string"), ("Positives without removed", "$y{0,000f}")], renderers=[lsi5b]))
p20.yaxis[0].formatter = bkh_mod.NumeralTickFormatter(format="0,000")
p20.legend.label_text_font_size = "9pt"
p20.background_fill_color ="gainsboro"
p20.sizing_mode = 'scale_width'
p20.legend.location = "bottom_right"
p20.legend.background_fill_alpha = 0.0
p20.legend.click_policy="hide"

p21 = bkh_plt.figure(tools = TOOLS_NEW, width=500, #àheight = 650,
                    title="SIR Model Forecast (for different sizes of susceptible population)",
                    #x_axis_label='x', 
                    #y_axis_label='Number of infected',
                    x_axis_type='datetime', y_axis_type = "log"
                    )

cmap = bkh_pal.inferno(len(list_of_scenarios1))
for i in range(len(list_of_scenarios1)):
    lsi6 = p21.line(x = 'Date', y = str(list_of_scenarios1[i]), source = source_sir1, color = cmap[i], legend_label='{:,}'.format((round(list_of_scenarios1[i],0))), line_width = 2)
    p21.add_tools(bkh_mod.HoverTool( tooltips=[("Date", "@date_string"),  ("Population Size", str(list_of_scenarios1[i])), ("Infected", "$y{0,000f}")], renderers=[lsi6]))
lsi6b = p21.circle(x = 'Date', y = "n_tot_pos", source = source_sir1, legend_label= "observed",
           color="red", size = 8, line_color = "red", alpha = 0.5, line_width = 2)
p21.add_tools(bkh_mod.HoverTool( tooltips=[("Date", "@date_string"), ("Positives without removed", "$y{0,000f}")], renderers=[lsi6b]))
#p21.yaxis[0].formatter = bkh_mod.NumeralTickFormatter(format="0,000.00")
p21.legend.label_text_font_size = "9pt"
p21.background_fill_color ="gainsboro"
p21.sizing_mode = 'scale_width'
p21.legend.location = "bottom_right"
p21.legend.background_fill_alpha = 0.0
p21.legend.click_policy="hide"
##############################################################################
note_it = bkh_mod_w.Div(text =
"""
Questa dashboard è stata sviluppata in Python Bokeh e riassume i dati rilevanti <br>
sulla diffusione in Italia, in Puglia e nel Mondo del COVID-19. <br><br>

Sono presenti inoltre delle previsioni basate sui modelli semplificati SI e SIR per l'Italia. <br><br>

I grafici sono completamente navigabili: è possibile eseguire zoom, attivare o
disattivare le curve (cliccando sulle voci della legenda) e far comparire ulteriori
informazioni di dettaglio passando con il mouse sui punti di interesse dei grafici. <br><br>

I risultati sono aggiornati ogni giorno alle 9:00 e alle 19:00 con un refresh automatico <br>
della sorgente dati dalle repository ufficiali su Github del<br>
<b> Dipartimento della Protezione Civile </b>: <a xmlns:dct="http://purl.org/dc/terms/" href="https://github.com/pcm-dpc/COVID-19" rel="dct:source">https://github.com/pcm-dpc/COVID-19</a><br>
<b> Johns Hopkins University (JHU CSSE) </b> : <a xmlns:dct="http://purl.org/dc/terms/" href="https://github.com/CSSEGISandData/COVID-19" rel="dct:source">https://github.com/CSSEGISandData/COVID-19</a>
<br><br><br>
La presente piattaforma sotto licenza Creative Commons <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons Licence" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/80x15.png" /></a><br>
<span xmlns:dct="http://purl.org/dc/terms/" property="dct:title"> ed è stata sviluppata </span> 
da <a xmlns:cc="http://creativecommons.org/ns#" href="https://www.linkedin.com/in/anglani/" property="cc:attributionName" rel="cc:attributionURL"> Roberto Anglani </a> e
<a xmlns:cc="http://creativecommons.org/ns#" href="https://www.linkedin.com/in/stefanonicotri/" property="cc:attributionName" rel="cc:attributionURL"> Stefano Nicotri </a> <br><br>

<b>Glossario:</b><br>
<i>Swabs</i>: Tamponi<br>
<i>Cases</i>: Casi<br>
<i>Positives</i>: Positivi/Infetti<br>
<i>Recovered</i>: Dimessi/Guariti<br>
<i>Hospitalized</i>: Ospedalizzati<br>
<i>IC</i>: Pazienti in Terapia Intensiva<br>
<i>Deaths</i>: Deceduti<br>
""", width=750, height=650
)
note_it.sizing_mode = 'scale_width'

note_en = bkh_mod_w.Div(text =
"""
The present dashboard has been developed in Python with Bokeh library and summarizes <br>
relevant data and information about COVID-19 spread outbreak in Italy, <br>
in the World and in Apulia Region of Italy. <br><br>

Forecasts based on SI and SIR models have been reported for Italy situation. <br><br>

All charts are completely navigable: you can zoom, activate or deactivate the curves
by clickin on legend items and show further details by hovering the mouse over the graphs. <br><br>

Charts are updated on a daily basis at 9:00 and 19:00 (Rome Time) through automatic refresh
of data sources at the official repository of
<b> Italian Civil Protection </b>: <a xmlns:dct="http://purl.org/dc/terms/" href="https://github.com/pcm-dpc/COVID-19" rel="dct:source">https://github.com/pcm-dpc/COVID-19</a><br>
<b> Johns Hopkins University (JHU CSSE) </b> : <a xmlns:dct="http://purl.org/dc/terms/" href="https://github.com/CSSEGISandData/COVID-19" rel="dct:source">https://github.com/CSSEGISandData/COVID-19</a>
<br><br><br>

The present platform is licensed under Creative Commons <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons Licence" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/80x15.png" /></a><br>
<span xmlns:dct="http://purl.org/dc/terms/" property="dct:title"> and has been developed </span> 
by <a xmlns:cc="http://creativecommons.org/ns#" href="https://www.linkedin.com/in/anglani/" property="cc:attributionName" rel="cc:attributionURL"> Roberto Anglani </a> and
<a xmlns:cc="http://creativecommons.org/ns#" href="https://www.linkedin.com/in/stefanonicotri/" property="cc:attributionName" rel="cc:attributionURL"> Stefano Nicotri </a> <br>
""", width=750, height=650
)
note_en.sizing_mode = 'scale_width'

    
### Costruzione delle Tab e dei titoli
ptab1 = bkh_plt.gridplot([[cover_1, LegClick, LegMeaning], [p01, p02, p03], [Sign]], toolbar_location = 'left')
ptab2 = bkh_plt.gridplot([[cover_1, LegClick, LegMeaning], [p04, p05, p06], [Sign]], toolbar_location = 'left')
ptab3 = bkh_plt.gridplot([[cover_1, LegClick, LegMeaning], [p07, p08, p09],[Sign]], toolbar_location = 'left')
ptab4 = bkh_plt.gridplot([[cover_1, LegClick, LegMeaning], [p10, p11, p12], [Sign]], toolbar_location = 'left')
ptab5 = bkh_plt.gridplot([[cover_world], [p13], [Sign]], toolbar_location = 'left')
ptab6 = bkh_plt.gridplot([[cover_world, LegClick, LegMeaning], [p15, p15b, p14], [Sign]], toolbar_location = 'left')
ptab7 = bkh_plt.gridplot([[cover_1, LegClick, LegMeaning], [p16, p17, p18b], [Sign]], toolbar_location = 'left')
ptab8 = bkh_plt.gridplot([[cover_1, LegClick, LegMeaning], [p19, p20, p21], [Sign]], toolbar_location = 'left')
ptabnote = bkh_plt.gridplot([[note_en, note_it]], toolbar_location = 'left')


ptab1.sizing_mode = 'scale_width'
ptab2.sizing_mode = 'scale_width'
ptab3.sizing_mode = 'scale_width'
ptab4.sizing_mode = 'scale_width'
ptab5.sizing_mode = 'scale_width'
ptab6.sizing_mode = 'scale_width'
ptab7.sizing_mode = 'scale_width'
ptab8.sizing_mode = 'scale_width'


ptabnote.sizing_mode = 'scale_width'

tab1 = bkh_mod.Panel(child=ptab1, title = 'Situation in Italy')
tab2 = bkh_mod.Panel(child=ptab2, title = 'Daily Variations in Italy')
tab3 = bkh_mod.Panel(child=ptab3, title = 'Situation in Apulia Region')
tab4 = bkh_mod.Panel(child=ptab4, title = 'Situation in Apulia Provinces')
tab5 = bkh_mod.Panel(child=ptab5, title = 'Global Spread Map')
tab6 = bkh_mod.Panel(child=ptab6, title = 'Global Statistics')
tab7 = bkh_mod.Panel(child=ptab7, title = 'SI Model - Italy')
tab8 = bkh_mod.Panel(child=ptab8, title = 'SIR Model - Italy')
tabnote = bkh_mod.Panel(child=ptabnote, title = 'About')



### Inserimento delle Tab in unico documento
tabsx = bkh_mod_w.Tabs(tabs = [
        tab1,
        tab2,
        tab7,
        tab8,
        tab3,
        tab4,
        tab5,
        tab6,
        tabnote
        ])
curdoc().add_root(tabsx)
### Mostra output
#show(tabsx)
bkh_plt.save(tabsx)
### Reset dell'output per evitare che il file non aumenti la size ad ogni refresh
reset_output()
