'''

The following script fetches informations like : beam and no of antennas for each epoch IF the summary of gmrt log files are provided (see all_logs : All_log_info_main.txt for more details).
'''

import psrchive, glob, os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import os
from astropy.time import Time
import matplotlib.ticker as ticker
from scipy.signal import savgol_filter as sv
from scipy.ndimage import gaussian_filter1d as gf1d
from scipy.ndimage import gaussian_filter as gf
import scipy.ndimage as ndimage
import scipy.signal as ss
import re



def get_log_info(given_mjd, freq_=None, Select_mode_only=['PA', 'IA', 'unknown']):
	pattern_mjd_start_stop = r'MJD_start\s*=\s*(\d*[.]*\d*);\s*MJD_stop\s*=\s*(\d*[.]*\d*).*'
	pattern_info = r'beam_dict\s*=\s*(.*)'
	for section_iter in obs_log_section:
		if not re.search(pattern_mjd_start_stop, section_iter):
			continue
		
		mjd_start, mjd_stop = list(map(float, re.search(pattern_mjd_start_stop, section_iter).groups()))
		#if not bool(mjd_start < given_mjd < mjd_stop):
		if not np.any((mjd_start < given_mjd) * (given_mjd < mjd_stop)):
			continue
		#print(section_iter)
		if re.search(pattern_info, section_iter):
			beam_dict = eval(re.findall(pattern_info, section_iter)[0])
		#return(beam_dict)
		else:
			continue

		if not any(Select_mode_only):
			if band_func(float(freq_)) in list(map(band_func, [i['Freq'] for i in beam_dict.values()])):
				return eval(re.findall(pattern_info, section_iter)[0])
			else:
				continue

		for mode_only in Select_mode_only:
			for beam_mode_val in beam_dict.values():
				if band_func(float(freq_)) != band_func(float(beam_mode_val['Freq'])):
					continue
				if beam_mode_val['Mode'].lower() == mode_only.lower():
					try:
						return beam_mode_val
					except:
						pass

def pfd_freq_mjd(pfd):
    var=psrchive.Archive_load(pfd)
    return var.get_frequencies(), var.get_mjds()

band_func = lambda nu : 5 if 1000<=nu<=1460 else (4 if 550<=nu<=1000 else (3 if 250<=nu<=500 else (2 if 100<=nu<=250 else 'No Band info')))


files = glob.glob('*.ar') + glob.glob('*.pfd')
all_logs = '/data/rahul/psr_file/ALL_MSP_PFD_fs2/All_log_info_main.txt'

with open(all_logs, 'r') as email_file:
	all_log_content = email_file.read()
obs_log_section = all_log_content.split('################################################################################################')

# get_log_info |: input : given_mjd, freq, Select_mode_only=['PA', 'IA'] ||: output: dict(mode, Ant_no, Freq) or None

files_mjd_freq_file = []
Non_log_file_pfd = []
unknown_beam_info = []
Non_unique_freq_log_vs_pfd = []
for f in files:
	print('File name : ', f)
	freq, mjd = np.median(pfd_freq_mjd(f)[0]), pfd_freq_mjd(f)[1]
	ok_bool = True
	try:
		#info_arr = [np.median(mjd), freq, str(f)]
		info_arr = [mjd[0], freq, str(f)]
		filename_splite = [t for s in f.split('_') for t in s.split('.')]
		if 'ia' in filename_splite:
			mode_file = 'IA'
		elif 'pa' in filename_splite:
			mode_file = 'PA'
		else:
			mode_file = 'unknown'
		print('Before log_info_list ')
		log_info_list = get_log_info(mjd, freq, [mode_file])
		print('After log_info_list ')
		if not log_info_list:
			ok_bool *= False
			print('Warning ! For file : ', f,' Something is off!')
			#Non_log_file_pfd.append(f)
			#continue
			log_info_dict = get_log_info(mjd, freq, [])
			if not log_info_dict:
				print('Log file for file ', f,' Not Found!')
				pa_mode_ = mode_file
				info_arr = info_arr + [pa_mode_, 'TBD']
				Non_log_file_pfd.append(info_arr)
				files_mjd_freq_file.append(info_arr)
				print('----------------------------------------------------')
				continue
			beam_list = [outer_dict['Mode'].lower() for outer_dict in log_info_dict.values() if 'Mode' in outer_dict and outer_dict['Mode'].lower() != 'voltage']
			Freq_list = [outer_dict['Freq'] for outer_dict in log_info_dict.values() if 'Freq' in outer_dict]
			if band_func(freq) in list(map(band_func, Freq_list)) or (band_func(freq) in [2,3] and (2 in list(map(band_func, Freq_list)) or 3 in list(map(band_func, Freq_list)))):
				for log_info_ in log_info_dict.values():
					#'''
					if not 'Mode' in log_info_.keys():
						print('Beam mode not mentioned in log file ! : Type 1')
						#pa_mode_ = mode_file
					#'''
					if not 'Ant_no' in log_info_.keys():
						print('Antenna Number not mentioned in logfile. Hence taking 24/12 antennas as default: Type 1')
					if band_func(float(log_info_['Freq'])) == band_func(freq):
						log_info_list = log_info_
						if not log_info_list['Mode'].lower() in beam_list:
							log_info_list['Mode'] = 'PA'
						print('Frequency Match found !')
						break
					if log_info_['Mode'].lower() == mode_file.lower():
						log_info_list = log_info_
					else:
						log_info_['Mode'] = mode_file.lower()
						log_info_list = log_info_
			else:
				print('Frequency info of log and filename doesn\'t match')
				#if band_func(freq) in [2,3] and (2 in list(map(band_func, Freq_list)) or 3 in list(map(band_func, Freq_list))):
				#log_info_list = log_info_
				
				Non_unique_freq_log_vs_pfd.append(f)
				pa_mode_ = mode_file
				info_arr = info_arr + [pa_mode_, 'TBD']
				files_mjd_freq_file.append(info_arr)
				print('----------------------------------------------------')
				continue
		if 'Mode' in log_info_list.keys():
			if log_info_list['Mode'].lower() == 'unknown':
				if mode_file.lower() == 'unknown':
					unknown_beam_info.append(f)
					ok_bool *= False
					print('Beam info is unknown')
				else:
					pa_mode_ = mode_file
				info_arr.append(pa_mode_)
			else:
				pa_mode_ = log_info_list['Mode']
				if log_info_list['Mode'].lower() == mode_file.lower():
					print('Beam info is consistent')
				else:
					ok_bool *= False
					
					print('Beam info is NOT consistent')
				info_arr.append(pa_mode_)
		else:
			print('Beam mode not mentioned in log file ! : Type 2')
			ok_bool *= False
			pa_mode_ = mode_file
			info_arr.append(pa_mode_)
		if 'Ant_no' in log_info_list.keys():
			ant_no_ = log_info_list['Ant_no']
		else:
			ok_bool *= False
			print('Antenna Number not mentioned in logfile. Hence taking 24/12 antennas as default: Type 2')
			ant_no_ = 'TBD'
		info_arr.append(ant_no_)
		files_mjd_freq_file.append(info_arr)
		if ok_bool:
			print('Everything ok for file : ', f)
		print('----------------------------------------------------')
	except:
		print('ERROR ! :', f)
		print('----------------------------------------------------')
		pass
	del freq, mjd, filename_splite, mode_file, log_info_list, ok_bool, pa_mode_
	try:
		del ant_no_, log_info_dict, beam_list, Freq_list
	except:
		pass




files_mjd_freq_file = np.array(files_mjd_freq_file, dtype=object)
files_mjd_freq_file = files_mjd_freq_file[files_mjd_freq_file[:,0].astype(float).argsort()]
# np.save('b5_files_mjd_freq_file', files_mjd_freq_file)
# files_mjd_freq_file = np.load('b*files_mjd_freq_file.npy', allow_pickle=True)


################################################################		Checks and details:

group_mjd_full = group(files_mjd_freq_file)

sum([isinstance(i, int) for i in files_mjd_freq_file[:, -1]])		# Checks the number of files which have antenna info from the log files, which indicates that these files have log files (assumptions if antenna info is there the file have their corresponding log file, which might not always true, like when the log file without antenna info. But the latter is very rare)

def group(X, th = 1000/86400):

	
	if X.ndim >1:
		B = np.append(np.diff(X[:,0]) > th, True)
	else:
		B = np.append(np.diff(X) > th, True)
	

	G = []
	g = []
	
	for i in np.arange(len(X)):

		if B[i]:
			
			g.append(X[i])
			G.append(g)
			
			g = []
			

			pass
		else:

			g.append(X[i])
			
	return np.array(G, dtype=object)

def search_epoch_group(epoch_):
	for G_ in group_mjd_full:
		if isinstance(epoch_, float):
			if epoch_ in np.array(G_)[:,0]:
				for g_ in G_:
					print(g_)
		elif isinstance(epoch_, str):
			search_str = '_'.join(np.array(G_)[:, 2])
			if epoch_ in [t for s in search_str.split('_') for t in s.split('.')]:
				for g_ in G_:
					print(g_)





