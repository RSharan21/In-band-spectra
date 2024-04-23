import psrchive, glob, os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import os
from astropy.time import Time
import matplotlib.ticker as ticker
from scipy.signal import savgol_filter as sv
from scipy.ndimage import gaussian_filter1d as gf
import scipy.ndimage as ndimage
import scipy.signal as ss

def pfd_data(pfd):
    var=psrchive.Archive_load(pfd)
    var.dedisperse()
    var.centre_max_bin()
    var.remove_baseline()
    return var.get_data(), var.get_frequencies(), var.get_mjds() #, var.get_dispersion_measure()



def pfd_data_b(pfd):
    var=psrchive.Archive_load(pfd)
    var.dedisperse()
    var.centre_max_bin()
    #var.remove_baseline()
    return var.get_data(), var.get_frequencies(), var.get_mjds() #, var.get_dispersion_measure()


def filter_clean_isolated_cells(array_, struct = ndimage.generate_binary_structure(2,2)):
	""" Return array with completely isolated single cells removed
	:param array: Array with completely isolated single cells
	:param struct: Structure array for generating unique regions
	:return: Array with minimum region size > 1
	"""

	filtered_array = np.copy(array_)
	#id_regions, num_ids = ndimage.label(filtered_array, structure=struct)
	try :
		id_regions, num_ids = ndimage.label(filtered_array, structure=struct)
	except:
		id_regions, num_ids = ndimage.label(filtered_array)
	id_sizes = np.array(ndimage.sum(array_, id_regions, range(num_ids + 1)))
	area_mask = (id_sizes == 1)
	filtered_array[area_mask[id_regions]] = 0
	return filtered_array



def filter_max_cells(array, struct = ndimage.generate_binary_structure(2,2)):
	""" Return array with completely isolated single cells removed
	:param array: Array with completely isolated single cells
	:param struct: Structure array for generating unique regions
	:return: Array with minimum region size > 1
	"""

	filtered_array = np.copy(array)
	try :
		id_regions, num_ids = ndimage.label(filtered_array, structure=struct)
	except:
		id_regions, num_ids = ndimage.label(filtered_array)
	id_sizes = np.array(ndimage.sum(filtered_array, id_regions, range(num_ids + 1)))
	if num_ids == 0:
		return array
	else:
		return id_regions == id_sizes.argmax()


def smoothing_2d_array(array_):
	# array_ should be a binary 2d array (boolean array)
	# in a 5x5 grid it searches for no of 1 values and only masks if # of points in such a grid is > 3
	# lastly it picks up the pixels common in array_ and mask5_
	
	mask5_ = ss.convolve2d(array_, np.ones((5,5)),mode='same') > 3
	return np.logical_and(array_, ndimage.binary_dilation(mask5_))

'''
def filter_prim_comp(array_):
	for i in range(array_.shape[0]):
		kernal = np.array([0,1,0]*5 + [1,1,1] + [0,1,0]*5).reshape(-1,3)
		mask_ = ss.correlate2d(array_,kernal,mode='same') > 0
	mask_ = filter_max_cells(mask_)
	return np.logical_and(array_,mask_)



def filter_prim_comp(array_, n):
	#mask_ = ss.correlate2d(array_, np.ones((2*array_.shape[0],1)),mode='same') > 0
	mask_ = ss.correlate2d(array_, np.ones((int(n),2)),mode='same') > 0
	mask_ = filter_max_cells(mask_)
	return np.logical_and(array_,mask_)
'''	
def mask_with_prim_comp(array_, thres_, n_chan_, prim_comp_ = True):
	mask = []
	#prim_comp_mask = []
	for i in np.arange(0,array_.shape[1], n_chan_):
		mask_per_chan = array_[:,i : i + n_chan_].sum(axis=(0,1))/(array_[:,i : i + n_chan_].std(axis=(0,1)) * np.sqrt(array_.shape[0]*array_[:,i : i + n_chan_].shape[1]) ) > thres_
		mask.append(mask_per_chan)

	mask = filter_clean_isolated_cells(mask)	# filter the isolated points
	if prim_comp_:
		ext_mask = ndimage.binary_fill_holes(mask, structure=np.array([[0,1,0]*3]).reshape(3,3))	# connects the mask vertically
		id_regions, num_ids = ndimage.label(ext_mask, structure=np.ones((3,3)) )	# id_regions, num_ids: contains different frequency dependent component info
		prim_comp_bool_array = np.array([np.any(np.where(id_regions == i)[-1] == id_regions.shape[-1]//2) for i in range(num_ids + 1) ])	# checks for the component intersecting the mid-point (horizontally speaking)(only if the main component is shifted to central phase bins) of the mask 
		prim_comp_bool_array[0] = False
		#prim_comp_region = filter_max_cells(id_regions)
		prim_comp_region = prim_comp_bool_array[id_regions]
		prim_comp_mask = np.logical_and(prim_comp_region, mask)
		
		return 	mask, prim_comp_mask	
	else:
		return mask



#########################################################################################################################################

def mask_with_prim_comp_flags(array_, flags_info, thres_, n_chan_, prim_comp_ = True):
	mask = []
	#prim_comp_mask = []
	for i in np.arange(0,array_.shape[1], n_chan_):
		support = np.zeros(array_.shape[1],dtype=bool)
		support[i: i+ n_chan_] = True
		mask_per_chan = array_[:,flags_info*support].sum((0,1))/(array_[:,flags_info*support].std((0,1)) * np.sqrt(array_[:,flags_info*support].shape[0]*array_[:,flags_info*support].shape[1]) ) > thres_
		mask.append(mask_per_chan)

	mask = filter_clean_isolated_cells(mask)	# filter the isolated points
	if prim_comp_:
		ext_mask = ndimage.binary_fill_holes(mask, structure=np.array([[0,1,0]*3]).reshape(3,3))	# connects the mask vertically
		id_regions, num_ids = ndimage.label(ext_mask, structure=np.ones((3,3)) )	# id_regions, num_ids: contains different frequency dependent component info
		prim_comp_bool_array = np.array([np.any(np.where(id_regions == i)[-1] == id_regions.shape[-1]//2) for i in range(num_ids + 1) ])	# checks for the component intersecting the mid-point (horizontally speaking)(only if the main component is shifted to central phase bins) of the mask 
		prim_comp_bool_array[0] = False
		#prim_comp_region = filter_max_cells(id_regions)
		prim_comp_region = prim_comp_bool_array[id_regions]
		prim_comp_mask = np.logical_and(prim_comp_region, mask)
		
		return 	mask, prim_comp_mask	
	else:
		return mask

#########################################################################################################################################

def mask_with_prim_comp_flags_median(array_, flags_info, thres_, n_chan_, prim_comp_ = True):
	mask = []
	#prim_comp_mask = []
	for i in np.arange(0,array_.shape[1], n_chan_):
		support = np.zeros(array_.shape[1],dtype=bool)
		support[i: i+ n_chan_] = True
		mad_stat = np.median(abs(array_[:,flags_info*support] - np.median(array_[:,flags_info*support], (0,1))[None, None, :] ), (0,1))
		mask_per_chan = array_[:,flags_info*support].sum((0,1))/(1.4826* mad_stat * np.sqrt(array_[:,flags_info*support].shape[0]*array_[:,flags_info*support].shape[1]) ) > thres_
		mask.append(mask_per_chan)

	mask = filter_clean_isolated_cells(mask)	# filter the isolated points
	if prim_comp_:
		ext_mask = ndimage.binary_fill_holes(mask, structure=np.array([[0,1,0]*3]).reshape(3,3))	# connects the mask vertically
		id_regions, num_ids = ndimage.label(ext_mask, structure=np.ones((3,3)) )	# id_regions, num_ids: contains different frequency dependent component info
		prim_comp_bool_array = np.array([np.any(np.where(id_regions == i)[-1] == id_regions.shape[-1]//2) for i in range(num_ids + 1) ])	# checks for the component intersecting the mid-point (horizontally speaking)(only if the main component is shifted to central phase bins) of the mask 
		prim_comp_bool_array[0] = False
		#prim_comp_region = filter_max_cells(id_regions)
		prim_comp_region = prim_comp_bool_array[id_regions]
		prim_comp_mask = np.logical_and(prim_comp_region, mask)
		
		return 	mask, prim_comp_mask	
	else:
		return mask

########################################################################################################################################

# arr_3d should be baseline subtracted data

def off_region_array_from_baseline_s_data(arr_3d, tol = 0.05):
	
	std_arr = arr_3d.std(axis=(0,1))
	mask = np.zeros_like(std_arr, dtype = bool)
	
	
	mask[np.argmin(std_arr)] = True
	std_arr[mask] = std_arr.max()
	
	mean_arr = [abs(arr_3d[:, :, mask].mean(axis=-1)).max()]
	
	count = 0
	while mean_arr[-1] > tol and count < arr_3d.shape[-1]:
		mask[np.argmin(std_arr)] = True
		std_arr[mask] = std_arr.max()
		mean_arr.append(abs(arr_3d[:, :, mask].mean(axis=-1)).max())
		count +=1
	return mask
########################################################################################################################################

# arr_3d should be actual data (without baseline subtraction)

def off_region_array_from_actual_data(arr_3d):
	
	box_car_len = int(np.round(0.15*arr_3d.shape[-1]))
	start_b = np.convolve(arr_3d.mean((0,1)),np.ones(box_car_len),mode='valid').argmin() -1
	mask = np.zeros(arr_3d.shape[-1],dtype=bool)
	if len(mask[start_b : start_b + box_car_len]):
		mask[start_b : start_b + box_car_len] = True
	else:
		start_b = np.convolve(np.roll(arr_3d.mean((0,1)), box_car_len ),np.ones(box_car_len),mode='valid').argmin() -1	
		mask = np.zeros(arr_3d.shape[-1],dtype=bool)
		mask[start_b : start_b + box_car_len] = True
		mask = np.roll(mask, -box_car_len)
	return mask




#############################################################################################################################

def bandpass_g_by_tsys(freq_arr):

	a0 = np.array([-0.0274432341259116, -3.94269512191062, -60.9659547181797, -57.790674973176])
	a1 = 1e-4 * np.array([6.53316475755705, 609.220731323399, 5298.1146165773, 2831.83485351112])
	a2 = 1e-6 * np.array([-5.75221466249264, -388.278427948319, -1911.99335049503, -576.846763506177])
	a3 = 1e-8 * np.array([2.26066919981535, 130.788311514484, 366.739380599885, 62.5315209850824])
	a4 = 1e-11 * np.array([-3.30079139610497, -245.688427095004, -394.286904604421, -38.047941517696])
	a5 = 1e-13 * np.array([0, 24.4226778956594, 22.5267804015136, 1.23211866985187])
	a6 = 1e-17 * np.array([0, -100.460621956708, -53.4321835013018, -1.65909077237512])
	

	coeff = np.array([a0, a1, a2, a3, a4, a5, a6])

	if isinstance(freq_arr, float) or isinstance(freq_arr, int):
		if freq_arr > 125 and freq_arr < 250:
			band = 2
			#Tsys = 253.846
			g_ant_ = 0.33
		elif freq_arr > 250 and freq_arr < 500:
			band = 3
			#Tsys =  84.615
			g_ant_ = 0.33
		elif freq_arr > 550 and freq_arr < 850:
			band = 4
			#Tsys = 87.071
			g_ant_ = 0.33
		elif freq_arr > 980 and freq_arr < 1500:
			band = 5
			#Tsys = 88.889
			g_ant_ = 0.32
		return np.sum([ coeff[i, band-2] * freq_arr**i for i in range(coeff.shape[0])]), g_ant_

	else:
		y = []
		if freq_arr.mean() > 125 and freq_arr.mean() < 250:
			band = 2
			#Tsys = 253.846
			g_ant_ = 0.33
		elif freq_arr.mean() > 250 and freq_arr.mean() < 500:
			band = 3
			#Tsys =  84.615
			g_ant_ = 0.33
		elif freq_arr.mean() > 550 and freq_arr.mean() < 850:
			band = 4
			#Tsys = 87.071
			g_ant_ = 0.33
		elif freq_arr.mean() > 980 and freq_arr.mean() < 1500:
			band = 5
			#Tsys = 88.889
			g_ant_ = 0.32

		for nu in freq_arr:
			y.append(np.sum([ coeff[i, band-2] * nu**i for i in range(coeff.shape[0])]))
	
		return np.array(y), g_ant_

#####################################################################################################################



def sefd(nu_, nu_c, time_s_, Tsky_, n_ant_ = 23, pa_beam_ =True, delta_chan_ = 1, n_pol_ = 2):
	Tsky_ = Tsky_ * (nu_c/nu_)**2.55
	Tdef = 22 * (408/nu_)**2.55
	g_by_Tsys, g_ant = bandpass_g_by_tsys(nu_)
	ant_pow_ind = int(pa_beam_ +1)
	return ( (1/g_by_Tsys) - (Tdef/g_ant) + (Tsky_/g_ant) ) /np.sqrt( (n_ant_**ant_pow_ind) * delta_chan_ *1e6 * time_s_ * n_pol_)
	


def initial_guess_n_bounds_func(n_, x_, y_):
	# n_ is the number of broken power law (1,2,3,.....)
	# x, y = log(freq), log(Flux_density)
	
	#x_chunks = [np.array([x_val for x_val in x_ if x_val >= breaking_points[i] and x_val < breaking_points[i+1]]) for i in range(len(breaking_points)-1)]

	while 2*n_ > len(x_):
		n_ -= 1
	
	if len(x_[:: len(x_)//n_]) == n_ + 1:
		breaking_points = np.append( x_[:: len(x_)//n_][:-1], x_[-1])
	elif len(x_[:: len(x_)//n_]) == n_:
		breaking_points = np.append( x_[:: len(x_)//n_], x_[-1])
	'''
	if len(x_)//n_ >1:
		breaking_points = x_[::len(x_)//n_]
		print('path 1')
	else: 
		n_ = 1
		breaking_points = np.linspace(x_[0], x_[-1], n_ + 1)
		print('path 2')
	'''
	init_guess = list(breaking_points[1:-1])
	

	for i in range(n_):
		
		param_init, _ = curve_fit(lambda x_vals_, *params_0_: fit_func_st_line(x_vals_, 1, *params_0_), x_[np.logical_and(breaking_points[i]<=x_, x_<breaking_points[i+1])],y_[np.logical_and(breaking_points[i]<=x_, x_<breaking_points[i+1])], p0=[-1,1], bounds=[[-np.inf,-np.inf], [np.inf, np.inf]], maxfev = 5000)
		#slope = 
		init_guess.append(param_init[0])
	init_guess.append(y_.mean())
	#############################################################################
	lower_bounds = [x_.min()]*(n_ - 1) + [-np.inf]*(n_ + 1)
	return init_guess, [[x_.min()]*(n_ - 1) + [-np.inf]*(n_ + 1), [x_.max()]*(n_ - 1) + [np.inf]*(n_ + 1)]

def bkn_st_lines_func(x_, breaks, alphas, const = 1):
	
	try: 
		ind = np.where(np.asarray(breaks) >= x_)[0][0]
		if ind == 0:
			y_ = x_*alphas[ind]
			 #- (xchunk[0]*alphas[ind])
			y_ += const
		else:
			
			for i in range(ind):
				const += (breaks[i]*(alphas[i] - alphas[i + 1]))
			y_ = const + (x_*alphas[ind])
	except:
		for i in range(len(breaks)):
			const += (breaks[i]*(alphas[i] - alphas[i + 1]))
		y_ = const + (x_*alphas[-1])

	return y_
	
def fit_func_st_line(x_, n_, *args):
	# n_ is the number of broken power law
	breaks = list(args[: n_ - 1])
	alphas = list(args[n_ - 1 : -1])
	const = args[-1]
	#print('breaks : ', breaks,' alphas : ', alphas,' const : ',const)
	try:
		len(x_)
		return np.array([bkn_st_lines_func(i, breaks, alphas, const) for i in x_])
	except:
		
		return bkn_st_lines_func(x_, breaks, alphas, const)	

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







##############################################################		optimal number of sub-bands

def band_number(Tsky, thr, **file_data):
	############# This return the subbands needed which yeilds maximum number of sub-bands greater than threshold
	
	if len(file_data.keys()) >2:
		data_, freq_, mjds_ = file_data['DATA'], file_data['FREQ'], file_data['MJD']
		
	else:
		file_ = file_data['FILE']
		data_, freq_, mjds_= pfd_data(file_)
		data_ = data_[:,0]

	
	unflag_freq = np.sum(data_, axis = (0,2)) != 0
	freq = freq_[unflag_freq]
	g_by_tsys = sefd(freq, freq.mean(), 1, Tsky)
	data_unflaged = data_[:,unflag_freq]
	data_unflaged = data_unflaged/g_by_tsys[np.newaxis, :, np.newaxis]

	snr_arr = []
	weight_on = []
	for no in range(data_.shape[1]-1):
		n_chan = no + 1
		
		
		fchunched_data = np.array([np.mean(data_unflaged[:,i : i + n_chan], axis = 1) for i in np.arange(0,data_unflaged.shape[1], n_chan) ]).transpose(1,0,2)
		profile_freq = fchunched_data.mean(0)

		mask = []
		for i in range(0,data_unflaged.shape[1], n_chan):
			mask.append(data_unflaged[:,i : i + n_chan].sum((0,1))/(data_unflaged[:,i : i + n_chan].std((0,1)) * np.sqrt(data_unflaged.shape[0]*data_unflaged[:,i : i + n_chan].shape[1]) ) > thr)
		on_weight_arr = np.array(mask)
		
		unflagged_flux_freq = np.array([ (profile_freq[i] - profile_freq[i, ~on_weight_arr[i]].mean() ).sum()   for i in range(profile_freq.shape[0]) ])
		unflagged_rms_freq = np.array([ profile_freq[i, ~on_weight_arr[i]].std() for i in range(profile_freq.shape[0]) ])
		snr = unflagged_flux_freq/(unflagged_rms_freq * np.sqrt( on_weight_arr.sum(1) ) )
		snr_arr.append(np.nan_to_num(snr, nan=0, neginf=0, posinf=0))
		weight_on.append(on_weight_arr)
	return snr_arr, weight_on

SNR_arr, W = band_number(26, 3, DATA=data, FREQ=F, MJD=mjd)

for ind, val  in enumerate(SNR_arr):
	snr_val = np.nan_to_num(val, nan=0, neginf=0, posinf=0) >5
	print(ind + 1, snr_val.sum(), len(val), W[ind].shape)


#####################################################################################################################

#####################################		Main Function


#	spectra_with_gain is the main function which uses above function above to create a
#	spectra (with gain correction, uses polynomial from etc calculator) for a data cube. 
#	It crunches full domain info and n_chan frequency channels to increase the SNR
# 	in each n_chan frequency range.



#def spectra_with_gain(file_, Tsky, n_chan = 10, sigma = 10, thres = 3, psrchive_method = False, show_plots = True, allow_fit = False):

def spectra_with_gain(Tsky, n_chan = 10, sigma = 10, thres = 3, Bandpass = True, primary_comp = True, show_plots = True, allow_fit = False, save_plot = False, n_ant = 23, **file_data):
	
	
	if len(file_data.keys()) >2:
		data_, freq_, mjds_ = file_data['DATA'], file_data['FREQ'], file_data['MJD']
		pa_beam = True

	else:
		file_ = file_data['FILE']
		pa_beam = not 'ia' in [t for s in file_.split('_') for t in s.split('.')]
		
		if Bandpass:
			data_, freq_, mjds_= pfd_data(file_)
			
		else:
			data_, freq_, mjds_= pfd_data_b(file_)
		data_ = data_[:,0]

	psrchive_method = False
	df = freq_[1] - freq_[0]
	delta_chan = 1
	
	#n_ant = 23	
	
	#on_weight = on_bool_array(data_.mean(axis=1), thres)
	
	unflag_freq = np.sum(data_, axis = (0,2)) != 0		# for removing fully flagged channels from the data_
	unflag_time = np.sum(data_, axis = (1,2)) != 0
	
	if not(isinstance(n_chan, int)):
		n_chan = unflag_freq.sum()
	
	freq = freq_[unflag_freq]
	#g_by_tsys = bandpass_g_by_tsys(freq)[0]
	data_unflaged = data_[unflag_time][:,unflag_freq]

	
	if Bandpass:
		g_by_tsys = sefd(freq, freq.mean(), 1, Tsky, pa_beam)
		#data_unflaged = data_unflaged/g_by_tsys[np.newaxis, :, np.newaxis]
		#off_bool_array = off_region_array_from_baseline_s_data(data_unflaged)
		#print('etc bandpass polynomial correction')
		
	else:
		#choose this option iff the data_ is NOT baseline subtracted
		off_bins = off_region_array_from_actual_data(data_unflaged)
		
		#g_by_tsys = gf(sv(data_unflaged[:,:,off_bins].mean((0,-1)), 5, 1),1)
		data_unflaged = data_unflaged/data_unflaged[:,:,off_bins].mean(-1)[:,:,None] - 1
		

	if psrchive_method:
		off_bool_array = off_region_array_from_baseline_s_data(data_unflaged)
	
	
	
	# fchunched_data = np.array([np.mean(data_unflaged[:,i : i + n_chan], axis = 1) for i in np.arange(0,data_unflaged.shape[1], n_chan) ])
	fchunched_data = np.array([np.mean(data_unflaged[:,i : i + n_chan], axis = 1) for i in np.arange(0,data_unflaged.shape[1], n_chan) ]).transpose(1,0,2)
	
	if not(psrchive_method):
		
		if primary_comp:
			on_weight_arr_f, on_weight_arr_p = mask_with_prim_comp(data_unflaged, thres, n_chan, prim_comp_ = True)
			# on_weight_arr_ = filter_prim_comp(np.array(mask))
			#on_weight_arr_rms = filter_clean_isolated_cells(np.array(mask))
		else:
			on_weight_arr_f = on_weight_arr_p = mask_with_prim_comp(data_unflaged, thres, n_chan, prim_comp_ = False)
			#on_weight_arr_ = filter_clean_isolated_cells(np.array(mask))
			
		
	#if not(psrchive_method):
	#	on_weight_arr_ = fchunched_data.sum(axis=1)/(fchunched_data.std(axis=1) * np.sqrt(fchunched_data.shape[1])) > thres		
		#weight_arr = np.array([on_bool_array(fchunched_data[i], thres) for i in range(fchunched_data.shape[0])])
	

	profile = data_unflaged.mean(axis=(0,1))
	# profile_freq = fchunched_data.mean(axis=1)#/gain_f[:, np.newaxis]
	profile_freq = fchunched_data.mean(axis=0)
	
	unflagged_fcrunch10_freq, chan_arr = [], []
	for i in np.arange(0, freq.shape[0], n_chan):
		chan_arr.append(freq[i : i + n_chan])
		unflagged_fcrunch10_freq.append(np.median(freq[i : i + n_chan]))
	
	unflagged_fcrunch10_freq = np.array(unflagged_fcrunch10_freq)
	chan_arr = np.array(chan_arr, dtype=object)
	time_s = (mjds_[-1] - mjds_[0] +  np.all(np.diff(mjds_,2) < abs(np.diff(mjds_)).min()*1e-3)*np.diff(mjds_)[0])*86400
	
	if psrchive_method:

		unflagged_flux_freq = profile_freq.sum(axis=-1)
		unflagged_rms_freq = profile_freq[:, off_bool_array].std(axis=-1)
		snr = unflagged_flux_freq/(unflagged_rms_freq * np.sqrt( (~off_bool_array).sum()) )

		noise = []
		for nu_arr in chan_arr:
			if len(nu_arr) == 1:
				noise.append((np.sqrt( (sefd(nu_arr, np.median(nu_arr), time_s, Tsky, n_ant,pa_beam)**2).sum() )/nu_arr * np.sqrt((~off_bool_array).sum()/(off_bool_array.sum())) )[0])
			else:
				noise.append(np.sqrt( (sefd(np.arange(nu_arr[0], nu_arr[-1],delta_chan), np.median(nu_arr), time_s, Tsky, n_ant, pa_beam)**2).sum() )/np.arange(nu_arr[0], nu_arr[-1],delta_chan).shape[0] * np.sqrt((~off_bool_array).sum()/(off_bool_array.sum())) )
	#unflagged_mean_flux_freq = unflagged_flux_freq/unflagged_fcrunch10.shape[1]
		noise = np.array(noise)
	else:
	
		# unflagged_flux_freq = np.array([ profile_freq[i, on_weight_arr_p[i]].sum() for i in range(profile_freq.shape[0]) ])
		unflagged_flux_freq = np.array([ profile_freq[i, on_weight_arr_p[i]].mean() for i in range(profile_freq.shape[0]) ])
		#unflagged_flux_freq = np.array([ (profile_freq[i] - profile_freq[i, ~on_weight_arr_f[i]].mean() ).sum()   for i in range(profile_freq.shape[0]) ])
		
		unflagged_rms_freq = np.array([ profile_freq[i, ~on_weight_arr_f[i]].std() for i in range(profile_freq.shape[0]) ])
		#unflagged_rms_freq = profile_freq[:, off_bool_array].std(axis=-1)
		
		#snr = unflagged_flux_freq/(unflagged_rms_freq * np.sqrt( on_weight_arr_p.sum(1) ) )
		snr = unflagged_flux_freq * np.sqrt( on_weight_arr_p.sum(axis=1))/unflagged_rms_freq
		# snr = unflagged_flux_freq * np.sqrt( np.median(on_weight_arr_p.sum(axis=1)))/unflagged_rms_freq 
		
		noise = []
		for f_ind, nu_arr in enumerate(chan_arr):
			if len(nu_arr) == 1:
				# noise.append((np.sqrt( (sefd(nu_arr + 0.5*df, np.median(nu_arr), time_s, Tsky, n_ant, pa_beam)**2).sum() ) * np.sqrt( on_weight_arr_[f_ind].sum()/(~on_weight_arr_[f_ind]).sum() ) )[0])
				nu_sub_arr = np.arange(chan_arr[f_ind - 1][-1] - 0.5*df, nu_arr[-1] + 0.5*df, delta_chan)
				width_term = np.sqrt( on_weight_arr_p[f_ind].sum()/(~on_weight_arr_p[f_ind]).sum() )
				
				noise.append(np.sqrt( (sefd(nu_sub_arr, np.median(nu_arr), time_s, Tsky, n_ant, pa_beam)**2).sum() )/nu_sub_arr.shape[0] * width_term )
			else:
				# noise.append(np.sqrt( (sefd(np.arange(nu_arr[0], nu_arr[-1],delta_chan), np.median(nu_arr), time_s, Tsky, n_ant, pa_beam)**2).sum() )/np.arange(nu_arr[0], nu_arr[-1],delta_chan).shape[0] * np.sqrt( on_weight_arr_p[f_ind].sum()/(~on_weight_arr_p[f_ind]).sum() ) )
				nu_sub_arr = np.arange(nu_arr[0] - 0.5*df, nu_arr[-1] + 0.5*df, delta_chan)
				width_term = np.sqrt( on_weight_arr_p[f_ind].sum()/(~on_weight_arr_p[f_ind]).sum() )
				noise.append(np.sqrt( (sefd(nu_sub_arr, np.median(nu_arr), time_s, Tsky, n_ant, pa_beam)**2).sum() )/nu_sub_arr.shape[0] * width_term )
		noise = np.array(noise)
	
	
	snr3_bool = np.ones_like(snr, dtype = bool)
	snr3_bool[np.nan_to_num(snr, nan=0,posinf=0.0, neginf=0.0) < sigma] = False	
	
	flux = (snr * noise)[snr3_bool] 
	error = np.sqrt(noise[snr3_bool]**2 + (0.1 * flux)**2)
	
	log_freq, log_flux = np.log10(unflagged_fcrunch10_freq[snr3_bool]), np.log10(flux)
	#fit_possible = False
	if allow_fit:
		AICC = []
		param_arr = []
		try:
			#fit_line = lambda x, m, c: (m*x) +c
			for n_pow_law_ind in range(1,int(len(log_freq)/2)):
				#n_pow_law_ind = 1
				params_0, bounds_0 = initial_guess_n_bounds_func( n_pow_law_ind, log_freq, log_flux)
				param_, _ = curve_fit(lambda nu_,*param_0: fit_func_st_line(nu_, n_pow_law_ind, *param_0), log_freq, log_flux, p0 = params_0, bounds = bounds_0, maxfev = 5000)
				Chi2 = ( ((log_flux -  fit_func_st_line(log_freq, n_pow_law_ind, *param_))/(error/(np.log(10)*flux)))**2 ).sum()
				aicc_ = Chi2 + 2*len(param_)*len(log_freq)/(len(log_freq) - len(param_) - 1)
				AICC.append(aicc_)
				param_arr.append(param_)
				del params_0, bounds_0, Chi2, param_, _, aicc_
				n_pow_law = int(np.argmin(AICC) + 1)
				opt_param = param_arr[np.argmin(AICC)]
			
			#fit_possible = True
		except:
			pass#fit_possible = False
	
	####################################################################
	if bool(save_plot + show_plots) :
		fig = plt.figure(figsize=(15,10))
		label_size = 15
		N_div = 10
		N_tick = len(log_freq)//N_div
		alpha0 = 1
		alpha1 = 0.2
		subp1 = fig.add_subplot(2, 3, (1,4))
		#subp1.imshow(data_unflaged.mean(axis=0))
		#subp1.set_yticks(np.arange(0,freq.shape[0],N_tick* n_chan), [int(freq[i]) for i in np.arange(0,freq.shape[0],N_tick* n_chan)])
		#subp1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
		ax1 = subp1.pcolormesh(np.arange(data_unflaged.shape[-1]), freq, data_unflaged.mean(0)/data_unflaged.mean())
		plt.colorbar(ax1, ax=subp1)
		for i in freq[::n_chan]: subp1.axhline(i, color='white',alpha=alpha0,linestyle='--')
		subp1.set_ylabel('Frequency', fontsize = label_size)
		subp1.set_xlabel('Phase bins', fontsize = label_size)
		subp1.tick_params(axis ='both', labelsize = label_size)
		#fit_exp = lambda x, s, A: A*(x**s)
		#x, _ = curve_fit(fit_exp, unflagged_fcrunch10_freq[snr3_bool], unflagged_flux_freq[snr3_bool])
		
		############################################################################
		subp2 = fig.add_subplot(2, 3, 2)
		if not(psrchive_method):
			#subp2.imshow(on_weight_arr_p, aspect='auto')
			#subp2.set_yticks(np.arange(0,on_weight_arr_p.shape[0],N_tick), [int(freq[i]) for i in np.arange(0,freq.shape[0],N_tick* n_chan)])
			#subp2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
			subp2.pcolormesh(np.arange(on_weight_arr_p.shape[-1]), unflagged_fcrunch10_freq, on_weight_arr_p )
			for i in freq[::n_chan]: subp2.axhline(i, color='white',alpha=alpha1,linestyle='--')
		else:
			#subp2.imshow(off_bool_array[np.newaxis, :]*np.ones_like(fchunched_data.mean(axis=0)), aspect='auto')
			#subp2.set_yticks(np.arange(0,fchunched_data.mean(axis=0).shape[0],N_tick), [int(freq[i]) for i in np.arange(0,freq.shape[0],N_tick* n_chan)])
			#subp2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
			subp2.pcolormesh(np.arange(data_unflaged.shape[-1]), freq, off_bool_array[np.newaxis, :]*np.ones_like(fchunched_data.mean(axis=0)) )
			for i in freq[::n_chan]: subp1.axhline(i)
		subp2.tick_params(axis ='both', labelsize = label_size)
		###############################################################################
		subp3 = fig.add_subplot(2, 3, 5)
		#subp3.imshow(fchunched_data.mean(axis=0), aspect='auto')
		subp3.pcolormesh(np.arange(data_unflaged.shape[-1]), unflagged_fcrunch10_freq, fchunched_data.mean(axis=0))
		for i in freq[::n_chan]: subp3.axhline(i, color='white',alpha=alpha1,linestyle='--')
		'''
		try:
			subp3.set_yticks(np.arange(0,on_weight_arr_p.shape[0],N_tick), [int(freq[i]) for i in np.arange(0,freq.shape[0],N_tick*  n_chan)])
		except:
			pass
		'''
		#subp3.yaxis.set_minor_locator(ticker.AutoMinorLocator())
		subp3.tick_params(axis ='both', labelsize = label_size)
		################################################################
		subp4 = fig.add_subplot(2, 3, 3)
		#subp4.plot(log_freq, log_flux,'.')
		subp4.errorbar(10**log_freq, flux, yerr = error, fmt = '.')
		if allow_fit:
			try:
				n_pow_law = int(np.argmin(AICC) + 1)
				opt_param = param_arr[np.argmin(AICC)]
				subp4.plot(10**np.linspace(log_freq[0], log_freq[-1], 1000), 10**fit_func_st_line(np.linspace(log_freq[0], log_freq[-1], 1000), n_pow_law, *opt_param) )
				#subp4.plot(10**log_freq, [10**fit_line(i, x[0], x[1]) for i in log_freq] )
				#subp4.plot(unflagged_fcrunch10_freq[snr3_bool], [fit_exp(i, x[0], x[1]) for i in unflagged_fcrunch10_freq[snr3_bool]])
				for i in range(n_pow_law): print("spectras :", opt_param[n_pow_law - 1: -1][i])
				subp4.set_title('Spectal Index:' + str(np.around(opt_param[n_pow_law - 1: -1],3)) )
			except:
				pass
		subp4.loglog()
		#subp4.minorticks_off()
		#subp4.tick_params(axis='x', which='minor', bottom = False, labelbottom = False)
		#subp4.set_yticks(flux, np.round(flux, 5))
		#subp4.set_xticks(10**log_freq[::N_tick], np.rint(10**log_freq).astype(int)[::N_tick],rotation=25)
		subp4.set_xticks(freq[::len(freq)//N_div + 1], freq.astype(int)[::len(freq)//N_div + 1],rotation=25)
		subp4.xaxis.set_minor_locator(ticker.AutoMinorLocator())
		#subp4.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
		subp4.tick_params(axis ='both', labelsize = label_size)
		####################################################################################
		#plt.
		
		subp5 = fig.add_subplot(2, 3, 6)
		subp5.plot(profile/profile.max())
		try:
			subp5.plot(np.arange(data_unflaged.shape[-1])[on_weight_arr[0]], (profile/profile.max())[on_weight_arr[0]], 'o')
		except:
			pass
		#subp5.plot(data_.std(axis=(0,1))/data_.std(axis=(0,1)).max())
		subp5.set_ylabel('Normalized amplitude', fontsize = label_size )
		subp5.set_xlabel('Phase bins', fontsize = label_size )
		#subp5.set_title('Total flux : ' + str(round(Total_flux,3)) + 'snr : ' + str(round(SNR, 3)) )
		
		try: 
			subp5.set_title('Total flux : ' + str(round(Total_flux[0],3)) + '(mJy); snr : ' + str(round(SNR[0], 3)), fontsize = label_size )
		except:
			pass
		subp5.tick_params(axis ='both', labelsize = label_size)
		try:
			epoch = Time(mjds_[0], format='mjd').ymdhms
			plt.suptitle('Epoch : ' + str(epoch[2]) + '/' + str(epoch[1]) + '/' + str(epoch[0])  + '; Time : ' + str(round(time_s,2)) + r'$(s) ; n_{chan} : $' + str(n_chan) + '; sigma : ' + str(sigma) + '; thres : ' + str(thres), fontsize = label_size)
			#plt.suptitle('File Name : ' + str(file_[:-16]) + '; Time : ' + str(round(time_s,2)) + '(s); n_chan : ' + str(n_chan) + '; sigma : ' + str(sigma) + '; thres : ' + str(thres), fontsize = label_size)
		except:
			pass
		plt.tight_layout()
		
		if save_plot:
			file_name_ = str('./plot_files/') + file_[:-4] + str('.png')
			plt.savefig(str(file_name_), dpi=200)
			
		if show_plots:
			plt.show()
		else:
			plt.close()
	####################################################################
	if allow_fit:
		try:	
			len(opt_param)
			if not psrchive_method:
			
				return opt_param[n_pow_law - 1: -1], snr, flux, noise, error, np.rint(10**log_freq), on_weight_arr_p#, data_unflaged.shape
			else:
				return opt_param[n_pow_law - 1: -1], snr, flux, noise, error, np.rint(10**log_freq), off_bool_array#, data_unflaged.shape
		except:
			if not psrchive_method:	
				return snr, flux, noise, error, np.rint(10**log_freq), on_weight_arr_p#, data_unflaged.shape
			else:
				return snr, flux, noise, error, np.rint(10**log_freq), off_bool_array#, data_unflaged.shape
			
	else:
		if not psrchive_method:	
			return snr, flux, noise, error, np.rint(10**log_freq), on_weight_arr_p#, data_unflaged.shape
		else:
			return snr, flux, noise, error, np.rint(10**log_freq), off_bool_array#, data_unflaged.shape



###########################		Main Function ends here

#####################################################################################################################


#####################################################################################################################

#####################################		Main Function with flags taken into account


#	spectra_with_gain is the main function which uses above function above to create a
#	spectra (with gain correction, uses polynomial from etc calculator) for a data cube. 
#	It crunches full domain info and n_chan frequency channels to increase the SNR
# 	in each n_chan frequency range.



#def spectra_with_gain(file_, Tsky, n_chan = 10, sigma = 10, thres = 3, psrchive_method = False, show_plots = True, allow_fit = False):

def spectra_with_gain_flags(Tsky, n_chan = 10, sigma = 10, thres = 3, primary_comp = True, show_plots = True, allow_fit = False, save_plot = False, n_ant = 23, **file_data):
	
	
	if len(file_data.keys()) >2:
		data_, freq_, mjds_ = file_data['DATA'], file_data['FREQ'], file_data['MJD']
		pa_beam = True

	else:
		file_ = file_data['FILE']
		pa_beam = not 'ia' in [t for s in file_.split('_') for t in s.split('.')]
		
		data_, freq_, mjds_= pfd_data(file_)
		data_ = data_[:,0]

	psrchive_method = False
	df = freq_[1] - freq_[0]
	delta_chan = 1
	
	#n_ant = 23	
	
	#on_weight = on_bool_array(data_.mean(axis=1), thres)
	
	unflag_freq = np.sum(data_, axis = (0,2)) != 0		# for removing fully flagged channels from the data_
	unflag_time = np.sum(data_, axis = (1,2)) != 0
	
	if not(isinstance(n_chan, int)):
		n_chan = unflag_freq.sum()
	
	freq = freq_[unflag_freq]
	# g_by_tsys = bandpass_g_by_tsys(freq)[0]
	data_unflaged = data_[unflag_time][:,unflag_freq]

	if primary_comp:
		on_weight_arr_f, on_weight_arr_p = mask_with_prim_comp_flags(data_, unflag_freq, thres, n_chan, prim_comp_ = True)
		# on_weight_arr_ = filter_prim_comp(np.array(mask))
		#on_weight_arr_rms = filter_clean_isolated_cells(np.array(mask))
	else:
		on_weight_arr_f = on_weight_arr_p = mask_with_prim_comp_flags(data_, unflag_freq, thres, n_chan, prim_comp_ = False)
		#on_weight_arr_ = filter_clean_isolated_cells(np.array(mask))
		
	time_s = (mjds_[-1] - mjds_[0] +  np.all(np.diff(mjds_,2) < abs(np.diff(mjds_)).min()*1e-3)*np.diff(mjds_)[0])*86400	
		
	fchrunched_data = []
	profile_freq = []
	fcrunch10_freq, chan_arr = [], []
	noise = []
	unflagged_flux_freq, unflagged_rms_freq, snr = [], [], []
	chunk_bool_arr = []
	for i in range(0, data_.shape[1], n_chan):
		support = np.zeros(data_.shape[1],dtype=bool)
		support[i: i+ n_chan] = True
		
		if not np.any(unflag_freq*support):
			chunk_bool_arr.append(False)
			profile_freq.append(np.zeros(data_.shape[-1]))
			fchrunched_data.append(np.zeros_like(data_[:,0]))
			noise.append(0)
			unflagged_flux_freq.append(0)
			unflagged_rms_freq.append(0)
			snr.append(0)
			fcrunch10_freq.append(np.mean(freq_[support]))
			continue
		chunk_bool_arr.append(True)
		chunk_data = data_[:, unflag_freq*support]
		freq_chunk = freq_[unflag_freq*support]
		
		fchrunched_data.append(np.nan_to_num(np.nanmean(chunk_data, axis =1),nan=0))
		profile_per_chan = np.nanmean(chunk_data, axis=(0,1))
		profile_freq.append(profile_per_chan)
		#chan_arr.append(freq_chunk)
		fcrunch10_freq.append(np.mean(freq_chunk))
		
		nu_sub_arr = np.arange(freq_chunk[0] - 0.5*df + 0.5, freq_chunk[-1] + 0.5*df, delta_chan)
		width_term = np.sqrt( on_weight_arr_p[i//n_chan].sum()/(~on_weight_arr_p[i//n_chan]).sum() )
		noise.append(np.sqrt( (sefd(nu_sub_arr, np.mean(nu_sub_arr), time_s, Tsky, n_ant, pa_beam)**2).sum() )/nu_sub_arr.shape[0] * width_term )		
		
		
		unflagged_flux_freq.append(np.nanmean(profile_per_chan[on_weight_arr_p[i//n_chan]]))
		unflagged_rms_freq.append(np.nanstd(profile_per_chan[~on_weight_arr_f[i//n_chan]]))
		snr.append(unflagged_flux_freq[-1] * np.sqrt( on_weight_arr_p[i//n_chan].sum())/unflagged_rms_freq[-1])

	fchrunched_data = np.array(fchrunched_data).transpose(1,0,2)
	profile_freq = np.array(profile_freq)
	fcrunch10_freq = np.nan_to_num(fcrunch10_freq)
	#chan_arr = np.array(chan_arr, dtype=object)
	unflagged_flux_freq = np.array(unflagged_flux_freq)
	unflagged_rms_freq = np.array(unflagged_rms_freq)
	noise = np.array(noise)
	snr = np.array(snr)
	chunk_bool_arr = np.array(chunk_bool_arr)
	# fchrunched_data = np.array([np.mean(data_unflaged[:,i : i + n_chan], axis = 1) for i in np.arange(0,data_unflaged.shape[1], n_chan) ])
	#fchrunched_data = np.array([np.mean(data_[:,i : i + n_chan], axis = 1) for i in np.arange(0,data_unflaged.shape[1], n_chan) ]).transpose(1,0,2)
	
	'''
	fcrunch10_freq, chan_arr = [], []
	for i in np.arange(0, freq.shape[0], n_chan):
		chan_arr.append(freq[i : i + n_chan])
		fcrunch10_freq.append(np.median(freq[i : i + n_chan]))
	
	fcrunch10_freq = np.array(fcrunch10_freq)
	chan_arr = np.array(chan_arr, dtype=object)
	time_s = (mjds_[-1] - mjds_[0] +  np.all(np.diff(mjds_,2) < abs(np.diff(mjds_)).min()*1e-3)*np.diff(mjds_)[0])*86400
	
	# unflagged_flux_freq = np.array([ profile_freq[i, on_weight_arr_p[i]].sum() for i in range(profile_freq.shape[0]) ])
	unflagged_flux_freq = np.array([ profile_freq[i, on_weight_arr_p[i]].mean() for i in range(profile_freq.shape[0]) ])
	#unflagged_flux_freq = np.array([ (profile_freq[i] - profile_freq[i, ~on_weight_arr_f[i]].mean() ).sum()   for i in range(profile_freq.shape[0]) ])
		
	unflagged_rms_freq = np.array([ profile_freq[i, ~on_weight_arr_f[i]].std() for i in range(profile_freq.shape[0]) ])
	#unflagged_rms_freq = profile_freq[:, off_bool_array].std(axis=-1)
	
	#snr = unflagged_flux_freq/(unflagged_rms_freq * np.sqrt( on_weight_arr_p.sum(1) ) )
	snr = unflagged_flux_freq * np.sqrt( on_weight_arr_p.sum(axis=1))/unflagged_rms_freq
	# snr = unflagged_flux_freq * np.sqrt( np.median(on_weight_arr_p.sum(axis=1)))/unflagged_rms_freq 
	
	noise = []
	for f_ind, nu_arr in enumerate(chan_arr):
		if len(nu_arr) == 1:
			# noise.append((np.sqrt( (sefd(nu_arr + 0.5*df, np.median(nu_arr), time_s, Tsky, n_ant, pa_beam)**2).sum() ) * np.sqrt( on_weight_arr_[f_ind].sum()/(~on_weight_arr_[f_ind]).sum() ) )[0])
			nu_sub_arr = np.arange(chan_arr[f_ind - 1][-1] - 0.5*df, nu_arr[-1] + 0.5*df, delta_chan)
			width_term = np.sqrt( on_weight_arr_p[f_ind].sum()/(~on_weight_arr_p[f_ind]).sum() )
			
			noise.append(np.sqrt( (sefd(nu_sub_arr, np.median(nu_arr), time_s, Tsky, n_ant, pa_beam)**2).sum() )/nu_sub_arr.shape[0] * width_term )
		else:
			# noise.append(np.sqrt( (sefd(np.arange(nu_arr[0], nu_arr[-1],delta_chan), np.median(nu_arr), time_s, Tsky, n_ant, pa_beam)**2).sum() )/np.arange(nu_arr[0], nu_arr[-1],delta_chan).shape[0] * np.sqrt( on_weight_arr_p[f_ind].sum()/(~on_weight_arr_p[f_ind]).sum() ) )
			nu_sub_arr = np.arange(nu_arr[0] - 0.5*df, nu_arr[-1] + 0.5*df, delta_chan)
			width_term = np.sqrt( on_weight_arr_p[f_ind].sum()/(~on_weight_arr_p[f_ind]).sum() )
			noise.append(np.sqrt( (sefd(nu_sub_arr, np.median(nu_arr), time_s, Tsky, n_ant, pa_beam)**2).sum() )/nu_sub_arr.shape[0] * width_term )
	noise = np.array(noise)
	'''
	
	profile = data_.mean(axis=(0,1))
	# profile_freq = fchrunched_data.mean(axis=1)#/gain_f[:, np.newaxis]
	#profile_freq = fchrunched_data.mean(axis=0)
	
	snr3_bool = np.nan_to_num(snr, nan=0,posinf=0.0, neginf=0.0) > sigma
	
	flux = (snr * noise)[snr3_bool]
	error = np.sqrt(noise[snr3_bool]**2 + (0.1 * flux)**2)
	
	log_freq, log_flux = np.log10(fcrunch10_freq[snr3_bool]), np.log10(flux)
	#fit_possible = False
	if allow_fit:
		AICC = []
		param_arr = []
		try:
			#fit_line = lambda x, m, c: (m*x) +c
			for n_pow_law_ind in range(1,int(len(log_freq)/2)):
				#n_pow_law_ind = 1
				params_0, bounds_0 = initial_guess_n_bounds_func( n_pow_law_ind, log_freq, log_flux)
				param_, _ = curve_fit(lambda nu_,*param_0: fit_func_st_line(nu_, n_pow_law_ind, *param_0), log_freq, log_flux, p0 = params_0, bounds = bounds_0, maxfev = 5000)
				Chi2 = ( ((log_flux -  fit_func_st_line(log_freq, n_pow_law_ind, *param_))/(error/(np.log(10)*flux)))**2 ).sum()
				aicc_ = Chi2 + 2*len(param_)*len(log_freq)/(len(log_freq) - len(param_) - 1)
				AICC.append(aicc_)
				param_arr.append(param_)
				del params_0, bounds_0, Chi2, param_, _, aicc_
				n_pow_law = int(np.argmin(AICC) + 1)
				opt_param = param_arr[np.argmin(AICC)]
			
			#fit_possible = True
		except:
			pass#fit_possible = False
	
	####################################################################
	if bool(save_plot + show_plots) :
		fig = plt.figure(figsize=(15,10))
		label_size = 15
		N_div = 10
		N_tick = len(log_freq)//N_div
		alpha0 = 1
		alpha1 = 0.2
		subp1 = fig.add_subplot(2, 3, (1,4))
		#subp1.imshow(data_unflaged.mean(axis=0))
		#subp1.set_yticks(np.arange(0,freq.shape[0],N_tick* n_chan), [int(freq[i]) for i in np.arange(0,freq.shape[0],N_tick* n_chan)])
		#subp1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
		ax1 = subp1.pcolormesh(np.arange(data_.shape[-1]), freq_, data_.mean(0)/data_.mean())
		plt.colorbar(ax1, ax=subp1)
		for i in freq[::n_chan]: subp1.axhline(i, color='white',alpha=alpha0,linestyle='--')
		subp1.set_ylabel('Frequency', fontsize = label_size)
		subp1.set_xlabel('Phase bins', fontsize = label_size)
		subp1.tick_params(axis ='both', labelsize = label_size)
		#fit_exp = lambda x, s, A: A*(x**s)
		#x, _ = curve_fit(fit_exp, fcrunch10_freq[snr3_bool], unflagged_flux_freq[snr3_bool])
		
		############################################################################
		subp2 = fig.add_subplot(2, 3, 2)
		#subp2.imshow(on_weight_arr_p, aspect='auto')
		#subp2.set_yticks(np.arange(0,on_weight_arr_p.shape[0],N_tick), [int(freq[i]) for i in np.arange(0,freq.shape[0],N_tick* n_chan)])
		#subp2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
		subp2.pcolormesh(np.arange(on_weight_arr_p.shape[-1]), fcrunch10_freq, on_weight_arr_p)#[chunk_bool_arr] )
		for i in freq[::n_chan]: subp2.axhline(i, color='white',alpha=alpha1,linestyle='--')
		
		subp2.tick_params(axis ='both', labelsize = label_size)
		###############################################################################
		subp3 = fig.add_subplot(2, 3, 5)
		#subp3.imshow(fchrunched_data.mean(axis=0), aspect='auto')
		subp3.pcolormesh(np.arange(data_unflaged.shape[-1]), fcrunch10_freq, fchrunched_data.mean(axis=0))
		for i in freq[::n_chan]: subp3.axhline(i, color='white',alpha=alpha1,linestyle='--')
		'''
		try:
			subp3.set_yticks(np.arange(0,on_weight_arr_p.shape[0],N_tick), [int(freq[i]) for i in np.arange(0,freq.shape[0],N_tick*  n_chan)])
		except:
			pass
		'''
		#subp3.yaxis.set_minor_locator(ticker.AutoMinorLocator())
		subp3.tick_params(axis ='both', labelsize = label_size)
		################################################################
		subp4 = fig.add_subplot(2, 3, 3)
		#subp4.plot(log_freq, log_flux,'.')
		subp4.errorbar(10**log_freq, flux, yerr = error, fmt = '.')
		if allow_fit:
			try:
				n_pow_law = int(np.argmin(AICC) + 1)
				opt_param = param_arr[np.argmin(AICC)]
				subp4.plot(10**np.linspace(log_freq[0], log_freq[-1], 1000), 10**fit_func_st_line(np.linspace(log_freq[0], log_freq[-1], 1000), n_pow_law, *opt_param) )
				#subp4.plot(10**log_freq, [10**fit_line(i, x[0], x[1]) for i in log_freq] )
				#subp4.plot(fcrunch10_freq[snr3_bool], [fit_exp(i, x[0], x[1]) for i in fcrunch10_freq[snr3_bool]])
				for i in range(n_pow_law): print("spectras :", opt_param[n_pow_law - 1: -1][i])
				subp4.set_title('Spectal Index:' + str(np.around(opt_param[n_pow_law - 1: -1],3)) )
			except:
				pass
		subp4.loglog()
		#subp4.minorticks_off()
		#subp4.tick_params(axis='x', which='minor', bottom = False, labelbottom = False)
		#subp4.set_yticks(flux, np.round(flux, 5))
		#subp4.set_xticks(10**log_freq[::N_tick], np.rint(10**log_freq).astype(int)[::N_tick],rotation=25)
		subp4.set_xticks(freq[::len(freq)//N_div + 1], freq.astype(int)[::len(freq)//N_div + 1],rotation=25)
		subp4.xaxis.set_minor_locator(ticker.AutoMinorLocator())
		#subp4.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
		subp4.tick_params(axis ='both', labelsize = label_size)
		####################################################################################
		#plt.
		
		subp5 = fig.add_subplot(2, 3, 6)
		subp5.plot(profile/profile.max())
		try:
			subp5.plot(np.arange(data_unflaged.shape[-1])[on_weight_arr[0]], (profile/profile.max())[on_weight_arr[0]], 'o')
		except:
			pass
		#subp5.plot(data_.std(axis=(0,1))/data_.std(axis=(0,1)).max())
		subp5.set_ylabel('Normalized amplitude', fontsize = label_size )
		subp5.set_xlabel('Phase bins', fontsize = label_size )
		#subp5.set_title('Total flux : ' + str(round(Total_flux,3)) + 'snr : ' + str(round(SNR, 3)) )
		
		try: 
			subp5.set_title('Total flux : ' + str(round(Total_flux[0],3)) + '(mJy); snr : ' + str(round(SNR[0], 3)), fontsize = label_size )
		except:
			pass
		subp5.tick_params(axis ='both', labelsize = label_size)
		try:
			epoch = Time(mjds_[0], format='mjd').ymdhms
			plt.suptitle('Epoch : ' + str(epoch[2]) + '/' + str(epoch[1]) + '/' + str(epoch[0])  + '; Time : ' + str(round(time_s,2)) + r'$(s) ; n_{chan} : $' + str(n_chan) + '; sigma : ' + str(sigma) + '; thres : ' + str(thres), fontsize = label_size)
			#plt.suptitle('File Name : ' + str(file_[:-16]) + '; Time : ' + str(round(time_s,2)) + '(s); n_chan : ' + str(n_chan) + '; sigma : ' + str(sigma) + '; thres : ' + str(thres), fontsize = label_size)
		except:
			pass
		plt.tight_layout()
		
		if save_plot:
			file_name_ = str('./plot_files/') + file_[:-4] + str('.png')
			plt.savefig(str(file_name_), dpi=200)
			
		if show_plots:
			plt.show()
		else:
			plt.close()
	####################################################################
	if allow_fit:
		try:	
			len(opt_param)
			return opt_param[n_pow_law - 1: -1], snr, flux, noise, error, np.rint(10**log_freq), on_weight_arr_p#, data_unflaged.shape
			
		except:
			return snr, flux, noise, error, np.rint(10**log_freq), on_weight_arr_p#, data_unflaged.shape
			
	else:
		return snr, flux, noise, error, np.rint(10**log_freq), on_weight_arr_p#, data_unflaged.shape
		

###########################		Main Function ends here

#####################################################################################################################
#####################################################################################################################

#####################################		Main Function with flags taken into account


#	spectra_with_gain is the main function which uses above function above to create a
#	spectra (with gain correction, uses polynomial from etc calculator) for a data cube. 
#	It crunches full domain info and n_chan frequency channels to increase the SNR
# 	in each n_chan frequency range.



#def spectra_with_gain(file_, Tsky, n_chan = 10, sigma = 10, thres = 3, psrchive_method = False, show_plots = True, allow_fit = False):

def spectra_with_gain_flags_median(Tsky, n_chan = 10, sigma = 10, thres = 3, beam_ = 'PA', primary_comp = True, show_plots = True, allow_fit = False, save_plot = False, n_ant = 23, **file_data):
	
	
	if len(file_data.keys()) >2:
		data_, freq_, mjds_ = file_data['DATA'], file_data['FREQ'], file_data['MJD']

	else:
		file_ = file_data['FILE']
		
		data_, freq_, mjds_= pfd_data(file_)
		data_ = data_[:,0]

	df = freq_[1] - freq_[0]
	delta_chan = 1
	
	#n_ant = 23	
	
	#on_weight = on_bool_array(data_.mean(axis=1), thres)
	
	unflag_freq = np.sum(data_, axis = (0,2)) != 0		# for removing fully flagged channels from the data_
	unflag_time = np.sum(data_, axis = (1,2)) != 0
	
	if not(isinstance(n_chan, int)):
		n_chan = unflag_freq.sum()
	
	freq = freq_[unflag_freq]
	# g_by_tsys = bandpass_g_by_tsys(freq)[0]
	data_unflaged = data_[unflag_time][:,unflag_freq]

	if primary_comp:
		on_weight_arr_f, on_weight_arr_p = mask_with_prim_comp_flags_median(data_, unflag_freq, thres, n_chan, prim_comp_ = True)
		# on_weight_arr_ = filter_prim_comp(np.array(mask))
		#on_weight_arr_rms = filter_clean_isolated_cells(np.array(mask))
	else:
		on_weight_arr_f = on_weight_arr_p = mask_with_prim_comp_flags_median(data_, unflag_freq, thres, n_chan, prim_comp_ = False)
		#on_weight_arr_ = filter_clean_isolated_cells(np.array(mask))
		
	time_s = (mjds_[-1] - mjds_[0] +  np.all(np.diff(mjds_,2) < abs(np.diff(mjds_)).min()*1e-3)*np.diff(mjds_)[0])*86400	
		
	fchrunched_data = []
	profile_freq = []
	fcrunch10_freq, chan_arr = [], []
	noise = []
	unflagged_flux_freq, unflagged_rms_freq, snr = [], [], []
	chunk_bool_arr = []
	for i in range(0, data_.shape[1], n_chan):
		support = np.zeros(data_.shape[1],dtype=bool)
		support[i: i+ n_chan] = True
		
		if not np.any(unflag_freq*support):
			chunk_bool_arr.append(False)
			profile_freq.append(np.zeros(data_.shape[-1]))
			fchrunched_data.append(np.zeros_like(data_[:,0]))
			noise.append(0)
			unflagged_flux_freq.append(0)
			unflagged_rms_freq.append(0)
			snr.append(0)
			fcrunch10_freq.append(np.mean(freq_[support]))
			continue
		chunk_bool_arr.append(True)
		chunk_data = data_[:, unflag_freq*support]
		freq_chunk = freq_[unflag_freq*support]
		
		fchrunched_data.append(np.nan_to_num(np.nanmean(chunk_data, axis =1),nan=0))
		profile_per_chan = np.nanmean(chunk_data, axis=(0,1))
		profile_freq.append(profile_per_chan)
		#chan_arr.append(freq_chunk)
		fcrunch10_freq.append(np.mean(freq_chunk))
		
		nu_sub_arr = np.arange(freq_chunk[0] - 0.5*df + 0.5, freq_chunk[-1] + 0.5*df, delta_chan)
		width_term = np.sqrt( on_weight_arr_p[i//n_chan].sum()/(~on_weight_arr_p[i//n_chan]).sum() )
		noise.append(np.sqrt( (sefd(nu_sub_arr, np.mean(nu_sub_arr), time_s, Tsky, n_ant, beam_)**2).sum() )/nu_sub_arr.shape[0] * width_term )		
		
		
		unflagged_flux_freq.append(np.nanmean(profile_per_chan[on_weight_arr_p[i//n_chan]]))
		unflagged_rms_freq.append(np.nanstd(profile_per_chan[~on_weight_arr_f[i//n_chan]]))
		snr.append(unflagged_flux_freq[-1] * np.sqrt( on_weight_arr_p[i//n_chan].sum())/unflagged_rms_freq[-1])

	fchrunched_data = np.array(fchrunched_data).transpose(1,0,2)
	profile_freq = np.array(profile_freq)
	fcrunch10_freq = np.nan_to_num(fcrunch10_freq)
	#chan_arr = np.array(chan_arr, dtype=object)
	unflagged_flux_freq = np.array(unflagged_flux_freq)
	unflagged_rms_freq = np.array(unflagged_rms_freq)
	noise = np.array(noise)
	snr = np.array(snr)
	chunk_bool_arr = np.array(chunk_bool_arr)
	# fchrunched_data = np.array([np.mean(data_unflaged[:,i : i + n_chan], axis = 1) for i in np.arange(0,data_unflaged.shape[1], n_chan) ])
	#fchrunched_data = np.array([np.mean(data_[:,i : i + n_chan], axis = 1) for i in np.arange(0,data_unflaged.shape[1], n_chan) ]).transpose(1,0,2)
	
	
	profile = data_.mean(axis=(0,1))
	# profile_freq = fchrunched_data.mean(axis=1)#/gain_f[:, np.newaxis]
	#profile_freq = fchrunched_data.mean(axis=0)
	
	snr3_bool = np.nan_to_num(snr, nan=0,posinf=0.0, neginf=0.0) > sigma
	
	flux = (snr * noise)[snr3_bool]
	error = np.sqrt(noise[snr3_bool]**2 + (0.1 * flux)**2)
	
	log_freq, log_flux = np.log10(fcrunch10_freq[snr3_bool]), np.log10(flux)
	#fit_possible = False
	if allow_fit:
		AICC = []
		param_arr = []
		try:
			#fit_line = lambda x, m, c: (m*x) +c
			for n_pow_law_ind in range(1,int(len(log_freq)/2)):
				#n_pow_law_ind = 1
				params_0, bounds_0 = initial_guess_n_bounds_func( n_pow_law_ind, log_freq, log_flux)
				param_, _ = curve_fit(lambda nu_,*param_0: fit_func_st_line(nu_, n_pow_law_ind, *param_0), log_freq, log_flux, p0 = params_0, bounds = bounds_0, maxfev = 5000)
				Chi2 = ( ((log_flux -  fit_func_st_line(log_freq, n_pow_law_ind, *param_))/(error/(np.log(10)*flux)))**2 ).sum()
				aicc_ = Chi2 + 2*len(param_)*len(log_freq)/(len(log_freq) - len(param_) - 1)
				AICC.append(aicc_)
				param_arr.append(param_)
				del params_0, bounds_0, Chi2, param_, _, aicc_
				n_pow_law = int(np.argmin(AICC) + 1)
				opt_param = param_arr[np.argmin(AICC)]
			
			#fit_possible = True
		except:
			pass#fit_possible = False
	
	####################################################################
	if bool(save_plot + show_plots) :
		fig = plt.figure(figsize=(15,10))
		label_size = 15
		N_div = 10
		N_tick = len(log_freq)//N_div
		alpha0 = 1
		alpha1 = 0.2
		subp1 = fig.add_subplot(2, 3, (1,4))
		#subp1.imshow(data_unflaged.mean(axis=0))
		#subp1.set_yticks(np.arange(0,freq.shape[0],N_tick* n_chan), [int(freq[i]) for i in np.arange(0,freq.shape[0],N_tick* n_chan)])
		#subp1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
		ax1 = subp1.pcolormesh(np.arange(data_.shape[-1]), freq_, data_.mean(0)/data_.mean())
		plt.colorbar(ax1, ax=subp1)
		for i in freq[::n_chan]: subp1.axhline(i, color='white',alpha=alpha0,linestyle='--')
		subp1.set_ylabel('Frequency', fontsize = label_size)
		subp1.set_xlabel('Phase bins', fontsize = label_size)
		subp1.tick_params(axis ='both', labelsize = label_size)
		#fit_exp = lambda x, s, A: A*(x**s)
		#x, _ = curve_fit(fit_exp, fcrunch10_freq[snr3_bool], unflagged_flux_freq[snr3_bool])
		
		############################################################################
		subp2 = fig.add_subplot(2, 3, 2)
		#subp2.imshow(on_weight_arr_p, aspect='auto')
		#subp2.set_yticks(np.arange(0,on_weight_arr_p.shape[0],N_tick), [int(freq[i]) for i in np.arange(0,freq.shape[0],N_tick* n_chan)])
		#subp2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
		subp2.pcolormesh(np.arange(on_weight_arr_p.shape[-1]), fcrunch10_freq, on_weight_arr_p)#[chunk_bool_arr] )
		for i in freq[::n_chan]: subp2.axhline(i, color='white',alpha=alpha1,linestyle='--')
		
		subp2.tick_params(axis ='both', labelsize = label_size)
		###############################################################################
		subp3 = fig.add_subplot(2, 3, 5)
		#subp3.imshow(fchrunched_data.mean(axis=0), aspect='auto')
		subp3.pcolormesh(np.arange(data_unflaged.shape[-1]), fcrunch10_freq, fchrunched_data.mean(axis=0))
		for i in freq[::n_chan]: subp3.axhline(i, color='white',alpha=alpha1,linestyle='--')
		'''
		try:
			subp3.set_yticks(np.arange(0,on_weight_arr_p.shape[0],N_tick), [int(freq[i]) for i in np.arange(0,freq.shape[0],N_tick*  n_chan)])
		except:
			pass
		'''
		#subp3.yaxis.set_minor_locator(ticker.AutoMinorLocator())
		subp3.tick_params(axis ='both', labelsize = label_size)
		################################################################
		subp4 = fig.add_subplot(2, 3, 3)
		#subp4.plot(log_freq, log_flux,'.')
		subp4.errorbar(10**log_freq, flux, yerr = error, fmt = '.')
		if allow_fit:
			try:
				n_pow_law = int(np.argmin(AICC) + 1)
				opt_param = param_arr[np.argmin(AICC)]
				subp4.plot(10**np.linspace(log_freq[0], log_freq[-1], 1000), 10**fit_func_st_line(np.linspace(log_freq[0], log_freq[-1], 1000), n_pow_law, *opt_param) )
				#subp4.plot(10**log_freq, [10**fit_line(i, x[0], x[1]) for i in log_freq] )
				#subp4.plot(fcrunch10_freq[snr3_bool], [fit_exp(i, x[0], x[1]) for i in fcrunch10_freq[snr3_bool]])
				for i in range(n_pow_law): print("spectras :", opt_param[n_pow_law - 1: -1][i])
				subp4.set_title('Spectal Index:' + str(np.around(opt_param[n_pow_law - 1: -1],3)) )
			except:
				pass
		subp4.loglog()
		#subp4.minorticks_off()
		#subp4.tick_params(axis='x', which='minor', bottom = False, labelbottom = False)
		#subp4.set_yticks(flux, np.round(flux, 5))
		#subp4.set_xticks(10**log_freq[::N_tick], np.rint(10**log_freq).astype(int)[::N_tick],rotation=25)
		subp4.set_xticks(freq[::len(freq)//N_div + 1], freq.astype(int)[::len(freq)//N_div + 1],rotation=25)
		subp4.xaxis.set_minor_locator(ticker.AutoMinorLocator())
		#subp4.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
		subp4.tick_params(axis ='both', labelsize = label_size)
		####################################################################################
		#plt.
		
		subp5 = fig.add_subplot(2, 3, 6)
		subp5.plot(profile/profile.max())
		try:
			subp5.plot(np.arange(data_unflaged.shape[-1])[on_weight_arr[0]], (profile/profile.max())[on_weight_arr[0]], 'o')
		except:
			pass
		#subp5.plot(data_.std(axis=(0,1))/data_.std(axis=(0,1)).max())
		subp5.set_ylabel('Normalized amplitude', fontsize = label_size )
		subp5.set_xlabel('Phase bins', fontsize = label_size )
		#subp5.set_title('Total flux : ' + str(round(Total_flux,3)) + 'snr : ' + str(round(SNR, 3)) )
		
		try: 
			subp5.set_title('Total flux : ' + str(round(Total_flux[0],3)) + '(mJy); snr : ' + str(round(SNR[0], 3)), fontsize = label_size )
		except:
			pass
		subp5.tick_params(axis ='both', labelsize = label_size)
		try:
			epoch = Time(mjds_[0], format='mjd').ymdhms
			plt.suptitle('Epoch : ' + str(epoch[2]) + '/' + str(epoch[1]) + '/' + str(epoch[0])  + '; Time : ' + str(round(time_s,2)) + r'$(s) ; n_{chan} : $' + str(n_chan) + '; sigma : ' + str(sigma) + '; thres : ' + str(thres), fontsize = label_size)
			#plt.suptitle('File Name : ' + str(file_[:-16]) + '; Time : ' + str(round(time_s,2)) + '(s); n_chan : ' + str(n_chan) + '; sigma : ' + str(sigma) + '; thres : ' + str(thres), fontsize = label_size)
		except:
			pass
		plt.tight_layout()
		
		if save_plot:
			file_name_ = str('./plot_files/') + file_[:-4] + str('.png')
			plt.savefig(str(file_name_), dpi=200)
			
		if show_plots:
			plt.show()
		else:
			plt.close()
	####################################################################
	if allow_fit:
		try:	
			len(opt_param)
			return opt_param[n_pow_law - 1: -1], snr, flux, noise, error, np.rint(10**log_freq), on_weight_arr_p#, data_unflaged.shape
			
		except:
			return snr, flux, noise, error, np.rint(10**log_freq), on_weight_arr_p#, data_unflaged.shape
			
	else:
		return snr, flux, noise, error, np.rint(10**log_freq), on_weight_arr_p#, data_unflaged.shape
		

###########################		Main Function ends here

#####################################################################################################################

#### This function gives the broken power law function
#### Inputs: array of x-axis (1d array)
####       : list of break point (n - 1 points, excluding the end points) ( within xvals.min() to xvals.max())
####       : list of alphas to be used in those broken power law components (n points)
#### Output: array of y axis corresponding to xvals

def bkn_pow(xvals, breaks, alphas, const = 1):
    	
	breakpoints = [np.min(xvals)*0.98] + breaks + [1.02*np.max(xvals)] # create a list of all the bounding x-values
	x_chunks = [np.array([x_ for x_ in xvals if x_ >= breakpoints[i] and x_ < breakpoints[i+1]]) for i in range(len(breakpoints)-1)]
    
	y_chunks = []
	for idx,xchunk in enumerate(x_chunks):
		yvals = xchunk**alphas[idx]
		y_chunks.append(yvals) # add this piece to the output
	c0 = 1
	for i in range(1,len(y_chunks)):
		# y_chunks[i] *= np.abs(y_chunks[i-1][-1]/(x_chunks[i - 1][-1]**alphas[i])) # scale the beginning of each piece to the end of the last so it is continuous
		c0 *= breaks[i - 1]**(alphas[i - 1] - alphas[i])
		y_chunks[i] *= c0

	return const * np.array([y for ychunk in y_chunks for y in ychunk])

def fit_bkn_pow(x_, n_, *args):
	breaks = list(args[: n_ - 1])
	alphas = list(args[n_ - 1 : -1])
	const = args[-1]
	return bkn_pow(x_, breaks, alphas, const)


def bkn_pow_func(x_, const_dc, breaks, alphas, const):
	# len(breaks) == len(alphas) - 1
	# breaks = param[: n]
	# alphas = param[n : -1]
	# const = param[-1]
	# Note: x_ shouldn't be a list or an array
	c0 = 1
	try: 
		ind = np.where(np.asarray(breaks) >= x_)[0][0]
		if ind == 0:
			y_ = x_**alphas[ind]
		else:
			
			for i in range(ind):
				c0 *= (breaks[i]**(alphas[i] - alphas[i + 1]))
			y_ = c0 * (x_**alphas[ind])
	except:
		for i in range(len(breaks)):
			c0 *= (breaks[i]**(alphas[i] - alphas[i + 1]))
		y_ = c0 * (x_**alphas[-1])

	return const * y_ + const_dc

def fit_func(x_, n_, *args):
	const_dc = args[0]
	breaks = list(args[1: n_])
	alphas = list(args[n_ : -1])
	const = args[-1]
	#print(const_dc ; ',const_dc,'breaks : ', breaks,' alphas : ', alphas,' const : ',const)
	try:
		len(x_)
		return np.array([bkn_pow_func(i, const_dc, breaks, alphas, const) for i in x_])
	except:
		
		return bkn_pow_func(x_, const_dc, breaks, alphas, const)	


#	return bkn_pow_func(x_, const_dc, breaks, alphas, const)


'''

	for ind, break_val in enumerate(breaks):
		if x < break_val:
			print(ind)
			if ind == 0:
				y = const * (x**alphas[ind])
				print(ind)
				break
			else:
				y = const * (breaks[ind - 1] ** (alphas[ind - 1] - alphas[ind])) * (x**alphas[ind])
				print(ind)
				break
		else:
			print(ind + 1000)
			y = const * (breaks[ind - 1] ** (alphas[ind - 1] - alphas[ind])) *  (x**alphas[-1])
			print(ind)
			break
	return y
'''
#####################################################################################################################################

def initial_guess_n_bounds_func(n_, x_, y_):
	# n_ is the number of broken power law (1,2,3,.....)
	# x, y = log(freq), log(Flux_density)
	
	#x_chunks = [np.array([x_val for x_val in x_ if x_val >= breaking_points[i] and x_val < breaking_points[i+1]]) for i in range(len(breaking_points)-1)]

	while 2*n_ > len(x_):
		n_ -= 1
	
	if len(x_[:: len(x_)//n_]) == n_ + 1:
		breaking_points = np.append( x_[:: len(x_)//n_][:-1], x_[-1])
	elif len(x_[:: len(x_)//n_]) == n_:
		breaking_points = np.append( x_[:: len(x_)//n_], x_[-1])
	'''
	if len(x_)//n_ >1:
		breaking_points = x_[::len(x_)//n_]
		print('path 1')
	else: 
		n_ = 1
		breaking_points = np.linspace(x_[0], x_[-1], n_ + 1)
		print('path 2')
	'''
	init_guess = list(breaking_points[1:-1])
	

	for i in range(n_):
		
		param_init, _ = curve_fit(lambda x_vals_, *params_0_: fit_func_st_line(x_vals_, 1, *params_0_), x_[np.logical_and(breaking_points[i]<=x_, x_<breaking_points[i+1])],y_[np.logical_and(breaking_points[i]<=x_, x_<breaking_points[i+1])], p0=[-1,1], bounds=[[-np.inf,-np.inf], [np.inf, np.inf]], maxfev = 5000)
		#slope = 
		init_guess.append(param_init[0])
	init_guess.append(y_.mean())
	#############################################################################
	lower_bounds = [x_.min()]*(n_ - 1) + [-np.inf]*(n_ + 1)
	return init_guess, [[x_.min()]*(n_ - 1) + [-np.inf]*(n_ + 1), [x_.max()]*(n_ - 1) + [np.inf]*(n_ + 1)]



#####################################################################################################################################


def bkn_st_lines_0(xvals, breaks, alphas, const = 1):
    	
	breakpoints = [np.min(xvals)*0.98] + breaks + [1.02*np.max(xvals)] # create a list of all the bounding x-values
	x_chunks = [np.array([x_ for x_ in xvals if x_ >= breakpoints[i] and x_ < breakpoints[i+1]]) for i in range(len(breakpoints)-1)]
	
	y_chunks = []
	for idx,xchunk in enumerate(x_chunks):
		if idx == 0:
			
			yvals = ((xchunk - xchunk[0])*alphas[idx]) + const
			
		else:
			yvals = ((xchunk - x_chunks[idx - 1][-1] )*alphas[idx]) + yvals[-1]
		y_chunks.append(yvals) # add this piece to the output

	return np.array([y for ychunk in y_chunks for y in ychunk])





def bkn_st_lines(xvals, breaks, alphas, const = 1):
    	
	breakpoints = [np.min(xvals)*0.98] + breaks + [1.02*np.max(xvals)] # create a list of all the bounding x-values
	x_chunks = [np.array([x_ for x_ in xvals if x_ >= breakpoints[i] and x_ < breakpoints[i+1]]) for i in range(len(breakpoints)-1)]
    
	y_chunks = []
	for idx,xchunk in enumerate(x_chunks):
		if idx == 0:
			
			yvals = xchunk*alphas[idx] 
			c0 = const #- (xchunk[0]*alphas[idx])
			yvals += c0
		else:
			yvals = xchunk*alphas[idx] 
			c0 += breaks[idx - 1]*(alphas[idx - 1] - alphas[idx])
			yvals += c0
		y_chunks.append(yvals) # add this piece to the output
    
	return np.array([y_var for ychunk in y_chunks for y_var in ychunk])


def bkn_st_lines_func(x_, breaks, alphas, const = 1):
	
	try: 
		ind = np.where(np.asarray(breaks) >= x_)[0][0]
		if ind == 0:
			y_ = x_*alphas[ind]
			 #- (xchunk[0]*alphas[ind])
			y_ += const
		else:
			
			for i in range(ind):
				const += (breaks[i]*(alphas[i] - alphas[i + 1]))
			y_ = const + (x_*alphas[ind])
	except:
		for i in range(len(breaks)):
			const += (breaks[i]*(alphas[i] - alphas[i + 1]))
		y_ = const + (x_*alphas[-1])

	return y_
	
def fit_func_st_line(x_, n_, *args):
	# n_ is the number of broken power law
	breaks = list(args[: n_ - 1])
	alphas = list(args[n_ - 1 : -1])
	const = args[-1]
	#print('breaks : ', breaks,' alphas : ', alphas,' const : ',const)
	try:
		len(x_)
		return np.array([bkn_st_lines_func(i, breaks, alphas, const) for i in x_])
	except:
		
		return bkn_st_lines_func(x_, breaks, alphas, const)	



###################################################################################################################################################################################################

file_loc = './'

file_loc = '/data/rahul/psr_file/fs_data/J2144-5237_ar/'


b3 = glob.glob(file_loc+'*500*.ar')
b4 = glob.glob(file_loc+'*550*.ar')
b5 = glob.glob(file_loc+'*1460*.ar')


fig = plt.figure(figsize=(14,7))
ax1 = plt.subplot2grid((2, 3), (0, 0),rowspan=2)
ax2 = plt.subplot2grid((2, 3), (0, 1),rowspan=2)
ax3 = plt.subplot2grid((2, 3), (0, 2))
ax4 = plt.subplot2grid((2, 3), (1, 2))
a = ax1.pcolormesh(np.arange(d.shape[-1]), F[unflag_freq], d.mean(0)/d.mean(0).max())
b = ax2.pcolormesh(np.arange(d.shape[-1]), np.linspace(0, time_s, mjd.shape[0]), d.mean(1)/d.mean(1).max())
ax3.errorbar(nu, flux_nu, yerr = error_nu, fmt = '.')
ax3.loglog()							# keep loglog above set_x/yticks /x/y_ticks: else ticks won't work

ax3.set_xticks(nu[::len(nu)//10+1], nu[::len(nu)//10+1], rotation=25)
ax3.xaxis.set_minor_locator(ticker.AutoMinorLocator())

ax3.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax4.plot(d.mean((0,1))/d.mean((0,1)).max())
plt.colorbar(a, ax = ax1)
plt.colorbar(b, ax = ax2)
plt.subplots_adjust(hspace=0.4,wspace=.4)
#plt.savefig('2mar2k22', dpi=300)
plt.show()




#####################################################################################################################
file_name = os.listdir('./')
# file_name = list of all the ar file in the 2144_tgss67_ar
ext = '.pfd'


band5 = []
band4 = []
band3 = []

#band3, 4, 5 are the files segregated according to the bands of observation (which is mentioned in the name of the ar file)

for i in file_name:

	if i.split('.')[-1] == ext:
		l = [t for s in i.split('_') for t in s.split('.')]

		if '500' in l:
			band3.append(i)
	
		if '550' in l:
			band4.append(i)

		if '1460' in l:
			band5.append(i)


#f = 'J2144-5237_500_200_1024_2.11feb2k19.raw0.dat_PSR_2144-5237.ar'

#data, F, mjds = pfd_data(f)	# data.shape should be (no of subints, pol, nchan, phase bins)
#data = data[:,0]

b5_gptool, b5_raw = [], []
for t in b5:
	l = [t for s in t.split('_') for t in s.split('.')]
	if 'gptool' in l:
		b5_gptool.append(t)
	else:
		b5_raw.append(t)


b5_raw_spec = []
tsky_b3, tsky_b4, tsky_b5 = 24, 7, 1	# for J2144-5237
tsky = tsky_b5
for f in b5_raw:
	data, F, mjd = pfd_data(f)
	data = data[:,0]
	
	print(f)
	try:
		p = data.mean(axis=(0,1))
		on_weight_arr = data.sum(axis=(0,1))/(data.std(axis=(0,1)) * np.sqrt(data.shape[0]*data.shape[1]) ) > 3
		snr = (p - p[~on_weight_arr].mean()).sum()/(p[~on_weight_arr].std() * np.sqrt((~on_weight_arr).sum()))
		print('snr : ' ,snr)
	except:
		continue


	fig, ax = plt.subplots(2)
	ax[0].imshow(data.mean(axis=0))
	ax[0].set_ylabel('Frequency (crunched in time)')
	ax[0].set_xlabel('Phase bins')
	
	ax[1].imshow(data.mean(axis=1))
	ax[1].set_ylabel('Time (crunched in frequency)')
	ax[1].set_xlabel('Phase bins')

	plt.show()
	proceed = input("Proceed ? (1/0):")

	while float(proceed):
		n_chan = 10
		sigma = 10
		thres = 3
		print(f)
		try:
			SNR, Total_flux, noise_nu, error_nu, freq, on_weight_arr = spectra_with_gain(tsky, 'full', float(sigma), 3, False, False, False, FILE = str(f))
			Total_flux = Total_flux *1e3
			spec, snr_nu, flux_nu, noise_nu, error_nu, nu, weight = spectra_with_gain(tsky, int(n_chan), float(sigma), float(thres), False, True, False, FILE = str(f))
		except:
			pass

		
		info = input('Info needed (spec, snr_nu, flux_nu, noise_nu, error_nu, nu) : ')
		try:		
			if info == 'snr_nu':
				print("snr_nu : ", snr_nu)
			elif info == 'spec':
				print("spectral index : ", spec)
			elif info == 'flux_nu':
				print("flux_nu : ", flux_nu)
			elif info == 'error_nu':
				print("error_nu : ", error_nu)
			elif info == 'noise_nu':
				print("noise_nu : ", noise_nu)
			elif info == 'nu':
				print("nu : ", nu)
		except:
			print('Info asked for doesn\'t exist:')

		y_n = input('Correction Needed (1/0):')
		
		while float(y_n):
	
			n_chan = input('N_chan:')
			sigma = input('Sigma:')
			thres = input('Thres:')
			
			try:
				SNR, Total_flux, noise_nu, error_nu, freq, on_weight_arr = spectra_with_gain(tsky, 'full', float(sigma), 3, False, False, False, FILE = str(f))
				Total_flux = Total_flux *1e3
				spec, snr_nu, flux_nu, noise_nu, error_nu, nu, weight = spectra_with_gain(tsky, int(n_chan), float(sigma), float(thres), False, True, False, FILE = str(f))
			except:
				pass
			
			
			info = input('Info needed (spec, snr_nu, flux_nu, noise_nu, error_nu, nu) : ')
			try:		
				if info == 'snr_nu':
					print("snr_nu : ", snr_nu)
				elif info == 'spec':
					print("spectral index : ", spec)
				elif info == 'flux_nu':
					print("flux_nu : ", flux_nu)
				elif info == 'error_nu':
					print("error_nu : ", error_nu)
				elif info == 'noise_nu':
					print("noise_nu : ", noise_nu)
				elif info == 'nu':
					print("nu : ", nu)
			except:
				print('Info asked for doesn\'t exist:')
			
			y_n = input('Correction Needed (1/0):')
		try:
			b5_raw_spec.append(spec)
		except:
			print('Spec not calculated due to no signal above', sigma, 'sigma')
			continue
		
		proceed = 0

	print("############################################################################################################")
	try:
		del spec, snr, flux_nu, error_nu, nu, data, F, mjd, p
	except:
		continue

###############################################################################################################################################################################################
# This version mainly does (data - data_mean[off pulse])/data_std(off_pulse)[:,:,None] -> create the on-off mask. if mask_with_prim_comp_flags_v7 is used then the definition of standard deviation for a chunk in frequecy changes (1/np.sqrt(sum_along_freq(1/variance_along_time)))



def ugmrt_in_band_flux_spectra_v2(Tsky, n_chan = 10, sigma_ = 10, thres = 3, beam_ = 'PA', primary_comp = True, show_plots = True, allow_fit = False, save_plot = False, n_ant = 23, **file_data):
	bandpass_correction = True
	robust_std = lambda x_ : 1.4826*np.median(abs(x_.flatten() - np.median(x_.flatten())))
	if len(file_data.keys()) >2:
		data_, freq_, mjds_, dm_, file_ = file_data['DATA'], file_data['FREQ'], file_data['MJD'], file_data['DM'], file_data['FILE']
		unflag_freq = np.sum(data_, axis = (0,2)) != 0		# for removing fully flagged channels from the data_
		unflag_freq_360_380 = np.logical_or(freq_ < 360, freq_ > 380)
		unflag_freq_360_380 = np.logical_and(unflag_freq_360_380,freq_>258)
		if unflag_freq.shape[0] == unflag_freq.sum():
			unflag_freq = abs(np.median(data_, axis=(0,-1))/np.median(data_, axis=(0,-1)).max()) >1e-1
		unflag_freq = np.logical_and(unflag_freq, unflag_freq_360_380)
		unflag_time = np.sum(data_, axis = (1,2)) != 0
		
		# beam_ = True

	else:
		file_ = file_data['FILE']
		
		if bandpass_correction:
			data_, freq_, mjds_, dm_= pfd_data_b(file_)
			data_ = data_[:,0]
			global_off_bins_bool_arr = off_region_array_from_actual_data(data_)
			#data_ = np.nan_to_num((data_ - np.nanmean(data_[:,:,global_off_bins_bool_arr], axis=-1,keepdims=True))/np.nanmean(data_[:,:,global_off_bins_bool_arr], axis=-1,keepdims=True), nan=0, neginf=0,posinf=0)
			data_ = np.nan_to_num((data_ - np.nanmean(data_[:,:,global_off_bins_bool_arr], axis=-1,keepdims=True))/np.nanstd(data_[:,:,global_off_bins_bool_arr], axis=-1,keepdims=True), nan=0, neginf=0,posinf=0)
			d_mean_ = data_.mean(-1)
			unflag_freq_360_380 = np.logical_or(freq_ < 360, freq_ > 380)
			unflag_freq_360_380 = np.logical_and(unflag_freq_360_380,freq_>258)
			unflag_freq = np.logical_and(d_mean_.std(0) > (np.median(d_mean_.std(0))*0.1) , d_mean_.std(0) < (np.median(d_mean_.std(0)) +15*robust_std(d_mean_.std(0)) ) )
			unflag_freq = np.logical_and(unflag_freq, unflag_freq_360_380)
			unflag_time = np.sum(data_, axis = (1,2)) != 0
		else:
			data_, freq_, mjds_, dm_ = pfd_data(file_)
			data_ = data_[:,0]
			unflag_freq_360_380 = np.logical_or(freq_ < 360, freq_ > 380)
			unflag_freq_360_380 = np.logical_and(unflag_freq_360_380,freq_>258)
			d_mean_ = data_.mean(-1)
			unflag_freq = np.logical_and(d_mean_.std(0) > (np.median(d_mean_.std(0))*0.1) , d_mean_.std(0) < (np.median(d_mean_.std(0)) +15*robust_std(d_mean_.std(0)) ) )
			unflag_freq = np.logical_and(unflag_freq, unflag_freq_360_380)
			unflag_time = np.sum(data_, axis = (1,2)) != 0

	psrchive_method = False
	df = freq_[1] - freq_[0]
	delta_chan = 1
	
	if not(isinstance(n_chan, int)):
		n_chan = freq_.shape[0]#unflag_freq.sum()
	
	freq = freq_[unflag_freq]
	data_ *= unflag_freq[None, :, None] * unflag_time[:, None, None]
	data_unflaged = data_[unflag_time][:,unflag_freq]

	if primary_comp:
		on_weight_arr_f, on_weight_arr_p, chunk_ind_arr = mask_with_prim_comp_flags_v2(data_, unflag_freq, thres, n_chan, prim_comp_ = True)
	else:
		on_weight_arr_, chunk_ind_arr = mask_with_prim_comp_flags_v2(data_, unflag_freq, thres, n_chan, prim_comp_ = False)
		on_weight_arr_f = on_weight_arr_p = on_weight_arr_
	
	#time_s = (mjds_[-1] - mjds_[0] +  np.all(np.diff(mjds_,2) < abs(np.diff(mjds_)).min()*1e-3)*np.diff(mjds_)[0])*86400	
	time_s = np.median(np.diff(mjds_))*mjds_.shape[0]*86400
	fchrunched_data = []
	profile_freq = []
	fcrunch10_freq, chan_arr = [], []
	noise = []
	unflagged_flux_freq, unflagged_rms_freq, snr = [], [], []
	on_off_mask = []
	on_ticks_ind = []	# This variable keeps the track of intra band divisions along with the flagged channels within
	width_term_arr = []
	width_bool_arr = []


	mask_ind = 0
	for chunk_ind in chunk_ind_arr:
		if not unflag_freq[chunk_ind]:
			
			fchrunched_data.append(np.zeros((data_.shape[0],data_.shape[-1])))
			on_off_mask.append(np.zeros(data_.shape[-1],dtype=bool))
			continue
		support = np.zeros(data_.shape[1],dtype=bool)
		support[chunk_ind: chunk_ind + n_chan] = True
		chunk_data = data_[:, unflag_freq*support]
		freq_chunk = freq_[unflag_freq*support]
		#fchrunched_data.append(np.nan_to_num(np.nanmean(chunk_data, axis =1),nan=0))
		# the following for loop is for creating on_off_mask and fchrunched_data of the desired shape (same as 2d shape data_.mean(1)) 
		for j in range(n_chan):
			if (chunk_ind+j)>=unflag_freq.shape[0]:
				break
			if not unflag_freq[chunk_ind+j]:
				# This part is to horizontal lines to be ploted for visual assistance for intra band flagged channels
				try:
					if unflag_freq[chunk_ind+j-1] and not(unflag_freq[chunk_ind+j+1]):
						on_ticks_ind.append(chunk_ind+j)
					if not(unflag_freq[chunk_ind+j-1]) and unflag_freq[chunk_ind+j+1]:
						on_ticks_ind.append(chunk_ind+j + 1)
				except:
					pass
				fchrunched_data.append(np.zeros((data_.shape[0],data_.shape[-1])))
				on_off_mask.append(np.zeros(data_.shape[-1],dtype=bool))
				continue
			fchrunched_data.append(np.nan_to_num(np.nanmean(chunk_data, axis =1),nan=0))
			on_off_mask.append(on_weight_arr_p[mask_ind])
			
		profile_per_chan = np.nanmean(chunk_data, axis=(0,1))
		profile_freq.append(profile_per_chan)
		fcrunch10_freq.append(np.mean(freq_chunk))

		nu_sub_arr = np.arange(freq_chunk[0] - 0.5*df + 0.5, freq_chunk[-1] + 0.5*df, delta_chan)
		width_term = np.sqrt( on_weight_arr_p[mask_ind].sum()/(~on_weight_arr_p[mask_ind]).sum() )
		noise.append(np.sqrt( (sefd(nu_sub_arr, np.mean(nu_sub_arr), time_s, Tsky, n_ant, beam_)**2).sum() )/nu_sub_arr.shape[0] * width_term )
		width_term_arr.append(width_term)
		width_bool_arr.append(on_weight_arr_p[mask_ind].sum() > 1)
		unflagged_flux_freq.append(np.nanmean(profile_per_chan[on_weight_arr_p[mask_ind]]))
		#unflagged_flux_freq.append(np.nanmean(profile_per_chan[mask_ind]))
		unflagged_rms_freq.append(np.nanstd(profile_per_chan[~on_weight_arr_f[mask_ind]]))
		snr.append(unflagged_flux_freq[-1] * np.sqrt( on_weight_arr_p[mask_ind].sum())/unflagged_rms_freq[-1])
		#snr.append(unflagged_flux_freq[-1]/unflagged_rms_freq[-1])
		mask_ind +=1
		on_ticks_ind.append(chunk_ind)
	on_ticks_ind = np.array(on_ticks_ind)
	on_off_mask = np.array(on_off_mask)
	fchrunched_data = np.array(fchrunched_data).transpose(1,0,2)
	profile_freq = np.array(profile_freq)
	fcrunch10_freq = np.nan_to_num(fcrunch10_freq)
	unflagged_flux_freq = np.array(unflagged_flux_freq)
	unflagged_rms_freq = np.array(unflagged_rms_freq)
	noise = np.array(noise)
	snr = np.array(snr)
	width_term_arr = np.array(width_term_arr)
	width_mask = width_term_arr >= np.median(width_term_arr) - 3*1.4*(np.median(abs(width_term_arr - np.median(width_term_arr))))
	#width_mask *= width_bool_arr
	#width_mask = True
	profile = data_.mean(axis=(0,1))
	snr3_bool = np.nan_to_num(snr, nan=0,posinf=0.0, neginf=0.0) > sigma_
	#print(unflagged_flux_freq, unflagged_rms_freq)
	flux = np.where(snr3_bool * width_mask, snr * noise, np.nan)#(snr * noise)[snr3_bool * width_mask]
	noise_ = np.where(snr3_bool * width_mask, noise, np.nan)#noise[snr3_bool * width_mask]
	error = np.sqrt(noise_**2 + (0.1 * flux)**2)
	log_freq, log_flux = np.where(snr3_bool * width_mask, np.log10(fcrunch10_freq), np.nan), np.log10(flux)
	error_y = error  # noise_
	#return log_flux, noise_, error, log_freq
	'''
	flux = (snr * noise)[snr3_bool * width_mask]
	noise_ = noise[snr3_bool * width_mask]
	error = np.sqrt(noise_**2 + (0.1 * flux)**2)
	log_freq, log_flux = np.log10(fcrunch10_freq[snr3_bool * width_mask]), np.log10(flux)
	'''
	#print(snr, flux, noise_, error)
	#return(data_)
	#flux = (snr * noise)[snr3_bool]
	#error = np.sqrt(noise[snr3_bool]**2 + (0.1 * flux)**2)
	#log_freq, log_flux = np.log10(fcrunch10_freq[snr3_bool]), np.log10(flux)
	#fit_possible = False
	if allow_fit:
		AICC = []
		param_arr = []
		param_err_arr = []
		try:
			for n_pow_law_ind in range(1,int(len(log_freq[~np.isnan(log_freq)])/2)):
				#n_pow_law_ind = 1
				params_0, bounds_0 = initial_guess_n_bounds_func( n_pow_law_ind, log_freq[~np.isnan(log_freq)], log_flux[~np.isnan(log_flux)])
				#print('TRIED THIS STEP : 1 ', params_0)
				#param_, _ = curve_fit(lambda nu_,*param_0: fit_func_st_line(nu_, n_pow_law_ind, *param_0), log_freq, log_flux, p0 = params_0, bounds = bounds_0, maxfev = 5000, nan_policy='omit')
				param_, _, infodict_, mesg_, ier_ = curve_fit(lambda nu_,*param_0: fit_func_st_line(nu_, n_pow_law_ind, *param_0), log_freq, log_flux, p0 = params_0, sigma = noise_[~np.isnan(noise_)]/(np.log(10)*flux[~np.isnan(log_flux)]) , bounds = bounds_0, maxfev = 5000,absolute_sigma=True,full_output=True, nan_policy='omit')
				#print('TRIED THIS STEP : 2 ', param_)
				#param_, _ = curve_fit(lambda nu_,*param_0: fit_func_st_line(nu_, n_pow_law_ind, *param_0), log_freq, log_flux, p0 = params_0, sigma = error_y/(np.log(10)*flux) , bounds = bounds_0, maxfev = 5000, nan_policy='omit')
				Chi2 = ( ((log_flux[~np.isnan(log_flux)] -  fit_func_st_line(log_freq[~np.isnan(log_freq)], n_pow_law_ind, *param_))/(noise_[~np.isnan(noise_)]/(np.log(10)*flux[~np.isnan(log_flux)])))**2 ).sum()

				#Obs_Model = log_flux[~np.isnan(log_flux)] -  fit_func_st_line(log_freq[~np.isnan(log_freq)], n_pow_law_ind, *param_))
				#del_f_param = lambda x_, param_0: np.array(list(param_0[n_pow_law_ind - 1: -2] - param_0[n_pow_law_ind : -1]) + list(np.array(list(param_0[: n_pow_law_ind - 1]) + [x_]) - np.array([0] + list(param_0[: n_pow_law_ind - 1]))) + [1])
				#del_f_per_obs_pt = np.array([del_f_param(nu_i , param_).T @ _ @ del_f_param(nu_i , param_) for nu_i in log_freq[~np.isnan(log_freq)]])
				#print('del_f_per_obs_pt : ', del_f_per_obs_pt)
				#print(infodict_)
				#print('TRIED THIS STEP : 3')
				#Chi2 = ( ((10**log_flux[~np.isnan(log_flux)] -  10**fit_func_st_line(log_freq[~np.isnan(log_freq)], n_pow_law_ind, *param_))/noise_[~np.isnan(noise_)])**2 ).sum()
				#print('number power law = ', n_pow_law_ind, ' ; chi2 = ',Chi2)
				aicc_ = Chi2 + 2*len(param_)*len(log_freq[~np.isnan(log_freq)])/(len(log_freq[~np.isnan(log_freq)]) - len(param_) - 1)
				AICC.append(aicc_)
				param_arr.append(param_)
				param_err_arr.append(_)
				#print(Chi2, param_, _, aicc_)
				#print('###############################################################')
				del params_0, bounds_0, Chi2, param_, _, aicc_
				n_pow_law = int(np.argmin(AICC) + 1)
				opt_param = param_arr[np.argmin(AICC)]
				opt_param_err = np.diag(param_err_arr[np.argmin(AICC)])
				
		except:
			print('FITTING PROCESS FAILED')
			pass
	
	####################################################################
	if bool(save_plot + show_plots) :
		fig = plt.figure(figsize=(15,10))
		label_size = 15
		N_div = 10
		N_tick = len(log_freq)//N_div
		alpha0 = 1
		alpha1 = 0.2
		subp1 = fig.add_subplot(2, 3, (1,4))
		ax1 = subp1.imshow(data_.mean(axis=0)/data_.mean(), aspect='auto', origin='lower') # ,extent=[0,data_.shape[-1],freq_.min(),freq_.max()])
		plt.colorbar(ax1, ax=subp1)
		subp1.set_yticks(on_ticks_ind - 0.5, freq_.astype(int)[on_ticks_ind])
		for i in on_ticks_ind: subp1.axhline(i - 0.5, color='white',alpha=alpha0,linestyle='--')
		subp1.set_ylabel('Frequency', fontsize = label_size)
		subp1.set_xlabel('Phase bins', fontsize = label_size)
		subp1.tick_params(axis ='both', labelsize = label_size)
		
		############################################################################
		subp2 = fig.add_subplot(2, 3, 2)
		subp2.imshow(on_off_mask, aspect='auto', origin='lower')# , extent=[0,data_.shape[-1],freq_.min(),freq_.max()])
		subp2.set_yticks(on_ticks_ind[::2] - 0.5, freq_.astype(int)[on_ticks_ind[::2]])
		for i in on_ticks_ind: subp2.axhline(i - 0.5, color='white',alpha=alpha1,linestyle='--')
		
		subp2.tick_params(axis ='both', labelsize = label_size)
		###############################################################################
		subp3 = fig.add_subplot(2, 3, 5)
		subp3.imshow(fchrunched_data.mean(axis=0), aspect='auto', origin='lower')# , extent=[0,data_.shape[-1],freq_.min(),freq_.max()])
		subp3.set_yticks(on_ticks_ind[::2] - 0.5, freq_.astype(int)[on_ticks_ind[::2]])
		for i in on_ticks_ind: subp3.axhline(i - 0.5, color='white',alpha=alpha1,linestyle='--')
		subp3.tick_params(axis ='both', labelsize = label_size)
		################################################################
		subp4 = fig.add_subplot(2, 3, 3)
		subp4.errorbar(10**log_freq, flux, yerr = error_y, fmt = '.')
		if allow_fit:
			try:
				#print('NOW PLOTTING THE SPECTRA')
				n_pow_law = int(np.argmin(AICC) + 1)
				opt_param = param_arr[np.argmin(AICC)]
				opt_param_err = np.diag(param_err_arr[np.argmin(AICC)])
				break_arr, spectral_index_arr = opt_param[: n_pow_law -1], opt_param[n_pow_law - 1: -1]
				break_err_arr, spectral_index_err_arr = opt_param_err[: n_pow_law -1], opt_param_err[n_pow_law - 1: -1]
				subp4.plot(10**np.linspace(log_freq[np.nanargmin(log_freq)], log_freq[np.nanargmax(log_freq)], 1000), 10**fit_func_st_line(np.linspace(log_freq[np.nanargmin(log_freq)], log_freq[np.nanargmax(log_freq)], 1000), n_pow_law, *opt_param) )
				for i in range(n_pow_law): print("spectral index {0} +- {1} ".format(round(spectral_index_arr[i],3), round(spectral_index_err_arr[i],3)))
				subp4.set_title('Spectal Index:' + "{0} +- {1} ".format(np.around(spectral_index_arr,3), np.around(spectral_index_err_arr,3)) + '\n' + 'Breaks at (MHz):' + "{0} +- {1}".format(np.around(10**break_arr,3), np.around(np.log(10)*break_err_arr*10**break_arr,3)))
			except:
				#print('$$$$$$$ ^^^^^^^ CAN\'T FIT THE SPECTRA')
				pass
		subp4.loglog()
		subp4.set_xticks(freq[::len(freq)//N_div + 1], freq.astype(int)[::len(freq)//N_div + 1],rotation=25)
		subp4.xaxis.set_minor_locator(ticker.AutoMinorLocator())
		subp4.tick_params(axis ='both', labelsize = label_size)
		####################################################################################
		
		subp5 = fig.add_subplot(2, 3, 6)
		subp5.plot(profile/profile.max())
		try:
			subp5.plot(np.arange(data_.shape[-1])[on_weight_arr.sum(0).astype(bool)], (profile/profile.max())[on_weight_arr.sum(0).astype(bool)], 'o')
		except:
			pass
		subp5.set_ylabel('Normalized amplitude', fontsize = label_size )
		subp5.set_xlabel('Phase bins', fontsize = label_size )
		
		try: 
			subp5.set_title('Total flux : ' + str(round(Total_flux[0],3)) + '(mJy); snr : ' + str(round(SNR[0], 3)), fontsize = label_size )
		except:
			pass
		subp5.tick_params(axis ='both', labelsize = label_size)
		try:
			epoch = Time(mjds_[0], format='mjd').ymdhms
			plt.suptitle('Epoch : ' + str(epoch[2]) + '/' + str(epoch[1]) + '/' + str(epoch[0])  + '; Time : ' + str(round(time_s,2)) + r'$(s) ; n_{chan} : $' + str(n_chan) + '; sigma : ' + str(sigma_) + '; thres : ' + str(thres)+'\n'+str(file_), fontsize = label_size)
		except:
			epoch = Time(mjds_[0], format='mjd').ymdhms
			plt.suptitle('Epoch : ' + str(epoch[2]) + '/' + str(epoch[1]) + '/' + str(epoch[0])  + '; Time : ' + str(round(time_s,2)) + r'$(s) ; n_{chan} : $' + str(n_chan) + '; sigma : ' + str(sigma_) + '; thres : ' + str(thres), fontsize = label_size)
			pass
		#plt.tight_layout()
		
		if save_plot:
			file_name_ = str('./plot_files/') + file_[:-4] + str('.png')
			plt.savefig(str(file_name_), dpi=200)
			
		if show_plots:
			plt.show()
		else:
			plt.close()
	####################################################################
	if allow_fit:
		try:	
			len(opt_param)
			return [opt_param, opt_param_err], snr, flux, noise, error, np.rint(10**log_freq), on_off_mask, dm_
			
		except:
			return snr, flux, noise, error, np.rint(10**log_freq), on_off_mask, dm_
			
	else:
		return snr, flux, noise, error, np.rint(10**log_freq), on_off_mask, dm_





###########################		Main Function ends here

#####################################################################################################################


###############################################################################################################################################################################################

####################################				Modified Code 
files = glob.glob('*pfd')

files_mjd_freq_file = []
for f in files:
	freq, mjd = pfd_data(f)[1][0], pfd_data(f)[2][0]
	try:
		files_mjd_freq_file.append([mjd, freq, str(f)])

	except:
		print(f)
		pass

files_mjd_freq_file = np.array(files_mjd_freq_file, dtype=object)
files_mjd_freq_file = files_mjd_freq_file[files_mjd_freq_file[:,0].astype(float).argsort()]

group_mjd = group(files_mjd_freq_file)

count = count_G = 0
tsky_b3, tsky_b4, tsky_b5 = 22, 6, 1	# J1120-3618
problem_file = []
for G in group_mjd:
	print(count_G)
	if np.array(G).shape[0] == 1:
		n_ant = 24
	else:
		if sum(abs(np.diff(np.array(G)[:,1].astype(float))) >100).astype(bool):
			n_ant = 12
		else:
			n_ant = 24
	for g in G:
		
		f = g[-1]
		data, F, mjd = pfd_data(f)
		data = data[:,0]
		if F.mean() > 250 and F.mean() < 500:
			tsky = tsky_b3
		elif F.mean() > 550 and F.mean() < 850:
			tsky = tsky_b4
		elif F.mean() > 980 and F.mean() < 1500:
			tsky = tsky_b5
		print('file : ', f)
		print('n_ant : ', n_ant)
		print('Sky temp : ',tsky)
		n_chan = 7
		sigma = 5
		thres = 3
		try:
			print(count)
			SNR, Total_flux, noise_nu, error_nu, freq, on_weight_arr = spectra_with_gain(tsky, 'full', float(sigma), 3, True, False, False, False, False, n_ant, FILE = str(f))
			Total_flux = Total_flux *1e3
			spec, snr_nu, flux_nu, noise_nu, error_nu, nu, weight = spectra_with_gain(tsky, int(n_chan), float(sigma), float(thres), True, False, False, True, True, n_ant, FILE = str(f))
			del data, F, mjd, SNR, Total_flux, noise_nu, error_nu, freq, on_weight_arr, spec, snr_nu, flux_nu, nu, weight
			print('=============================================================')
			print('Completed')
			print('=============================================================')
		except:
			print(count + 10000)
			print('problem file : ' )
			print(f)
			problem_file.append(str(f))
			pass
		print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
		count += 1
	print('#####################################################')	
	count_G += 1
#######################################################################################################################################################################


b3_mjd_f = []
tsky_b3, tsky_b4, tsky_b5 = 22, 6, 1
tsky = tsky_b3
for f in band3:
	mjd = pfd_data(f)[-1].astype(np.float128)
	SNR, Total_flux, noise_nu, error_nu, freq, on_weight_arr,ch_arr = spectra_with_gain(tsky, 'full', 5, 3, False, False, False, FILE = str(f))
	Total_flux = Total_flux *1e3
	error_nu = error_nu * 1e3
	try:
		b3_mjd_f.append([mjd[0], Total_flux[0], error_nu[0]])
		
	except:
		print(f)
		pass

b3_mjd_f = np.array(b3_mjd_f, dtype=object)



plt.plot(b3_mjd_f[:,0], b3_mjd_f[:,1],'v')

plt.plot(b3_mjd_f[b3_mjd_f[:,0].argsort()][:,0], b3_mjd_f[b3_mjd_f[:,0].argsort()][:,1])

plt.plot(b3_mjd_f[b3_mjd_f[:,0].argsort()][:,0], b3_mjd_f[b3_mjd_f[:,0].argsort()][:,1],label='band3')

a = b3_mjd_f[b3_mjd_f[:,0].argsort()]

b3_epoch_bool = np.append(np.diff(a[:,0]) > 1000/86400 , True)
plt.plot(a[b3_epoch_bool][:,0], a[b3_epoch_bool][:,1],'.')
plt.show()
#################################################################################################################
#####################	Subdividing the mjd array as same epochs
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

###################################################################################################################



def group_max_snr(X):
	if X.ndim >2:
		B = np.append(np.diff(X[:,0]) > 1000/86400, True)
	else:
		B = np.append(np.diff(X) > 1000/86400, True)
	#G = []
	ind_g = []
	flux_arr, flux_error, mjd = [], [], []
	
	for ind, value in enumerate(X[:,0]):

		if B[ind]:
			
			ind_g.append(ind)
			#G.append(g)
			
			flux_max_ind = X[ind_g,1].argmax()

			mjd.append(X[ind_g, 0][flux_max_ind])
			flux_arr.append(X[ind_g, 1][flux_max_ind])
			flux_error.append(X[ind_g, 2][flux_max_ind])
			ind_g = []
			
			pass
		else:

			ind_g.append(ind)
			
	return np.array([mjd, flux_arr, flux_error]).T


a_f_max = group(a)
plt.errorbar(a_f_max[:,0], a_f_max[:,1], yerr=a_f_max[:,2], fmt='.',label='band3')
plt.show()


b3_h_f = a_f_max[a_f_max[:,1] > np.median(a_f_max[:,1])*3][:,1:]
b3_l_f = a_f_max[a_f_max[:,1] < np.median(a_f_max[:,1])*3][:,1:]

fit_line = lambda x, m, c: (m*x) +c
x, _ = curve_fit(fit_line, np.log10([400,650,1260]), np.log10([b3_h_f[:,0].mean(),b4_h_f[:,0].mean(),b5_h_f[:,0].mean()]) )

plt.errorbar([400,650,1260], [b3_h_f[:,0].mean(),b4_h_f[:,0].mean(),b5_h_f[:,0].mean()], yerr=[b3_h_f[:,-1].max(),b4_h_f[:,-1].max(),b5_h_f[:,-1].max()], fmt='o')	
plt.plot([400,650,1260], [10**fit_line(i, x[0], x[1]) for i in np.log10([400,650,1260])] )
plt.xticks([400,650,1260], [400,650,1260])
#plt.yticks([b3_h_f[:,0].mean(),b4_h_f[:,0].mean(),b5_h_f[:,0].mean()],[round(b3_h_f[:,0].mean(),3),round(b4_h_f[:,0].mean(),3),round(b5_h_f[:,0].mean(),3)])
plt.tick_params(axis='x', which='minor', bottom = False, labelbottom = False)
plt.tick_params(axis='y', which='minor', bottom = False, labelbottom = False)
plt.tick_params(axis='both', which='minor', bottom = False, labelbottom = False)
plt.loglog()
plt.show()


x, _ = curve_fit(fit_line, np.log10([400,650,1260]), np.log10([b3_l_f[:,0].mean(),b4_l_f[:,0].mean(),b5_l_f[:,0].mean()]) )
plt.errorbar([400,650,1260], [b3_l_f[:,0].mean(),b4_l_f[:,0].mean(),b5_l_f[:,0].mean()], yerr=[b3_l_f[:,-1].max(),b4_l_f[:,-1].max(),b5_l_f[:,-1].max()], fmt='o')	
plt.plot([400,650,1260], [10**fit_line(i, x[0], x[1]) for i in np.log10([400,650,1260])] )
plt.xticks([400,650,1260], [400,650,1260])
#plt.yticks([b3_h_f[:,0].mean(),b4_h_f[:,0].mean(),b5_h_f[:,0].mean()],[round(b3_h_f[:,0].mean(),3),round(b4_h_f[:,0].mean(),3),round(b5_h_f[:,0].mean(),3)])
plt.tick_params(axis='x', which='minor', bottom = False, labelbottom = False)
plt.tick_params(axis='y', which='minor', bottom = False, labelbottom = False)
plt.tick_params(axis='both', which='minor', bottom = False, labelbottom = False)
plt.loglog()
plt.show()





#############################################################################################################################################
###################				code for ankita

file_name = os.listdir('./')
band3 = []
for f in file_name:
	s = []
	for x in f.split('_'): s.extend(x.split('.'))
	
	if s[-1] == 'pfd':
		band3.append(f)



b3_mjd_freq_file = []
for f in band3:
	freq, mjd = pfd_data(f)[1][0], pfd_data(f)[2][0]
	try:
		b3_mjd_freq_file.append([mjd, freq, str(f)])

	except:
		print(f)
		pass


b3_mjd_freq_file = np.array(b3_mjd_freq_file, dtype=object)
b3_mjd_freq_file = b3_mjd_freq_file[b3_mjd_freq_file[:,0].astype(float).argsort()]


b3_mjd_SNR_flux_error_file = []
tsky_b3, tsky_b4, tsky_b5 = 32, 6, 1

for G in group_mjd:
	if np.array(G).shape[0] == 1:
		n_ant = 24
	else:
		if sum(abs(np.diff(np.array(G)[:,1].astype(float))) >100).astype(bool):
			n_ant = 12
		else:
			n_ant = 24
	for g in G:
		
		f = g[-1]
		data, F, mjd = pfd_data(f)
		data = data[:,0]
		if F.mean() > 250 and F.mean() < 500:
			tsky = tsky_b3
		elif F.mean() > 550 and F.mean() < 850:
			tsky = tsky_b4
		elif F.mean() > 980 and F.mean() < 1500:
			tsky = tsky_b5
		print(f)
		print(n_ant)
		print(tsky)
		SNR, Total_flux, noise_nu, error_nu, freq, on_weight_arr = spectra_with_gain(tsky, 'full', 5, 3, False, False, False, n_ant, FILE = str(f))
		Total_flux = Total_flux *1e3
		error_nu = error_nu * 1e3
		try:
			b3_mjd_SNR_flux_error_file.append([mjd[0], SNR[0], Total_flux[0], error_nu[0], str(f)])
			
		except:
			print(f)
			pass
	print('#####################################################')

np.savetxt('band3.csv', b3_mjd_SNR_flux_error_file , delimiter=',', fmt= '% s')


############################################################################################################
#######################							Code for sangita


b34 = res_list = [y for x in [band3, band4] for y in x]


b34_mjd_freq_file = []
for f in b34:
	freq, mjd = pfd_data(f)[1][0], pfd_data(f)[2][0]
	try:
		b34_mjd_freq_file.append([mjd, freq, str(f)])
	     
	except:
		print(f)
		pass


b34_mjd_freq_file = np.array(b34_mjd_freq_file, dtype=object)
b34_mjd_freq_file = b34_mjd_freq_file[b34_mjd_freq_file[:,0].astype(float).argsort()]


def group(X):
	B = np.append(np.diff(X[:,0].astype(float)) > 1000/86400, True)
	G = []
	g = []
	
	for i in np.arange(X.shape[0]):

		if B[i]:
			
			g.append(X[i])
			G.append(g)
			
			g = []
			

			pass
		else:

			g.append(X[i])
			
	return np.array(G, dtype=object)

group_mjd = group(b34_mjd_freq_file)


#b34_mjd_SNR_flux_error_file = []
tsky_b3, tsky_b4, tsky_b5 = 35, 10, 1

for G in group_mjd:
	if np.array(G).shape[0] == 1:
		n_ant = 24
	else:
		if sum(abs(np.diff(np.array(G)[:,1].astype(float))) >100).astype(bool):
			n_ant = 12
		else:
			n_ant = 24
	for g in G:
		
		f = g[-1]
		data, F, mjd = pfd_data(f)
		data = data[:,0]
		if F.mean() > 250 and F.mean() < 500:
			tsky = tsky_b3
		elif F.mean() > 550 and F.mean() < 850:
			tsky = tsky_b4
		elif F.mean() > 980 and F.mean() < 1500:
			tsky = tsky_b5
		print(f)
		print(n_ant)
		print(tsky)
		try:
			SNR, Total_flux, noise_nu, error_nu, freq, on_weight_arr = spectra_with_gain(tsky, 'full', 5, 3, False, False, False, n_ant, FILE = str(f))
			Total_flux = Total_flux *1e3
			error_nu = error_nu * 1e3
			#b34_mjd_SNR_flux_error_file.append([mjd[0], SNR[0], Total_flux[0], error_nu[0], str(f)])
			
			
		except:
			print(f)
			pass
	print('#####################################################')

#############################		searching for eclipse region in time axis


# data, F, mjd = pfd_data(f)
# data = data[:,0]

# mask = data.sum(1)/(data.std(1) * np.sqrt(data.shape[1]) ) > 3

# snr = (data.mean(1)*mask - (data.mean(1)*(~mask)).mean(-1)[:,None]).sum(-1)/((data.mean(1)*(~mask)).std(-1) *np.sqrt(mask.sum(-1)))




eclipse_duration = []

####################################################################################################################

def filter_clean_isolated_cells(array, struct = ndimage.generate_binary_structure(2,2)):
	""" Return array with completely isolated single cells removed
	:param array: Array with completely isolated single cells
	:param struct: Structure array for generating unique regions
	:return: Array with minimum region size > 1
	"""
	filtered_array = np.copy(array)
	try :
		id_regions, num_ids = ndimage.label(filtered_array, structure=struct)
	except:
		id_regions, num_ids = ndimage.label(filtered_array)
	id_sizes = np.array(ndimage.sum(array, id_regions, range(num_ids + 1)))
	area_mask = (id_sizes == 1)
	filtered_array[area_mask[id_regions]] = 0
	return filtered_array



def smoothing_2d_array(array_):
	# array_ should be a binary 2d array (boolean array)
	# in a 5x5 grid it searches for no of 1 values and only masks if # of points in such a grid is > 3
	# lastly it picks up the pixels common in array_ and mask5_
	
	mask5_ = ss.convolve2d(array_, np.ones((5,5)),mode='same') > 3
	return np.logical_and(array_, ndimage.binary_dilation(mask5_))

##################################################################################################################

def eclipse_region(d_, thr_f_, thr_t_, smoothing = True, smoothing_1d_snr = True):
	unflag_freq_ = d_.sum((0,2)) != 0
	d_ = d_[:, unflag_freq_]
	###########		using frequency axis to calculate the statistics; Also NOTE: thr_f decides on-off bins but thr_t depends on the definition of eclipse region.
	mask_ = d_.sum(1)/(d_.std(1) * np.sqrt(d_.shape[1]) ) > thr_f_		# mask.shape = (# of subints, # of phase bins)
	if smoothing:
		mask_ = smoothing_2d_array(mask_)
	#snr_t = (d.mean(1)*mask_ - (d.mean(1)*(~mask_)).mean(-1)[:,None]).sum(-1)/((d.mean(1)*(~mask_)).std(-1) *np.sqrt(mask_.sum(-1)))
	#snr_t = np.nan_to_num((d_.mean(1)*mask_).sum(-1)/((d_.mean(1)*(~mask_)).std(-1) *np.sqrt(mask_.sum(-1))), nan=0, neginf=0, posinf=0)
	snr_t = np.nan_to_num(np.array([d_.mean(1)[i, mask_[i]].sum()/(d_.mean(1)[i, ~mask_[i]].std() * np.sqrt(mask_[i].sum())) for i in np.arange(mask_.shape[0])]), nan=0, neginf=0, posinf=0)
	if smoothing_1d_snr:
		snr_t = np.convolve(snr_t, np.ones(3)/3, mode='same')
	return snr_t < thr_t_, snr_t, mask_

##################################################################################################################

'''
thr_f, thr_t = 3, 5
count = 0
tsky_b3, tsky_b4, tsky_b5 = 23, 6, 1
for f in files:
	print('#################################################################################################################################')
	try:
		print(count)
		data, F, mjd = pfd_data(f)
		data = data[:,0]
		# eclipse region:
		if F.mean() > 250 and F.mean() < 500:
			tsky = tsky_b3
		elif F.mean() > 550 and F.mean() < 850:
			tsky = tsky_b4
		elif F.mean() > 980 and F.mean() < 1500:
			tsky = tsky_b5
		try:
			e_r, SNR_t, snr_mask = eclipse_region(data, thr_f, thr_t)
			SNR, Total_flux, noise_nu, error_nu, freq, on_weight_arr = spectra_with_gain(tsky, 'full', thr_f, thr_t, True, False,False,False, False, DATA = data[~e_r], FREQ = F, MJD = mjd[~e_r])
		except:
			e_r, SNR_t, snr_mask = eclipse_region(data, thr_f, thr_t, False)
			SNR, Total_flux, noise_nu, error_nu, freq, on_weight_arr = spectra_with_gain(tsky, 'full', thr_f, thr_t, True, False,False,False, False, DATA = data[~e_r], FREQ = F, MJD = mjd[~e_r])
		
		Total_flux, error_nu = Total_flux *1e3, error_nu*1e3
		
		fig, ax = plt.subplots(1,5)
		ax[0].pcolormesh(data.mean(0))
		ax[1].pcolormesh(data.mean(1))
		ax[2].pcolormesh(snr_mask)
		ax[3].pcolormesh(np.arange(data.shape[-1]),((mjd - T_asc)/P_b)%1, data.mean(1))
		ax[4].plot(SNR_t, np.arange(data.shape[0]))
		ax[4].plot(np.ones(data.shape[0]) * thr_t, np.arange(data.shape[0]))
		ax[4].plot(SNR_t[e_r], np.arange(data.shape[0])[e_r], 'o')
		ax[4].set_xlabel('SNRs')
		ax[4].set_ylabel('MJDs')
		plt.suptitle('SNR : ' + str(round(SNR[0],2)) + ' , Total Flux (in mJy) : ' + str(round(Total_flux[0], 2)) + '+\-' + str(round(error_nu[0],2)) + '\n' + str(f))
		#plt.savefig(str('./plot_files/' +''.join(f.split('.')[0] + '_' + f.split('.')[1])), dpi=200)
		file_name = str('./plot_files/') + f[:-4] + str('.png')
		plt.savefig(file_name, dpi=200)
		plt.close()
		plt.show()
		print('eclipse fraction : ', e_r.sum()/data.shape[0])
		del data, F, mjd, SNR, Total_flux, noise_nu, error_nu, freq, on_weight_arr, e_r, SNR_t, snr_mask
	except:
		print(count + 10000)
		print(f)
		#break

	count += 1

'''

# Now the cut-off frequency:


files = glob.glob('*.pfd')
b34_mjd_freq_file = []
for f in files:
	freq, mjd = pfd_data(f)[1][0], pfd_data(f)[2][0]
	try:
		b34_mjd_freq_file.append([mjd, freq, str(f)])
	     
	except:
		print(f)
		pass
b34_mjd_freq_file = np.array(b34_mjd_freq_file, dtype=object)
b34_mjd_freq_file = b34_mjd_freq_file[b34_mjd_freq_file[:,0].astype(float).argsort()]

group_mjd = group(b34_mjd_freq_file)

band_func = lambda nu : 5 if 1000<=nu<=1460 else (4 if 550<=nu<=1000 else (3 if 250<=nu<=500 else (2 if 100<=nu<=250 else 'No Band info')))

thr_f, thr_t = 4, 5
count = count_G = 0
tsky_b3, tsky_b4, tsky_b5 = 23, 6, 1 # For J1544+4937

P_b, T_asc = 0.1207729895, 56124.7701121 # For J1544+4937
orb_ph = lambda mjd_arr: ((mjd_arr - T_asc)/P_b)%1
Table_SNR_Total_flux_error_Tobs_ant = []

eclipse_start, eclipse_stop = 0.15, 0.32
for G in group_mjd:
	print(count_G)
	if np.array(G).shape[0] == 1:
		n_ant = 24
	else:
		if sum(abs(np.diff(np.array(G)[:,1].astype(float))) >100).astype(bool):
			n_ant = 12
		else:
			n_ant = 24
	for g in G:
		
		f = g[-1]
		data, F, mjd, dm_ = pfd_data(f)
		data = data[:,0]
		#'''
		flag_300_345 = np.logical_and(300 < F, F < 345)
		data = data[:, flag_300_345]
		F = F[flag_300_345]
		#'''
		if band_func(F.mean()) == 2:
			continue
			tsky = tsky_b2
		elif band_func(F.mean()) == 3:
			#continue
			tsky = tsky_b3		# Note: some epochs where the 250 < F < 450, the bandshape polynomial misbehaves near 250. Be mindful !
		elif band_func(F.mean()) == 4:
			continue
			tsky = tsky_b4
		elif band_func(F.mean()) == 5:
			continue
			tsky = tsky_b5
		
		print('file : ', f)
		print('n_ant : ', n_ant)
		print('Sky temp : ',tsky)
		sigma = 5
		thres = 4
		try:
			try:
				print(count)
				e_r_0, SNR_t, snr_mask = eclipse_region(data[:,F>258], thr_f, thr_t)
				e_r_0 = filter_max_cells(e_r_0)
				print('eclipse region calculated: step 1')
				#SNR_t, snr_mask = eclipse_region(data, thr_f, thr_t)[1:]
				e_r_23 = np.logical_and(orb_ph(mjd) > eclipse_start, orb_ph(mjd) < eclipse_stop)
				e_r = np.logical_or(e_r_0, e_r_23)
				print('eclipse region calculated: step 2')
				if np.all(e_r):
					e_r = np.logical_and(e_r_0, e_r_23)
				else:
					e_r = np.logical_or(e_r_0, e_r_23)
				print('eclipse region calculated: step 2')
				SNR, Total_flux, noise_nu, error_nu, freq, on_weight_arr, dm_ = ugmrt_in_band_flux_spectra_v2(tsky, 'full', float(sigma), float(thres), 'PA', False,False,False,False,n_ant,DATA=data[~e_r][:,F>258],FREQ=F[F>258], MJD=mjd[~e_r],DM=dm_,FILE=f)
				m = mjd[~e_r]
				print('method : smoothing')
				#if not bool(Total_flux):

			except:			
				e_r_0, SNR_t, snr_mask = eclipse_region(data[:,F>258], thr_f, thr_t, False, False)
				e_r_0 = filter_max_cells(e_r_0)
				#SNR_t, snr_mask = eclipse_region(data, thr_f, thr_t)[1:]
				e_r_23 = np.logical_and(orb_ph(mjd) > eclipse_start, orb_ph(mjd) < eclipse_stop)
				e_r = np.logical_or(e_r_0, e_r_23)
				if np.all(e_r):
					e_r = np.logical_and(e_r_0, e_r_23)
				else:
					e_r = np.logical_or(e_r_0, e_r_23)
				SNR, Total_flux, noise_nu, error_nu, freq, on_weight_arr, dm_ = ugmrt_in_band_flux_spectra_v2(tsky, 'full', float(sigma), float(thres), 'PA', False,False,False,False, n_ant, DATA=data[~e_r][:,F>258], FREQ=F[F>258], MJD=mjd[~e_r],DM=dm_,FILE=f)
				m = mjd[~e_r]
				print('method : Non-smoothing')
			Total_flux, error_nu = Total_flux *1e3, error_nu*1e3
			Table_SNR_Total_flux_error_Tobs_ant.append([Time(mjd.mean(), format='mjd').strftime("%d %B %Y"), round(SNR[0], 3), round(Total_flux[0],3), round(error_nu[0], 3), round((mjd[-1] - mjd[0])*1440, 3), n_ant, (np.all(~e_r)*(mjd[-1] - mjd[0]) + np.any(e_r)*np.median(np.diff(m))*m.shape[0])*24*60, noise_nu[0]*1e3, on_weight_arr.mean(0).astype(bool).sum()/(on_weight_arr.shape[-1]), round(np.median(np.diff(F))*F.shape[0], 3)])
			print('checkpoint 1')
			# fig = plt.figure(figsize=(15,10))
			fig, ax = plt.subplots(1,5, figsize=(20,10))
			ax[0].imshow(data.mean(0), origin='lower', aspect = 'auto')
			ax[0].set_yticks(np.arange(0, data.shape[1], int(data.shape[1]/7)), np.round(F,3)[::int(data.shape[1]/7)])
			ax[1].imshow(data.mean(1), origin='lower', aspect = 'auto')
			ax[2].imshow(snr_mask, origin='lower', aspect = 'auto')
			ax[3].imshow(data.mean(1), origin='lower', aspect = 'auto')
			ax[3].set_yticks(np.arange(0, data.shape[0], int(data.shape[0]/7)), np.round(orb_ph(mjd),3)[::int(data.shape[0]/7)])
			ax[4].plot(SNR_t, np.arange(data.shape[0]))
			#ax[4].plot(np.ones(data.shape[0]) * thr_t, np.arange(data.shape[0]))
			#ax[4].plot(np.ones(data.shape[0]) * (thr_t - 1), np.arange(data.shape[0]))
			for i in [2.3, 3, 3.5, 4, 4.5]: ax[4].plot(np.ones(data.shape[0]) * i, np.arange(data.shape[0]), label = 'SNR = ' + str(i))
			ax[4].plot(SNR_t[e_r], np.arange(data.shape[0])[e_r], 'o')
			ax[4].set_ylim(0, data.shape[0]-1)
			ax[4].set_xlabel('SNRs')
			ax[4].set_ylabel('MJDs')
			ax[4].legend()
			plt.suptitle('SNR : ' + str(round(SNR[0],2)) + ' , Total Flux (in mJy) : ' + str(round(Total_flux[0], 2)) + '+\-' + str(round(error_nu[0],2)) + '\n' + str(f))
			file_name = str('./plot_files/') + f[:-4] + str('.png')
			plt.savefig(file_name, dpi=200)
			print('Completed=========================================')
			plt.close()
			plt.show()
			print('eclipse fraction : ', e_r.sum()/data.shape[0])
			del data, F, mjd, SNR, Total_flux, noise_nu, error_nu, freq, on_weight_arr, e_r, SNR_t, snr_mask, dm_, m
			
		except:
			print(count + 10000)
			print('problem file : ' )
			print(f)
			pass
		print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
		count += 1
	print('#####################################################')	
	count_G += 1

import pandas

df_b3 = {'Date' : np.array(Table_SNR_Total_flux_error_Tobs_ant)[:,0], 'SNR' : np.array(Table_SNR_Total_flux_error_Tobs_ant)[:,1], 'Total flux density (mJy)' : np.array(Table_SNR_Total_flux_error_Tobs_ant)[:,2], 'error of flux density (mJy)' : np.array(Table_SNR_Total_flux_error_Tobs_ant)[:,3], 'On source time (min)' : np.array(Table_SNR_Total_flux_error_Tobs_ant)[:,4], 'Antenna :' : np.array(Table_SNR_Total_flux_error_Tobs_ant)[:,5], 'Total time (non eclipse in min)' : np.array(Table_SNR_Total_flux_error_Tobs_ant)[:,6], 'RMS in (mJy)' : np.array(Table_SNR_Total_flux_error_Tobs_ant)[:,7], 'W/P' : np.array(Table_SNR_Total_flux_error_Tobs_ant)[:,8], 'Bandwidth (MHz)' : np.array(Table_SNR_Total_flux_error_Tobs_ant)[:,9]}
my_df = pandas.DataFrame(df_b3)
my_df.to_csv('plot_files/b3_Date_SNR_Flux_error_dataframe.csv')

#############################################			in-band spectra 
files = glob.glob('*.pfd')
b34_mjd_freq_file = []
for f in files:
	freq, mjd = pfd_data(f)[1][0], pfd_data(f)[2][0]
	try:
		b34_mjd_freq_file.append([mjd, freq, str(f)])
	     
	except:
		print(f)
		pass
b34_mjd_freq_file = np.array(b34_mjd_freq_file, dtype=object)
b34_mjd_freq_file = b34_mjd_freq_file[b34_mjd_freq_file[:,0].astype(float).argsort()]

group_mjd = group(b34_mjd_freq_file)

band_func = lambda nu : 5 if 1000<=nu<=1460 else (4 if 550<=nu<=1000 else (3 if 250<=nu<=500 else (2 if 100<=nu<=250 else 'No Band info')))

thr_f, thr_t = 3, 5
count = count_G = 0
tsky_b3, tsky_b4, tsky_b5 = 23, 6, 1 # For J1544+4937
eclipse_start, eclipse_stop = 0.15, 0.32
P_b, T_asc = 0.1207729895, 56124.7701121 # For J1544+4937
orb_ph = lambda mjd_arr: ((mjd_arr - T_asc)/P_b)%1
problem_file = []
for G in group_mjd:
	print(count_G)
	if np.array(G).shape[0] == 1:
		n_ant = 24
	else:
		if sum(abs(np.diff(np.array(G)[:,1].astype(float))) >100).astype(bool):
			n_ant = 12
		else:
			n_ant = 24
	for g in G:
		
		f = g[-1]
		data, F, mjd, dm_ = pfd_data(f)
		data = data[:,0]
		if band_func(F.mean()) == 2:
			continue
			tsky = tsky_b2
		elif band_func(F.mean()) == 3:
			#continue
			tsky = tsky_b3
		elif band_func(F.mean()) == 4:
			continue
			tsky = tsky_b4
		elif band_func(F.mean()) == 5:
			continue
			tsky = tsky_b5
		print('file : ', f)
		print('n_ant : ', n_ant)
		print('Sky temp : ',tsky)
		n_chan = 10
		sigma = 5
		thres = 5
		try:
			try:
				print(count)
				e_r_0, SNR_t, snr_mask = eclipse_region(data[:,F>258], thr_f, thr_t)
				e_r_0 = filter_max_cells(e_r_0)
				print('eclipse region calculated: step 1')
				#SNR_t, snr_mask = eclipse_region(data, thr_f, thr_t)[1:]
				e_r_23 = np.logical_and(orb_ph(mjd) > eclipse_start, orb_ph(mjd) < eclipse_stop)
				e_r = np.logical_or(e_r_0, e_r_23)
				print('eclipse region calculated: step 2')
				if np.all(e_r):
					e_r = np.logical_and(e_r_0, e_r_23)
				else:
					e_r = np.logical_or(e_r_0, e_r_23)
				print('eclipse region calculated: step 2')
				SNR, Total_flux, noise, error, freq, on_weight_arr, dm = ugmrt_in_band_flux_spectra_v2(tsky, 'full', float(sigma), float(thres), 'PA', False, False, False, False, n_ant, DATA=data[~e_r][:,F>258], FREQ=F[F>258], MJD=mjd[~e_r],DM=dm_,FILE=f)
				Total_flux = Total_flux *1e3
				error *=1e3
				spec_par, snr_nu, flux_nu, noise_nu, error_nu, nu, weight, dm = ugmrt_in_band_flux_spectra_v2(tsky, int(n_chan), float(sigma), float(thres), 'PA', False, False, True, True, n_ant, DATA=data[~e_r][:,F>258], FREQ=F[F>258], MJD=mjd[~e_r],DM=dm_,FILE=f)
				del F, mjd, SNR, Total_flux, noise, error, noise_nu, error_nu, freq, on_weight_arr, dm, spec_par, snr_nu, flux_nu, nu, weight
				print('method : smoothing')
				#if not bool(Total_flux):

			except:			
				e_r_0, SNR_t, snr_mask = eclipse_region(data[:,F>258], thr_f, thr_t, False, False)
				e_r_0 = filter_max_cells(e_r_0)
				#SNR_t, snr_mask = eclipse_region(data, thr_f, thr_t)[1:]
				e_r_23 = np.logical_and(orb_ph(mjd) > eclipse_start, orb_ph(mjd) < eclipse_stop)
				e_r = np.logical_or(e_r_0, e_r_23)
				if np.all(e_r):
					e_r = np.logical_and(e_r_0, e_r_23)
				else:
					e_r = np.logical_or(e_r_0, e_r_23)
				SNR, Total_flux, noise, error, freq, on_weight_arr, dm = ugmrt_in_band_flux_spectra_v2(tsky, 'full', float(sigma), float(thres), 'PA', False, False, False, False, n_ant, DATA=data[~e_r][:,F>258], FREQ=F[F>258], MJD=mjd[~e_r],DM=dm_,FILE=f)
				Total_flux = Total_flux *1e3
				error *=1e3
				spec_par, snr_nu, flux_nu, noise_nu, error_nu, nu, weight, dm = ugmrt_in_band_flux_spectra_v2(tsky, int(n_chan), float(sigma), float(thres), 'PA', False, False, True, True, n_ant, DATA=data[~e_r][:,F>258], FREQ=F[F>258], MJD=mjd[~e_r],DM=dm_,FILE=f)
				del F, mjd, SNR, Total_flux, noise, error, noise_nu, error_nu, freq, on_weight_arr, dm, spec_par, snr_nu, flux_nu, nu, weight
				print('method : Non-smoothing')
			
			print('=============================================================')
			print('Completed')
			print('=============================================================')
		except:
			print(count + 10000)
			print('problem file : ' )
			print(f)
			problem_file.append(str(f))
			pass
		print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
		count += 1
	print('#####################################################')	
	count_G += 1




#		Plots for Sangita

P_b, T_asc = 0.1207729895, 56124.7701121

def plot_psr_images(f3, f4, P_b_, T_asc_, a):

	orb_ph = lambda mjd_arr: ((mjd_arr - T_asc_)/P_b_)%1
	
	data_3, F_3, mjd_3 = pfd_data(str(f3))
	data_3 = data_3[:,0]
	# off_bins_3 = off_region_array_from_baseline_s_data(data_3)
	d_3 = data_3.mean(1)/data_3.mean(1).mean()
	
	fig,ax = plt.subplots(nrows=2,ncols=2,figsize=(5,7), sharex =True, gridspec_kw={'height_ratios': [1, 2]})
	ax[0,0].plot(np.linspace(0,1, data_3.shape[-1]), data_3.mean((0,1))/data_3.mean((0,1)).max())
	ax[0,0].set_xlim(0,1)
	ax[0,0].set_title('Band 3 (300 - 500 MHz)',size=10,fontweight="bold")
	ax[0,0].set_ylabel('Relative amplitude')
	plt.subplots_adjust(hspace=0)
	a_ = ax[1,0].pcolormesh(np.linspace(0,1, data_3.shape[-1]), (mjd_3 - mjd_3[0])*86400/3600, d_3, vmin=a*d_3.min(), vmax=a*d_3.max(), cmap='hot')
	#plt.colorbar(a_,ax=ax[1,0])
	ax[1,0].axhline((mjd_3[np.argmin(abs(orb_ph(mjd_3) - 0.2))] - mjd_3[0])*86400/3600, color='white')
	ax[1,0].axhline((mjd_3[np.argmin(abs(orb_ph(mjd_3) - 0.3))] - mjd_3[0])*86400/3600, color='white')
	ax[1,0].set_ylabel('Time (in Hrs)')
	ax[1,0].set_xlabel('Phase')
	
	##########################################################################################################
	data_4, F_4, mjd_4 = pfd_data(str(f4))
	data_4 = data_4[:,0]
	d_4 = data_4.mean(1)/data_4.mean(1).mean()
	
	ax[0,1].plot(np.linspace(0,1, data_4.shape[-1]),data_4.mean((0,1))/data_4.mean((0,1)).max())
	ax[0,1].set_xlim(0,1)
	ax[0,1].set_title('Band 4 (550 - 750 MHz)',size=10,fontweight="bold")
	#ax[0,0].set_ylabel('Relative amplitude')
	b_ = ax[1,1].pcolormesh(np.linspace(0,1, data_4.shape[-1]), (mjd_4 - mjd_4[0])*86400/3600, d_4, vmin=a*d_4.min(), vmax=a*d_4.max(), cmap='hot')
	#plt.colorbar(b_,ax=ax[1,1])
	#for ax_ind in ax : ax.ind.label_outer()
	#ax[1,0].set_ylabel('Time (in Hrs)')
	ax[1,1].set_xlabel('Phase')
	ax[1,1].axhline((mjd_4[np.argmin(abs(orb_ph(mjd_4) - 0.2))] - mjd_4[0])*86400/3600, color='white')
	ax[1,1].axhline((mjd_4[np.argmin(abs(orb_ph(mjd_4) - 0.3))] - mjd_4[0])*86400/3600, color='white')
	plt.show()



def plot_psr_images(f3, f4, P_b_, T_asc_, a):

	orb_ph = lambda mjd_arr: ((mjd_arr - T_asc_)/P_b_)%1
	
	data_3, F_3, mjd_3 = pfd_data_b(str(f3))
	data_3 = data_3[:,0]
	unflag_f_3 = data_3.sum((0,-1)) != 0
	data_3 = data_3[:, unflag_f_3]
	off_bins_3 = off_region_array_from_actual_data(data_3)
	b3_d = data_3/(data_3[:,:,off_bins_3].mean(-1)[:,:,None]) - 1
	d_3 = b3_d.mean(1)/b3_d.mean(1).max()
	#return 
	fig,ax = plt.subplots(nrows=2,ncols=2,figsize=(5,7), sharex =True, gridspec_kw={'height_ratios': [1, 2]})
	ax[0,0].plot(np.linspace(0,1, b3_d.shape[-1]), b3_d.mean((0,1))/b3_d.mean((0,1)).max())
	ax[0,0].set_xlim(0,1)
	ax[0,0].set_title('Band 3 (300 - 500 MHz)',size=10,fontweight="bold")
	ax[0,0].set_ylabel('Relative amplitude')
	plt.subplots_adjust(hspace=0)
	a_ = ax[1,0].pcolormesh(np.linspace(0,1, b3_d.shape[-1]), (mjd_3 - mjd_3[0])*86400/3600, d_3, vmin=a*d_3.min(), vmax=a*d_3.max(), cmap='hot')
	#plt.colorbar(a_,ax=ax[1,0])
	ax[1,0].axhline((mjd_3[np.argmin(abs(orb_ph(mjd_3) - 0.2))] - mjd_3[0])*86400/3600, color='white')
	ax[1,0].axhline((mjd_3[np.argmin(abs(orb_ph(mjd_3) - 0.3))] - mjd_3[0])*86400/3600, color='white')
	ax[1,0].set_ylabel('Time (in Hrs)')
	ax[1,0].set_xlabel('Phase')
	
	##########################################################################################################
	data_4, F_4, mjd_4 = pfd_data_b(str(f4))
	data_4 = data_4[:,0]
	
	unflag_f_4 = data_4.sum((0,-1)) != 0
	data_4 = data_4[:, unflag_f_4]
	off_bins_4 = off_region_array_from_actual_data(data_4)
	b4_d = data_4/(data_4[:,:,off_bins_4].mean(-1)[:,:,None]) - 1
	d_4 = b4_d.mean(1)/b4_d.mean(1).max()

	ax[0,1].plot(np.linspace(0,1, b4_d.shape[-1]),b4_d.mean((0,1))/b4_d.mean((0,1)).max())
	ax[0,1].set_xlim(0,1)
	ax[0,1].set_title('Band 4 (550 - 750 MHz)',size=10,fontweight="bold")
	#ax[0,0].set_ylabel('Relative amplitude')
	b_ = ax[1,1].pcolormesh(np.linspace(0,1, data_4.shape[-1]), (mjd_4 - mjd_4[0])*86400/3600, d_4, vmin=a*d_4.min(), vmax=a*d_4.max(), cmap='hot')
	#plt.colorbar(b_,ax=ax[1,1])
	#for ax_ind in ax : ax.ind.label_outer()
	#ax[1,0].set_ylabel('Time (in Hrs)')
	ax[1,1].set_xlabel('Phase')
	ax[1,1].axhline((mjd_4[np.argmin(abs(orb_ph(mjd_4) - 0.2))] - mjd_4[0])*86400/3600, color='white')
	ax[1,1].axhline((mjd_4[np.argmin(abs(orb_ph(mjd_4) - 0.3))] - mjd_4[0])*86400/3600, color='white')
	plt.show()

#############################################################################################################################################################
#################				experimentation with dynamic-range

def plot_psr_images(f3, f4, P_b_, T_asc_, a, b):

	orb_ph = lambda mjd_arr: ((mjd_arr - T_asc_)/P_b_)%1
	
	data_3, F_3, mjd_3 = pfd_data(str(f3))
	data_3 = data_3[:,0]
	# off_bins_3 = off_region_array_from_baseline_s_data(data_3)
	d_3 = np.sign(data_3.mean(1))*(abs(data_3.mean(1)/data_3.mean(1).mean()))**a
	
	fig,ax = plt.subplots(nrows=2,ncols=2,figsize=(5,7), sharex =True, gridspec_kw={'height_ratios': [1, 2]})
	ax[0,0].plot(np.linspace(0,1, data_3.shape[-1]), data_3.mean((0,1))/data_3.mean((0,1)).max())
	ax[0,0].set_xlim(0,1)
	ax[0,0].set_title('Band 3 (300 - 500 MHz)',size=10,fontweight="bold")
	ax[0,0].set_ylabel('Relative amplitude')
	plt.subplots_adjust(hspace=0)
	a_ = ax[1,0].pcolormesh(np.linspace(0,1, data_3.shape[-1]), (mjd_3 - mjd_3[0])*86400/3600, d_3, vmin=d_3.min(), vmax=d_3.max(), cmap='hot')
	#plt.colorbar(a_,ax=ax[1,0])
	ax[1,0].axhline((mjd_3[np.argmin(abs(orb_ph(mjd_3) - 0.2))] - mjd_3[0])*86400/3600, color='white')
	ax[1,0].axhline((mjd_3[np.argmin(abs(orb_ph(mjd_3) - 0.3))] - mjd_3[0])*86400/3600, color='white')
	ax[1,0].set_ylabel('Time (in Hrs)')
	ax[1,0].set_xlabel('Phase')
	
	##########################################################################################################
	data_4, F_4, mjd_4 = pfd_data(str(f4))
	data_4 = data_4[:,0]
	d_4 = np.sign(data_4.mean(1))*(abs(data_4.mean(1)/data_4.mean(1).mean()))**b
	
	ax[0,1].plot(np.linspace(0,1, data_4.shape[-1]),data_4.mean((0,1))/data_4.mean((0,1)).max())
	ax[0,1].set_xlim(0,1)
	ax[0,1].set_title('Band 4 (550 - 750 MHz)',size=10,fontweight="bold")
	#ax[0,0].set_ylabel('Relative amplitude')
	b_ = ax[1,1].pcolormesh(np.linspace(0,1, data_4.shape[-1]), (mjd_4 - mjd_4[0])*86400/3600, d_4, vmin=d_4.min(), vmax=d_4.max(), cmap='hot')
	#plt.colorbar(b_,ax=ax[1,1])
	#for ax_ind in ax : ax.ind.label_outer()
	#ax[1,0].set_ylabel('Time (in Hrs)')
	ax[1,1].set_xlabel('Phase')
	ax[1,1].axhline((mjd_4[np.argmin(abs(orb_ph(mjd_4) - 0.2))] - mjd_4[0])*86400/3600, color='white')
	ax[1,1].axhline((mjd_4[np.argmin(abs(orb_ph(mjd_4) - 0.3))] - mjd_4[0])*86400/3600, color='white')
	ax[1,1].tick_params(axis='both', labelsize=22)
	ax[1,0].tick_params(axis='both', labelsize=22)
	plt.show()

#################				experimentation with dynamic-range 2 d plot

def plot_psr_images(f3, P_b_, T_asc_, a):

	#orb_ph = lambda mjd_arr: ((mjd_arr - T_asc_)/P_b_)%1
	robust_std = lambda x_ : 1.4826*np.median(abs(x_.flatten() - np.median(x_.flatten())))
	data_3, F_3, mjd_3, dm_ = pfd_data(str(f3))
	data_3 = data_3[:,0]
	# off_bins_3 = off_region_array_from_baseline_s_data(data_3)
	d_3 = np.sign(data_3.mean(1))*(abs(data_3.mean(1)/data_3.mean(1).mean()))**a
	
	fig,ax = plt.subplots(nrows=2, figsize=(5,7), sharex =True, gridspec_kw={'height_ratios': [1, 2]})
	ax[0].plot(np.linspace(0,1, data_3.shape[-1]), data_3.mean((0,1))/data_3.mean((0,1)).max())
	ax[0].set_xlim(0,1)
	#ax[0].set_title('Band 3 (300 - 500 MHz)',size=10,fontweight="bold")
	ax[0].set_ylabel('Relative amplitude')
	plt.subplots_adjust(hspace=0)
	#a_ = ax[1].pcolormesh(np.linspace(0,1, data_3.shape[-1]), (mjd_3 - mjd_3[0])*24, d_3, vmin=d_3.mean() - 3*robust_std(d_3), vmax=d_3.mean() + 3*robust_std(d_3), cmap='hot')
	a_ = ax[1,0].pcolormesh(np.linspace(0,1, data_3.shape[-1]), (mjd_3 - mjd_3[0])*86400/3600, d_3, vmin=d_3.min(), vmax=d_3.max(), cmap='hot')
	#plt.colorbar(a_,ax=ax[1,0])
	#ax[1].axhline((mjd_3[np.argmin(abs(orb_ph(mjd_3) - 0.2))] - mjd_3[0])*86400/3600, color='white')
	#ax[1].axhline((mjd_3[np.argmin(abs(orb_ph(mjd_3) - 0.3))] - mjd_3[0])*86400/3600, color='white')
	ax[1].set_ylabel('Time (in Hrs)')
	ax[1].set_xlabel('Phase')
	
	##########################################################################################################
	
	#plt.show()


def plot_psr_images_1(f3, a):

	#orb_ph = lambda mjd_arr: ((mjd_arr - T_asc_)/P_b_)%1
	robust_std = lambda x_ : 1.4826*np.median(abs(x_.flatten() - np.median(x_.flatten())))
	data_3, F_3, mjd_3, dm_ = pfd_data(str(f3))
	data_3 = data_3[:,0]
	# off_bins_3 = off_region_array_from_baseline_s_data(data_3)
	d_3 = np.sign(data_3.mean(1) - data_3.mean(1).mean())*(abs(data_3.mean(1)/data_3.mean(1).mean() - 1))**a
	
	fig,ax = plt.subplots(nrows=2, figsize=(5,17), sharex =True, gridspec_kw={'height_ratios': [1, 2]})
	ax[0].plot(np.linspace(0,1, data_3.shape[-1]), data_3.mean((0,1))/data_3.mean((0,1)).max())
	ax[0].set_xlim(0,1)
	#ax[0].set_title('Band 3 (300 - 500 MHz)',size=10,fontweight="bold")
	ax[0].set_ylabel('Relative amplitude')
	plt.subplots_adjust(hspace=0)
	#a_ = ax[1].pcolormesh(np.linspace(0,1, data_3.shape[-1]), (mjd_3 - mjd_3[0])*24, d_3, vmin=d_3.mean() - 3*robust_std(d_3), vmax=d_3.mean() + 3*robust_std(d_3), cmap='hot')
	a_ = ax[1].pcolormesh(np.linspace(0,1, data_3.shape[-1]), (mjd_3 - mjd_3[0])*86400/3600, d_3, vmin=d_3.mean() - 2*robust_std(d_3), vmax=d_3.mean() + 2*robust_std(d_3), cmap='hot')
	#plt.colorbar(a_,ax=ax[1,0])
	#ax[1].axhline((mjd_3[np.argmin(abs(orb_ph(mjd_3) - 0.2))] - mjd_3[0])*86400/3600, color='white')
	#ax[1].axhline((mjd_3[np.argmin(abs(orb_ph(mjd_3) - 0.3))] - mjd_3[0])*86400/3600, color='white')
	ax[1].set_ylabel('Time (in Hrs)')
	ax[1].set_xlabel('Phase')
	
	##########################################################################################################
	
	#plt.show()


