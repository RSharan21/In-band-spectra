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
import pandas


def pfd_data(pfd):
    var=psrchive.Archive_load(pfd)
    var.dedisperse()
    var.centre_max_bin()
    var.remove_baseline()
    return var.get_data().astype(np.float64), var.get_frequencies(), var.get_mjds(), var.get_dispersion_measure()



def pfd_data_b(pfd):
    var=psrchive.Archive_load(pfd)
    var.dedisperse()
    var.centre_max_bin()
    #var.remove_baseline()
    return var.get_data().astype(np.float64), var.get_frequencies(), var.get_mjds() , var.get_dispersion_measure()



robust_std = lambda x_ : 1.4826*np.median(abs(x_.flatten() - np.median(x_.flatten())))
# np.seterr(divide='ignore', invalid='ignore')


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

#############################################################################################################################################

def mask_with_prim_comp_flags_v2(array_, flags_info, thres_, n_chan_, prim_comp_ = True):
	mask = []
	chunk_ind = 0
	chunk_ind_arr_ = []
	while chunk_ind <flags_info.shape[0]:
		if flags_info[chunk_ind]:
			chunk_ind_arr_.append(chunk_ind)
			support = np.zeros(array_.shape[1],dtype=bool)
			support[chunk_ind: chunk_ind + n_chan_] = True
			chunk_ind += n_chan_
			mask_per_chan = np.nansum(array_[:,flags_info*support],axis=(0,1))/(np.nanstd(array_[:,flags_info*support], axis=(0,1)) * np.sqrt(array_[:,flags_info*support].shape[0]*array_[:,flags_info*support].shape[1]) ) > thres_
			mask.append(mask_per_chan)
		else:
			chunk_ind_arr_.append(chunk_ind)
			chunk_ind += 1
			continue

	mask = filter_clean_isolated_cells(mask)	# filter the isolated points
	if prim_comp_:
		ext_mask = ndimage.binary_fill_holes(mask, structure=np.array([[0,1,0]*3]).reshape(3,3))	# connects the mask vertically
		id_regions, num_ids = ndimage.label(ext_mask, structure=np.ones((3,3)) )	# id_regions, num_ids: contains different frequency dependent component info
		prim_comp_bool_array = np.array([np.any(np.where(id_regions == i)[-1] == id_regions.shape[-1]//2) for i in range(num_ids + 1) ])	# checks for the component intersecting the mid-point (horizontally speaking)(only if the main component is shifted to central phase bins) of the mask 
		prim_comp_bool_array[0] = False
		prim_comp_region = prim_comp_bool_array[id_regions]
		prim_comp_mask = np.logical_and(prim_comp_region, mask)
		
		return mask, prim_comp_mask, chunk_ind_arr_#, flag_ind_
	else:
		return mask, chunk_ind_arr_#, flag_ind_
#############################################################################################################################################
# Incomplete
def mask_with_prim_comp_flags_v3(array_, flags_info, thres_, n_chan_, prim_comp_ = True):
	mask = []
	chunk_ind = 0
	chunk_ind_arr_ = []
	snr_array_ = []
	while chunk_ind <flags_info.shape[0]:
		if flags_info[chunk_ind]:
			chunk_ind_arr_.append(chunk_ind)
			support = np.zeros(array_.shape[1],dtype=bool)
			support[chunk_ind: chunk_ind + n_chan_] = True
			chunk_ind += n_chan_
			snr_array_per_chunk = np.nansum(array_[:,flags_info*support],axis=(0,1))/(np.nanstd(array_[:,flags_info*support], axis=(0,1)) * np.sqrt(array_[:,flags_info*support].shape[0] * array_[:,flags_info*support].shape[1]) )
			snr_array_per_chunk = gf1d(snr_array_per_chunk,1)
			snr_array_.append(snr_array_per_chunk)
			mask_per_chan = snr_array_per_chunk > thres_
			mask.append(mask_per_chan)
		else:
			chunk_ind_arr_.append(chunk_ind)
			chunk_ind += 1
			continue

	mask = filter_clean_isolated_cells(mask)	# filter the isolated points

	if prim_comp_:
		ext_mask = ndimage.binary_fill_holes(mask, structure=np.array([[0,1,0]*3]).reshape(3,3))	# connects the mask vertically
		id_regions, num_ids = ndimage.label(ext_mask, structure=np.ones((3,3)) )	# id_regions, num_ids: contains different frequency dependent component info
		prim_comp_bool_array = np.array([np.any(np.where(id_regions == i)[-1] == id_regions.shape[-1]//2) for i in range(num_ids + 1) ])	# checks for the component intersecting the mid-point (horizontally speaking)(only if the main component is shifted to central phase bins) of the mask 
		prim_comp_bool_array[0] = False
		prim_comp_region = prim_comp_bool_array[id_regions]
		prim_comp_mask = np.logical_and(prim_comp_region, mask)
		
		return mask, prim_comp_mask, chunk_ind_arr_
	else:
		return mask, chunk_ind_arr_
#############################################################################################################################################
# Incomplete
def mask_with_prim_comp_flags_v3_smoothened(array_, flags_info, thres_, n_chan_, prim_comp_ = True):
	mask = []
	chunk_ind = 0
	chunk_ind_arr_ = []
	snr_array_ = []
	while chunk_ind <flags_info.shape[0]:
		if flags_info[chunk_ind]:
			chunk_ind_arr_.append(chunk_ind)
			support = np.zeros(array_.shape[1],dtype=bool)
			support[chunk_ind: chunk_ind + n_chan_] = True
			chunk_ind += n_chan_
			snr_array_per_chunk = np.nansum(array_[:,flags_info*support],axis=(0,1))/(np.nanstd(array_[:,flags_info*support], axis=(0,1)) * np.sqrt(array_[:,flags_info*support].shape[0] * array_[:,flags_info*support].shape[1]) )
			snr_array_per_chunk = np.convolve(snr_array_per_chunk,np.ones(2)/2,mode='same')
			snr_array_.append(snr_array_per_chunk)
			mask_per_chan = snr_array_per_chunk > thres_
			mask.append(mask_per_chan)
		else:
			chunk_ind_arr_.append(chunk_ind)
			chunk_ind += 1
			continue

	mask = filter_clean_isolated_cells(mask)	# filter the isolated points

	if prim_comp_:
		ext_mask = ndimage.binary_fill_holes(mask, structure=np.array([[0,1,0]*3]).reshape(3,3))	# connects the mask vertically
		id_regions, num_ids = ndimage.label(ext_mask, structure=np.ones((3,3)) )	# id_regions, num_ids: contains different frequency dependent component info
		prim_comp_bool_array = np.array([np.any(np.where(id_regions == i)[-1] == id_regions.shape[-1]//2) for i in range(num_ids + 1) ])	# checks for the component intersecting the mid-point (horizontally speaking)(only if the main component is shifted to central phase bins) of the mask 
		prim_comp_bool_array[0] = False
		prim_comp_region = prim_comp_bool_array[id_regions]
		prim_comp_mask = np.logical_and(prim_comp_region, mask)
		
		return mask, prim_comp_mask, chunk_ind_arr_
	else:
		return mask, chunk_ind_arr_
#############################################################################################################################################
#############################################################################################################################################
# Incomplete
def mask_with_prim_comp_flags_v3_gaussian_1d_smoothened(array_, flags_info, thres_, n_chan_, prim_comp_ = True):
	mask = []
	chunk_ind = 0
	chunk_ind_arr_ = []
	snr_array_ = []
	while chunk_ind <flags_info.shape[0]:
		if flags_info[chunk_ind]:
			chunk_ind_arr_.append(chunk_ind)
			support = np.zeros(array_.shape[1],dtype=bool)
			support[chunk_ind: chunk_ind + n_chan_] = True
			chunk_ind += n_chan_
			snr_array_per_chunk = np.nansum(array_[:,flags_info*support],axis=(0,1))/(np.nanstd(array_[:,flags_info*support], axis=(0,1)) * np.sqrt(array_[:,flags_info*support].shape[0] * array_[:,flags_info*support].shape[1]) )
			snr_array_per_chunk = gf1d(snr_array_per_chunk,1)
			snr_array_.append(snr_array_per_chunk)
			mask_per_chan = snr_array_per_chunk > thres_
			mask.append(mask_per_chan)
		else:
			chunk_ind_arr_.append(chunk_ind)
			chunk_ind += 1
			continue

	mask = filter_clean_isolated_cells(mask)	# filter the isolated points

	if prim_comp_:
		ext_mask = ndimage.binary_fill_holes(mask, structure=np.array([[0,1,0]*3]).reshape(3,3))	# connects the mask vertically
		id_regions, num_ids = ndimage.label(ext_mask, structure=np.ones((3,3)) )	# id_regions, num_ids: contains different frequency dependent component info
		prim_comp_bool_array = np.array([np.any(np.where(id_regions == i)[-1] == id_regions.shape[-1]//2) for i in range(num_ids + 1) ])	# checks for the component intersecting the mid-point (horizontally speaking)(only if the main component is shifted to central phase bins) of the mask 
		prim_comp_bool_array[0] = False
		prim_comp_region = prim_comp_bool_array[id_regions]
		prim_comp_mask = np.logical_and(prim_comp_region, mask)
		
		return mask, prim_comp_mask, chunk_ind_arr_
	else:
		return mask, chunk_ind_arr_

#############################################################################################################################################
#############################################################################################################################################
# Incomplete
def mask_with_prim_comp_flags_v3_savgol_1d_smoothened(array_, flags_info, thres_, n_chan_, prim_comp_ = True):
	mask = []
	chunk_ind = 0
	chunk_ind_arr_ = []
	snr_array_ = []
	while chunk_ind <flags_info.shape[0]:
		if flags_info[chunk_ind]:
			chunk_ind_arr_.append(chunk_ind)
			support = np.zeros(array_.shape[1],dtype=bool)
			support[chunk_ind: chunk_ind + n_chan_] = True
			chunk_ind += n_chan_
			snr_array_per_chunk = np.nansum(array_[:,flags_info*support],axis=(0,1))/(np.nanstd(array_[:,flags_info*support], axis=(0,1)) * np.sqrt(array_[:,flags_info*support].shape[0] * array_[:,flags_info*support].shape[1]) )
			snr_array_per_chunk = sv(snr_array_per_chunk, 3, 2)
			snr_array_.append(snr_array_per_chunk)
			mask_per_chan = snr_array_per_chunk > thres_
			mask.append(mask_per_chan)
		else:
			chunk_ind_arr_.append(chunk_ind)
			chunk_ind += 1
			continue
	
	mask = filter_clean_isolated_cells(mask)	# filter the isolated points
	#return snr_array_, mask
	if prim_comp_:
		ext_mask = ndimage.binary_fill_holes(mask, structure=np.array([[0,1,0]*3]).reshape(3,3))	# connects the mask vertically
		id_regions, num_ids = ndimage.label(ext_mask, structure=np.ones((3,3)) )	# id_regions, num_ids: contains different frequency dependent component info
		prim_comp_bool_array = np.array([np.any(np.where(id_regions == i)[-1] == id_regions.shape[-1]//2) for i in range(num_ids + 1) ])	# checks for the component intersecting the mid-point (horizontally speaking)(only if the main component is shifted to central phase bins) of the mask 
		prim_comp_bool_array[0] = False
		prim_comp_region = prim_comp_bool_array[id_regions]
		prim_comp_mask = np.logical_and(prim_comp_region, mask)
		
		return mask, prim_comp_mask, chunk_ind_arr_
	else:
		return mask, chunk_ind_arr_

#############################################################################################################################################
#############################################################################################################################################
# Incomplete
def mask_with_prim_comp_flags_v3_gaussian_smoothened(array_, flags_info, thres_, n_chan_, prim_comp_ = True):
	mask = []
	chunk_ind = 0
	chunk_ind_arr_ = []
	snr_array_ = []
	while chunk_ind <flags_info.shape[0]:
		if flags_info[chunk_ind]:
			chunk_ind_arr_.append(chunk_ind)
			support = np.zeros(array_.shape[1],dtype=bool)
			support[chunk_ind: chunk_ind + n_chan_] = True
			chunk_ind += n_chan_
			snr_array_per_chunk = np.nansum(array_[:,flags_info*support],axis=(0,1))/(np.nanstd(array_[:,flags_info*support], axis=(0,1)) * np.sqrt(array_[:,flags_info*support].shape[0] * array_[:,flags_info*support].shape[1]) )
			snr_array_.append(snr_array_per_chunk)
		else:
			chunk_ind_arr_.append(chunk_ind)
			chunk_ind += 1
			continue
	snr_array_ = gf(snr_array_,1)
	mask = snr_array_ > thres_
	mask = filter_clean_isolated_cells(mask)	# filter the isolated points

	if prim_comp_:
		ext_mask = ndimage.binary_fill_holes(mask, structure=np.array([[0,1,0]*3]).reshape(3,3))	# connects the mask vertically
		id_regions, num_ids = ndimage.label(ext_mask, structure=np.ones((3,3)) )	# id_regions, num_ids: contains different frequency dependent component info
		prim_comp_bool_array = np.array([np.any(np.where(id_regions == i)[-1] == id_regions.shape[-1]//2) for i in range(num_ids + 1) ])	# checks for the component intersecting the mid-point (horizontally speaking)(only if the main component is shifted to central phase bins) of the mask 
		prim_comp_bool_array[0] = False
		prim_comp_region = prim_comp_bool_array[id_regions]
		prim_comp_mask = np.logical_and(prim_comp_region, mask)
		
		return mask, prim_comp_mask, chunk_ind_arr_
	else:
		return mask, chunk_ind_arr_
#############################################################################################################################################
#############################################################################################################################################
# Incomplete
def mask_with_prim_comp_flags_v3_smoothened_and_1d_gaussian_smoothened(array_, flags_info, thres_, n_chan_, prim_comp_ = True):
	mask = []
	chunk_ind = 0
	chunk_ind_arr_ = []
	snr_array_ = []
	while chunk_ind <flags_info.shape[0]:
		if flags_info[chunk_ind]:
			chunk_ind_arr_.append(chunk_ind)
			support = np.zeros(array_.shape[1],dtype=bool)
			support[chunk_ind: chunk_ind + n_chan_] = True
			chunk_ind += n_chan_
			snr_array_per_chunk = np.nansum(array_[:,flags_info*support],axis=(0,1))/(np.nanstd(array_[:,flags_info*support], axis=(0,1)) * np.sqrt(array_[:,flags_info*support].shape[0] * array_[:,flags_info*support].shape[1]) )
			snr_array_per_chunk_1 = np.convolve(snr_array_per_chunk,np.ones(2)/2,mode='same')
			snr_array_per_chunk_2 = gf1d(snr_array_per_chunk,1)
			#snr_array_.append(snr_array_per_chunk)
			mask_per_chan = snr_array_per_chunk > thres_
			mask.append(mask_per_chan)
		else:
			chunk_ind_arr_.append(chunk_ind)
			chunk_ind += 1
			continue

	mask = filter_clean_isolated_cells(mask)	# filter the isolated points

	if prim_comp_:
		ext_mask = ndimage.binary_fill_holes(mask, structure=np.array([[0,1,0]*3]).reshape(3,3))	# connects the mask vertically
		id_regions, num_ids = ndimage.label(ext_mask, structure=np.ones((3,3)) )	# id_regions, num_ids: contains different frequency dependent component info
		prim_comp_bool_array = np.array([np.any(np.where(id_regions == i)[-1] == id_regions.shape[-1]//2) for i in range(num_ids + 1) ])	# checks for the component intersecting the mid-point (horizontally speaking)(only if the main component is shifted to central phase bins) of the mask 
		prim_comp_bool_array[0] = False
		prim_comp_region = prim_comp_bool_array[id_regions]
		prim_comp_mask = np.logical_and(prim_comp_region, mask)
		
		return mask, prim_comp_mask, chunk_ind_arr_
	else:
		return mask, chunk_ind_arr_
#############################################################################################################################################

#############################################################################################################################################
'''
in v6 I am experimenting with smoothing before taking the binary mask. It mainly done with the intent to remove the intrinsic pulsar's flux density variation along time and frequency. For some this could be a crime !!!

This version isn't implimented correctly, Use v7 instead
'''

def mask_with_prim_comp_flags_v6(array_, flags_info, thres_, n_chan_, prim_comp_ = True):
	mask = []
	chunk_ind = 0
	chunk_ind_arr_ = []
	snr_array_ = []
	while chunk_ind <flags_info.shape[0]:
		if flags_info[chunk_ind]:
			chunk_ind_arr_.append(chunk_ind)
			support = np.zeros(array_.shape[1],dtype=bool)
			support[chunk_ind: chunk_ind + n_chan_] = True
			chunk_ind += n_chan_
			chunk_data_3d = array_[:,flags_info*support]
			std_on_freq_axis = np.nan_to_num(np.sqrt(1/np.nansum(np.nan_to_num(1/chunk_data_3d.var(0),nan=np.nan, neginf=np.nan, posinf=np.nan), axis=0)),nan=0,neginf=0,posinf=0)
			#mean_on_chunk_axis = chunk_data_3d.mean(0)
			#snr_array_per_chunk_0 = np.nan_to_num((chunk_data_3d - mean_on_chunk_axis[None, :, :])/std_on_chunk_axis[None, :, :])
			#snr_array_per_chunk_0 = np.nansum(chunk_data_3d, axis=(0,1))/(std_on_freq_axis * np.sqrt(array_[:,flags_info*support].shape[0] ) )
			snr_array_per_chunk_0 = np.nansum(chunk_data_3d.mean(1), axis=0)/(std_on_freq_axis* np.sqrt(array_[:,flags_info*support].shape[0] ) )
			#snr_array_per_chunk_0 = np.nansum(array_[:,flags_info*support],axis=(0,1))/(np.nanstd(array_[:,flags_info*support], axis=(0,1)) * np.sqrt(array_[:,flags_info*support].shape[0] * array_[:,flags_info*support].shape[1]) )
			snr_array_.append(snr_array_per_chunk_0)
			mask_per_chan = snr_array_per_chunk_0 > thres_
			#mask_per_chan = np.logical_or(snr_array_per_chunk_1 > thres_, snr_array_per_chunk_2 > thres_)
			mask.append(mask_per_chan)
		else:
			chunk_ind_arr_.append(chunk_ind)
			chunk_ind += 1
			continue

	mask = filter_clean_isolated_cells(mask)	# filter the isolated points

	if prim_comp_:
		ext_mask = ndimage.binary_fill_holes(mask, structure=np.array([[0,1,0]*3]).reshape(3,3))	# connects the mask vertically
		id_regions, num_ids = ndimage.label(ext_mask, structure=np.ones((3,3)) )	# id_regions, num_ids: contains different frequency dependent component info
		prim_comp_bool_array = np.array([np.any(np.where(id_regions == i)[-1] == id_regions.shape[-1]//2) for i in range(num_ids + 1) ])	# checks for the component intersecting the mid-point (horizontally speaking)(only if the main component is shifted to central phase bins) of the mask 
		prim_comp_bool_array[0] = False
		prim_comp_region = prim_comp_bool_array[id_regions]
		prim_comp_mask = np.logical_and(prim_comp_region, mask)
		
		return mask, prim_comp_mask, chunk_ind_arr_
	else:
		return mask, chunk_ind_arr_
#############################################################################################################################################
#############################################################################################################################################
'''
in v6 I am experimenting with smoothing before taking the binary mask. It mainly done with the intent to remove the intrinsic pulsar's flux density variation along time and frequency. For some this could be a crime !!!
'''

def mask_with_prim_comp_flags_v7(array_, flags_info, thres_, n_chan_, prim_comp_ = True):
	mask = []
	chunk_ind = 0
	chunk_ind_arr_ = []
	snr_array_ = []
	while chunk_ind <flags_info.shape[0]:
		if flags_info[chunk_ind]:
			chunk_ind_arr_.append(chunk_ind)
			support = np.zeros(array_.shape[1],dtype=bool)
			support[chunk_ind: chunk_ind + n_chan_] = True
			chunk_ind += n_chan_
			chunk_data_3d = array_[:,flags_info*support]
			std_on_freq_axis = np.nan_to_num(np.sqrt(1/np.nansum(np.nan_to_num(chunk_data_3d.shape[0]/chunk_data_3d.var(0),nan=np.nan, neginf=np.nan, posinf=np.nan), axis=0)),nan=0,neginf=0,posinf=0)
			snr_array_per_chunk_0 = np.nansum((chunk_data_3d.mean(0) * chunk_data_3d.shape[0])/chunk_data_3d.var(0), axis=0) * std_on_freq_axis
			snr_array_.append(snr_array_per_chunk_0)
			mask_per_chan = snr_array_per_chunk_0 > thres_
			mask.append(mask_per_chan)
		else:
			chunk_ind_arr_.append(chunk_ind)
			chunk_ind += 1
			continue

	mask = filter_clean_isolated_cells(mask)	# filter the isolated points
	if prim_comp_:
		ext_mask = ndimage.binary_fill_holes(mask, structure=np.array([[0,1,0]*3]).reshape(3,3))	# connects the mask vertically
		id_regions, num_ids = ndimage.label(ext_mask, structure=np.ones((3,3)) )	# id_regions, num_ids: contains different frequency dependent component info
		prim_comp_bool_array = np.array([np.any(np.where(id_regions == i)[-1] == id_regions.shape[-1]//2) for i in range(num_ids + 1) ])	# checks for the component intersecting the mid-point (horizontally speaking)(only if the main component is shifted to central phase bins) of the mask 
		prim_comp_bool_array[0] = False
		prim_comp_region = prim_comp_bool_array[id_regions]
		prim_comp_mask = np.logical_and(prim_comp_region, mask)
		
		return mask, prim_comp_mask, chunk_ind_arr_
	else:
		return mask, chunk_ind_arr_
#############################################################################################################################################
########################################################################################################################################

# arr_3d should be actual data (without baseline subtraction)

def off_region_array_from_actual_data(arr_3d):
	
	box_car_len = int(np.round(0.15*arr_3d.shape[-1])) # int(0.15*arr_3d.shape[-1])
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



def sefd(nu_, nu_c, time_s_, Tsky_, n_ant_ = 23, beam_ ='PA', delta_chan_ = 1, n_pol_ = 2):
	Tsky_ = Tsky_ * (nu_c/nu_)**2.55
	Tdef = 22 * (408/nu_)**2.55
	g_by_Tsys, g_ant = bandpass_g_by_tsys(nu_)
	if beam_.lower() == str('PA').lower():
		ant_pow_ind = 2
	elif beam_.lower() == str('IA').lower():
		ant_pow_ind = 1
	
	return ( (1/g_by_Tsys) - (Tdef/g_ant) + (Tsky_/g_ant) ) /np.sqrt( (n_ant_**ant_pow_ind) * delta_chan_ *1e6 * time_s_ * n_pol_)
######################################################################################################################

###################		This part of code contains code required for fitting multiple straight lines of the power law plots.
###################		For this, as of NOW, initial guess (calculated as per number of components required to fit) is being
###################		used to fit multiple compenents. Optimal number of components are found by using AICC (i.e. the AICC
###################		with least value corresponds to the optimal number of component).


def initial_guess_n_bounds_func(n_, x_, y_):
	# n_ is the number of broken power law (1,2,3,.....)
	# x, y = log(freq), log(Flux_density)
	
	#x_chunks = [np.array([x_val for x_val in x_ if x_val >= breaking_points[i] and x_val < breaking_points[i+1]]) for i in range(len(breaking_points)-1)]

	while 2*n_ > len(x_):
		n_ -= 1
	'''
	if len(x_[:: len(x_)//n_]) == n_ + 1:
		breaking_points = np.append( x_[:: len(x_)//n_][:-1], x_[-1])
	elif len(x_[:: len(x_)//n_]) == n_:
		breaking_points = np.append( x_[:: len(x_)//n_], x_[-1])
	'''
	breaking_points = []
	for j in range(n_+1):
		breaking_points.append(x_[np.argmin(abs(x_- ( min(x_) + (j*(max(x_)-min(x_))/n_) ) ))])
	
	init_guess = list(breaking_points[1:-1])
	for i in range(n_):
		param_init, _ = curve_fit(lambda x_vals_, *params_0_: fit_func_st_line(x_vals_, 1, *params_0_), x_[np.logical_and(breaking_points[i]<=x_, x_<breaking_points[i+1])],y_[np.logical_and(breaking_points[i]<=x_, x_<breaking_points[i+1])], p0=[-1,1], bounds=[[-np.inf,-np.inf], [np.inf, np.inf]], maxfev = 5000)
		#slope = 
		init_guess.append(param_init[0])
	init_guess.append(y_.mean())
	#############################################################################
	lower_bounds = [x_.min()]*(n_ - 1) + [-np.inf]*(n_ + 1)
	return init_guess, [[x_.min()]*(n_ - 1) + [-np.inf]*(n_ + 1), [x_.max()]*(n_ - 1) + [np.inf]*(n_ + 1)]


##########################################################################################################################
#### This function gives the broken straight line function
#### Inputs	: array of x-axis (1d array of log of frequency)
####		: list of break point (n - 1 points, excluding the end points) ( within xvals.min() to xvals.max())
####		: list of alphas to be used in those broken power law components (n points)
####		: const DC, which by default is kept to 1 (0 in the frequency
#### Output: array of y axis corresponding to xvals

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
###########################################################################################################################################


def initial_guess_n_bounds_pow_law_func(n_, x_, y_):
	# n_ is the number of broken power law (1,2,3,.....)
	# x, y = log(freq), log(Flux_density)
	
	#x_chunks = [np.array([x_val for x_val in x_ if x_val >= breaking_points[i] and x_val < breaking_points[i+1]]) for i in range(len(breaking_points)-1)]

	x_ = x_[~np.isnan(x_)]
	y_ = y_[~np.isnan(y_)]
	while 2*n_ > len(x_):
		n_ -= 1
	'''
	if len(x_[:: len(x_)//n_]) == n_ + 1:
		breaking_points = np.append( x_[:: len(x_)//n_][:-1], x_[-1])
	elif len(x_[:: len(x_)//n_]) == n_:
		breaking_points = np.append( x_[:: len(x_)//n_], x_[-1])
	'''
	breaking_points = []
	for j in range(n_+1):
		breaking_points.append(x_[np.argmin(abs(x_- ( min(x_) + (j*(max(x_)-min(x_))/n_) ) ))])
	
	init_guess = list(breaking_points[1:-1])
	for i in range(n_):
		range_ = np.logical_and(breaking_points[i]<=x_, x_<breaking_points[i+1])
		param_init, _ = curve_fit(lambda x_vals_, *params_0_: fit_func_power_law(x_vals_, 1, *params_0_), x_[range_],y_[range_], p0=[-1,1], bounds=[[-np.inf,0], [np.inf, np.inf]], maxfev = 5000)
		if i==0:
			c0 = param_init[-1]
		init_guess.append(param_init[0])
	init_guess.append(c0)
	#############################################################################
	lower_bounds = [x_.min()]*(n_ - 1) + [-np.inf]*(n_ + 1)
	return init_guess, [[x_.min()]*(n_ - 1) + [-np.inf]*n_ + [0], [x_.max()]*(n_ - 1) + [np.inf]*(n_ + 1)]



def bkn_pow_law_func(x_, breaks, alphas, const = 1):
	if not breaks:
		y_ = const * (x_**alphas[-1])
		return y_
	ind = np.where(np.asarray(breaks) >= x_)[0]
	if ind.size == 0:
		ind = len(breaks)
	else:
		ind = ind[0]
	for i in range(ind):
		const *= (breaks[i]**(alphas[i] - alphas[i + 1]))
	y_ = const*(x_**alphas[ind])
	return y_



def fit_func_power_law(x_, n_, *args):
	# n_ is the number of broken power law
	breaks = list(args[: n_ - 1])
	alphas = list(args[n_ - 1 : -1])
	const = args[-1]
	#print('breaks : ', breaks,' alphas : ', alphas,' const : ',const)
	try:
		len(x_)
		return np.array([bkn_pow_law_func(i, breaks, alphas, const) for i in x_])
	except:
		
		return bkn_pow_law_func(x_, breaks, alphas, const)


########################################################################################################################
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

######################################################################################################################
####		This function extracts inofrmation (Mode: 'IA'/'PA'; Ant_no; Frequency of obs) from the log file once the "required" text file is supplied.
####		



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

		if re.search(pattern_info, section_iter):
			beam_dict = eval(re.findall(pattern_info, section_iter)[0])
		else:
			continue

		if not any(Select_mode_only):
			return eval(re.findall(pattern_info, section_iter)[0])

		for mode_only in Select_mode_only:
			for beam_mode_val in beam_dict.values():
				if band_func(float(freq_)) != band_func(float(beam_mode_val['Freq'])):
					continue
				if beam_mode_val['Mode'].lower() == mode_only.lower():
					try:
						return beam_mode_val
					except:
						pass

##############################################################################################################################################################################

# This version mainly does (data - data_mean[off pulse]) / data_std[off pulse]. if mask_with_prim_comp_flags_v6 is used then the definition of standard deviation for a chunk in frequecy changes (1/np.sqrt(sum_along_freq(1/variance_along_time)))



def ugmrt_in_band_flux_spectra_v1(Tsky, n_chan = 10, sigma = 10, thres = 3, beam_ = 'PA', primary_comp = True, show_plots = True, allow_fit = False, save_plot = False, n_ant = 23, **file_data):
	bandpass_correction = True
	robust_std = lambda x_ : 1.4826*np.median(abs(x_.flatten() - np.median(x_.flatten())))
	if len(file_data.keys()) >2:
		data_, freq_, mjds_ = file_data['DATA'], file_data['FREQ'], file_data['MJD']
		unflag_freq = np.sum(data_, axis = (0,2)) != 0		# for removing fully flagged channels from the data_
		
		if unflag_freq.shape[0] == unflag_freq.sum():
			unflag_freq = abs(np.median(data_, axis=(0,-1))/np.median(data_, axis=(0,-1)).max()) >1e-1
		
		unflag_time = np.sum(data_, axis = (1,2)) != 0
		
		# beam_ = True

	else:
		file_ = file_data['FILE']
		
		if bandpass_correction:
			if bool(re.search('gpt',file_)):
				data_, freq_, mjds_, dm_= pfd_data_b(file_)
				data_ = data_[:,0]
				global_off_bins_bool_arr = off_region_array_from_actual_data(data_)
				data_ = np.nan_to_num((data_ - np.nanmean(data_[:,:,global_off_bins_bool_arr], axis=-1,keepdims=True))/np.nanstd(data_[:,:,global_off_bins_bool_arr], axis=-1,keepdims=True), nan=0, neginf=0,posinf=0)
				d_mean_ = data_.mean(-1)
				unflag_freq = np.sum(data_, axis = (0,2)) != 0		# for removing fully flagged channels from the data_
				unflag_freq = np.logical_and(d_mean_.std(0) > (np.median(d_mean_.std(0))*0.1) , d_mean_.std(0) < (np.median(d_mean_.std(0)) +15*robust_std(d_mean_.std(0)) ) )
				unflag_time = np.sum(data_, axis = (1,2)) != 0
				#data_ = unflag_freq[None, :, None]*data_
				
			else:
				data_, freq_, mjds_, dm_= pfd_data_b(file_)
				data_ = data_[:,0]
				global_off_bins_bool_arr = off_region_array_from_actual_data(data_)
				#####			Bandshape correction from the data
				
				data_ = np.nan_to_num((data_ - np.nanmean(data_[:,:,global_off_bins_bool_arr], axis=-1,keepdims=True))/np.nanstd(data_[:,:,global_off_bins_bool_arr], axis=-1,keepdims=True), nan=0, neginf=0,posinf=0)
				d_mean_ = data_.mean(-1)
				unflag_freq = np.sum(data_, axis = (0,2)) != 0		# for removing fully flagged channels from the data_
				unflag_freq = np.logical_and(d_mean_.std(0) > (np.median(d_mean_.std(0))*0.1) , d_mean_.std(0) < (np.median(d_mean_.std(0)) +15*robust_std(d_mean_.std(0)) ) )
				#unflag_freq = np.logical_and(unflag_freq, d_mean_.std(0) < (d_mean_.std(0).mean() +15*robust_std(d_mean_.std(0)) ) )
				unflag_time = np.sum(data_, axis = (1,2)) != 0
				#data_ = unflag_freq[None, :, None]*data_
				#data_ = np.nan_to_num(data_/(np.nanmean(data_[:,:,global_off_bins_bool_arr], axis=-1)[:,:,None]) - 1, nan=0, neginf=0, posinf=0)
		else:
			data_, freq_, mjds_, dm_ = pfd_data(file_)
			data_ = data_[:,0]
			
			d_mean_ = data_.mean(-1)
			unflag_freq = np.logical_and(d_mean_.std(0) > (np.median(d_mean_.std(0))*0.1) , d_mean_.std(0) < (np.median(d_mean_.std(0)) +15*robust_std(d_mean_.std(0)) ) )
			unflag_time = np.sum(data_, axis = (1,2)) != 0

		

	psrchive_method = False
	df = freq_[1] - freq_[0]
	delta_chan = 1
	
	
	
	if not(isinstance(n_chan, int)):
		n_chan = freq_.shape[0]#unflag_freq.sum()
	
	freq = freq_[unflag_freq]
	data_ = unflag_freq[None, :, None]*data_
	data_unflaged = data_[unflag_time][:,unflag_freq]

	if primary_comp:
		on_weight_arr_f, on_weight_arr_p, chunk_ind_arr = mask_with_prim_comp_flags_v6(data_, unflag_freq, thres, n_chan, prim_comp_ = True)
	else:
		on_weight_arr_, chunk_ind_arr = mask_with_prim_comp_flags_v6(data_, unflag_freq, thres, n_chan, prim_comp_ = False)
		on_weight_arr_f = on_weight_arr_p = on_weight_arr_
		

	time_s = (mjds_[-1] - mjds_[0] +  np.all(np.diff(mjds_,2) < abs(np.diff(mjds_)).min()*1e-3)*np.diff(mjds_)[0])*86400	
	
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
		unflagged_rms_freq.append(np.nanstd(profile_per_chan[~on_weight_arr_f[mask_ind]]))
		snr.append(unflagged_flux_freq[-1] * np.sqrt( on_weight_arr_p[mask_ind].sum())/unflagged_rms_freq[-1])
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
	snr3_bool = np.nan_to_num(snr, nan=0, posinf=0.0, neginf=0.0) > sigma
	#print(unflagged_flux_freq, unflagged_rms_freq)
	flux = (snr * noise)[snr3_bool * width_mask]
	noise_ = noise[snr3_bool * width_mask]
	error = np.sqrt(noise_**2 + (0.1 * flux)**2)
	log_freq, log_flux = np.log10(fcrunch10_freq[snr3_bool * width_mask]), np.log10(flux)
	error_y = error  # noise_
	
	
	if allow_fit:
		AICC = []
		param_arr = []
		param_err_arr = []
		try:
			for n_pow_law_ind in range(1,int(len(log_freq)/2)):
				#n_pow_law_ind = 1
				params_0, bounds_0 = initial_guess_n_bounds_func( n_pow_law_ind, log_freq, log_flux)
				#param_, _ = curve_fit(lambda nu_,*param_0: fit_func_st_line(nu_, n_pow_law_ind, *param_0), log_freq, log_flux, p0 = params_0, bounds = bounds_0, maxfev = 5000)
				param_, _ = curve_fit(lambda nu_,*param_0: fit_func_st_line(nu_, n_pow_law_ind, *param_0), log_freq, log_flux, p0 = params_0, sigma = noise_/flux , bounds = bounds_0, maxfev = 5000)
				#param_, _ = curve_fit(lambda nu_,*param_0: fit_func_st_line(nu_, n_pow_law_ind, *param_0), log_freq, log_flux, p0 = params_0, sigma = error_y/flux , bounds = bounds_0, maxfev = 5000)
				Chi2 = ( ((log_flux -  fit_func_st_line(log_freq, n_pow_law_ind, *param_))/(noise_/(np.log(10)*flux)))**2 ).sum()
				#Chi2 = ( ((10**log_flux -  10**fit_func_st_line(log_freq, n_pow_law_ind, *param_))/noise_)**2 ).sum()
				#print('number power law = ', n_pow_law_ind, ' ; chi2 = ',Chi2)
				aicc_ = Chi2 + 2*len(param_)*len(log_freq)/(len(log_freq) - len(param_) - 1)
				AICC.append(aicc_)
				param_arr.append(param_)
				param_err_arr.append(_)
				del params_0, bounds_0, Chi2, param_, _, aicc_
				n_pow_law = int(np.argmin(AICC) + 1)
				opt_param = param_arr[np.argmin(AICC)]
				opt_param_err = np.diag(param_err_arr[np.argmin(AICC)])
				
		except:
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
		subp4.errorbar(10**log_freq, flux, yerr = error_y, fmt = '.')
		if allow_fit:
			try:
				n_pow_law = int(np.argmin(AICC) + 1)
				opt_param = param_arr[np.argmin(AICC)]
				opt_param_err = np.diag(param_err_arr[np.argmin(AICC)])
				subp4.plot(10**np.linspace(log_freq[0], log_freq[-1], 1000), 10**fit_func_st_line(np.linspace(log_freq[0], log_freq[-1], 1000), n_pow_law, *opt_param) )
				#subp4.plot(10**log_freq, [10**fit_line(i, x[0], x[1]) for i in log_freq] )
				for i in range(n_pow_law): print("spectral index {0} +- {1} ".format(around(opt_param[n_pow_law - 1: -1][i],3), around(opt_param_err[n_pow_law - 1: -1][i],3)))
				subp4.set_title('Spectal Index:' + "{0} +- {1} ".format(np.around(opt_param[n_pow_law - 1: -1],3), np.around(opt_param_err[n_pow_law - 1: -1],3)))
				#subp4.set_title('Spectal Index:' + str(np.around(opt_param[n_pow_law - 1: -1],3)) )
			except:
				pass
		subp4.loglog()
		subp4.set_xticks(freq[::len(freq)//N_div + 1], freq.astype(int)[::len(freq)//N_div + 1],rotation=25)
		subp4.xaxis.set_minor_locator(ticker.AutoMinorLocator())
		subp4.tick_params(axis ='both', labelsize = label_size)
		####################################################################################
		#plt.
		
		subp5 = fig.add_subplot(2, 3, 6)
		subp5.plot(profile/profile.max())
		try:
			#subp5.plot(np.arange(data_unflaged.shape[-1])[on_weight_arr[0]], (profile/profile.max())[on_weight_arr[0]], 'o')
			subp5.plot(np.arange(data_.shape[-1])[on_weight_arr.sum(0).astype(bool)], (profile/profile.max())[on_weight_arr.sum(0).astype(bool)], 'o')
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
			plt.suptitle('Epoch : ' + str(epoch[2]) + '/' + str(epoch[1]) + '/' + str(epoch[0])  + '; Time : ' + str(round(time_s,2)) + r'$(s) ; n_{chan} : $' + str(n_chan) + '; sigma : ' + str(sigma) + '; thres : ' + str(thres)+'\n'+str(file_), fontsize = label_size)
		except:
			epoch = Time(mjds_[0], format='mjd').ymdhms
			plt.suptitle('Epoch : ' + str(epoch[2]) + '/' + str(epoch[1]) + '/' + str(epoch[0])  + '; Time : ' + str(round(time_s,2)) + r'$(s) ; n_{chan} : $' + str(n_chan) + '; sigma : ' + str(sigma) + '; thres : ' + str(thres), fontsize = label_size)
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
			return opt_param[n_pow_law - 1: -1], snr, flux, noise, error, np.rint(10**log_freq), on_off_mask, dm_
			
		except:
			return snr, flux, noise, error, np.rint(10**log_freq), on_off_mask, dm_
			
	else:
		return snr, flux, noise, error, np.rint(10**log_freq), on_off_mask, dm_

#####################################################################################################################


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
				param_, _ = curve_fit(lambda nu_,*param_0: fit_func_st_line(nu_, n_pow_law_ind, *param_0), log_freq, log_flux, p0 = params_0, sigma = noise_[~np.isnan(noise_)]/(np.log(10)*flux[~np.isnan(log_flux)]) , bounds = bounds_0, maxfev = 5000, nan_policy='omit')
				#print('TRIED THIS STEP : 2 ', param_)
				#param_, _ = curve_fit(lambda nu_,*param_0: fit_func_st_line(nu_, n_pow_law_ind, *param_0), log_freq, log_flux, p0 = params_0, sigma = error_y/(np.log(10)*flux) , bounds = bounds_0, maxfev = 5000, nan_policy='omit')
				Chi2 = ( ((log_flux[~np.isnan(log_flux)] -  fit_func_st_line(log_freq[~np.isnan(log_freq)], n_pow_law_ind, *param_))/(noise_[~np.isnan(noise_)]/(np.log(10)*flux[~np.isnan(log_flux)])))**2 ).sum()
				#print('TRIED THIS STEP : 3')
				#Chi2 = ( ((10**log_flux[~np.isnan(log_flux)] -  10**fit_func_st_line(log_freq[~np.isnan(log_freq)], n_pow_law_ind, *param_))/noise_[~np.isnan(noise_)])**2 ).sum()
				#print('number power law = ', n_pow_law_ind, ' ; chi2 = ',Chi2)
				aicc_ = Chi2 + 2*len(param_)*len(log_freq[~np.isnan(log_freq)])/(len(log_freq[~np.isnan(log_freq)]) - len(param_) - 1)
				AICC.append(aicc_)
				param_arr.append(param_)
				param_err_arr.append(_)
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
#####################################################################################################################


# This version is identical to ugmrt_in_band_flux_spectra_v2 except for cosmetic changes like tick adjustments etc.


def ugmrt_in_band_flux_spectra_v2_tick_adjustments(Tsky, n_chan = 10, sigma_ = 10, thres = 3, beam_ = 'PA', primary_comp = True, show_plots = True, allow_fit = False, save_plot = False, n_ant = 23, **file_data):
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
				param_, _ = curve_fit(lambda nu_,*param_0: fit_func_st_line(nu_, n_pow_law_ind, *param_0), log_freq, log_flux, p0 = params_0, sigma = noise_[~np.isnan(noise_)]/(np.log(10)*flux[~np.isnan(log_flux)]) , bounds = bounds_0, maxfev = 5000, nan_policy='omit')
				#print('TRIED THIS STEP : 2 ', param_)
				#param_, _ = curve_fit(lambda nu_,*param_0: fit_func_st_line(nu_, n_pow_law_ind, *param_0), log_freq, log_flux, p0 = params_0, sigma = error_y/(np.log(10)*flux) , bounds = bounds_0, maxfev = 5000, nan_policy='omit')
				Chi2 = ( ((log_flux[~np.isnan(log_flux)] -  fit_func_st_line(log_freq[~np.isnan(log_freq)], n_pow_law_ind, *param_))/(noise_[~np.isnan(noise_)]/(np.log(10)*flux[~np.isnan(log_flux)])))**2 ).sum()
				#print('TRIED THIS STEP : 3')
				#Chi2 = ( ((10**log_flux[~np.isnan(log_flux)] -  10**fit_func_st_line(log_freq[~np.isnan(log_freq)], n_pow_law_ind, *param_))/noise_[~np.isnan(noise_)])**2 ).sum()
				#print('number power law = ', n_pow_law_ind, ' ; chi2 = ',Chi2)
				aicc_ = Chi2 + 2*len(param_)*len(log_freq[~np.isnan(log_freq)])/(len(log_freq[~np.isnan(log_freq)]) - len(param_) - 1)
				AICC.append(aicc_)
				param_arr.append(param_)
				param_err_arr.append(_)
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
		#plt.colorbar(ax1, ax=subp1)
		subp1.set_yticks(on_ticks_ind - 0.5, freq_.astype(int)[on_ticks_ind])
		for i in on_ticks_ind: subp1.axhline(i - 0.5, color='white',alpha=alpha0,linestyle='--')
		subp1.set_ylabel('Frequency', fontsize = label_size)
		subp1.set_xlabel('Phase bins', fontsize = label_size)
		subp1.tick_params(axis ='both', labelsize = label_size)
		#subp1.annotate(xy=(0,100), xytext=(-10, 0),  text='a', fontsize= 2*label_size)
		#subp1.annotate(xy=(50,600), xycoords='figure points', text='a', fontsize= 2*label_size)
		subp1.annotate(xy=(50,650), xycoords='figure points', text='a', fontsize= 2*label_size)
		#subp1.annotate(xy=(50,600), xycoords='figure points', text='a', fontsize= 2*label_size)
		#subp1.annotate(xy=(40,600), xycoords='subfigure pixels',  text='a', fontsize= 2*label_size)
		############################################################################
		subp2 = fig.add_subplot(2, 3, 2)
		subp2.imshow(on_off_mask, aspect='auto', origin='lower')# , extent=[0,data_.shape[-1],freq_.min(),freq_.max()])
		subp2.set_yticks(on_ticks_ind[::2] - 0.5, freq_.astype(int)[on_ticks_ind[::2]])
		for i in on_ticks_ind: subp2.axhline(i - 0.5, color='white',alpha=alpha1,linestyle='--')
		
		subp2.tick_params(axis ='both', labelsize = label_size)
		subp2.annotate(xy=(410,650), xycoords='subfigure points', text='b', fontsize= 2*label_size)
		#subp2.annotate(xy=(410,600), xycoords='subfigure points', text='b', fontsize= 2*label_size)
		###############################################################################
		subp3 = fig.add_subplot(2, 3, 5)
		subp3.imshow(fchrunched_data.mean(axis=0), aspect='auto', origin='lower')# , extent=[0,data_.shape[-1],freq_.min(),freq_.max()])
		subp3.set_yticks(on_ticks_ind[::2] - 0.5, freq_.astype(int)[on_ticks_ind[::2]])
		for i in on_ticks_ind: subp3.axhline(i - 0.5, color='white',alpha=alpha1,linestyle='--')
		subp3.tick_params(axis ='both', labelsize = label_size)
		subp3.annotate(xy=(410,300), xycoords='figure points', text='c', fontsize= 2*label_size)
		#subp3.annotate(xy=(410,280), xycoords='figure points', text='c', fontsize= 2*label_size)
		################################################################
		subp4 = fig.add_subplot(2, 3, 3)
		subp4.errorbar(10**log_freq, 1e3*flux, yerr = 1e3*error_y, fmt = '.')
		if allow_fit:
			try:
				#print('NOW PLOTTING THE SPECTRA')
				n_pow_law = int(np.argmin(AICC) + 1)
				opt_param = param_arr[np.argmin(AICC)]
				opt_param_err = np.diag(param_err_arr[np.argmin(AICC)])
				break_arr, spectral_index_arr = opt_param[: n_pow_law -1], opt_param[n_pow_law - 1: -1]
				break_err_arr, spectral_index_err_arr = opt_param_err[: n_pow_law -1], opt_param_err[n_pow_law - 1: -1]
				subp4.plot(10**np.linspace(log_freq[np.nanargmin(log_freq)], log_freq[np.nanargmax(log_freq)], 1000), 1e3*10**fit_func_st_line(np.linspace(log_freq[np.nanargmin(log_freq)], log_freq[np.nanargmax(log_freq)], 1000), n_pow_law, *opt_param) )
				for i in range(n_pow_law): print("spectral index {0} +- {1} ".format(round(spectral_index_arr[i],3), round(spectral_index_err_arr[i],3)))
				#subp4.set_title('Spectal Index:' + "{0} +- {1} ".format(np.around(spectral_index_arr,3), np.around(spectral_index_err_arr,3)) + '\n' + 'Breaks at (MHz):' + "{0} +- {1}".format(np.around(10**break_arr,3), np.around(np.log(10)*break_err_arr*10**break_arr,3)))
			except:
				#print('$$$$$$$ ^^^^^^^ CAN\'T FIT THE SPECTRA')
				pass
		subp4.loglog()
		subp4.set_xlabel('Frequency (MHz)', fontsize = label_size)
		subp4.set_ylabel('Flux (mJy)', fontsize = label_size)
		subp4.set_xticks(freq[::len(freq)//N_div + 1], freq.astype(int)[::len(freq)//N_div + 1],rotation=25)
		subp4.yaxis.set_minor_formatter(ticker.ScalarFormatter())
		subp4.yaxis.set_major_formatter(ticker.ScalarFormatter())
		subp4.xaxis.set_minor_locator(ticker.AutoMinorLocator())
		subp4.tick_params(axis ='both', labelsize = label_size)
		subp4.annotate(xy=(750,650), xycoords='figure points', text='d', fontsize= 2*label_size)
		#subp4.annotate(xy=(750,600), xycoords='figure points', text='d', fontsize= 2*label_size)
		####################################################################################
		
		subp5 = fig.add_subplot(2, 3, 6)
		subp5.plot(profile/profile.max())
		try:
			subp5.plot(np.arange(data_.shape[-1])[on_weight_arr.sum(0).astype(bool)], (profile/profile.max())[on_weight_arr.sum(0).astype(bool)], 'o')
		except:
			pass
		subp5.set_ylabel('Normalized amplitude', fontsize = label_size )
		subp5.set_xlabel('Phase bins', fontsize = label_size )
		subp5.annotate(xy=(740,300), xycoords='figure points', text='e', fontsize= 2*label_size)
		#subp5.annotate(xy=(750,280), xycoords='figure points', text='e', fontsize= 2*label_size)
		'''
		try: 
			#subp5.set_title('Total flux : ' + str(round(Total_flux[0],3)) + '(mJy); snr : ' + str(round(SNR[0], 3)), fontsize = label_size )
		except:
			pass
		'''
		subp5.tick_params(axis ='both', labelsize = label_size)
		try:
			epoch = Time(mjds_[0], format='mjd').ymdhms
			#plt.suptitle('Epoch : ' + str(epoch[2]) + '/' + str(epoch[1]) + '/' + str(epoch[0])  + '; Time : ' + str(round(time_s,2)) + r'$(s) ; n_{chan} : $' + str(n_chan) + '; sigma : ' + str(sigma_) + '; thres : ' + str(thres)+'\n'+str(file_), fontsize = label_size)
		except:
			epoch = Time(mjds_[0], format='mjd').ymdhms
			#plt.suptitle('Epoch : ' + str(epoch[2]) + '/' + str(epoch[1]) + '/' + str(epoch[0])  + '; Time : ' + str(round(time_s,2)) + r'$(s) ; n_{chan} : $' + str(n_chan) + '; sigma : ' + str(sigma_) + '; thres : ' + str(thres), fontsize = label_size)
			pass
		plt.tight_layout()
		
		if save_plot:
			#file_name_ = str('./plot_files/') + file_[:-4] + str('.png')
			file_name_ = str(file_.split('/')[-1])[:-4] + str('.png')
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


#####################################################################################################################


# This version mainly does (data - data_mean[off pulse])/data_std(off_pulse)[:,:,None] -> create the on-off mask. if mask_with_prim_comp_flags_v7 is used then the definition of standard deviation for a chunk in frequecy changes (1/np.sqrt(sum_along_freq(1/variance_along_time)))



def ugmrt_in_band_flux_spectra_v2_fit_variant(Tsky, n_chan = 10, sigma_ = 10, thres = 3, beam_ = 'PA', primary_comp = True, show_plots = True, allow_fit = False, save_plot = False, n_ant = 23, **file_data):
	bandpass_correction = True
	robust_std = lambda x_ : 1.4826*np.median(abs(x_.flatten() - np.median(x_.flatten())))
	if len(file_data.keys()) >2:
		data_, freq_, mjds_ = file_data['DATA'], file_data['FREQ'], file_data['MJD']
		unflag_freq = np.sum(data_, axis = (0,2)) != 0		# for removing fully flagged channels from the data_
		unflag_freq_360_380 = np.logical_or(freq_ < 360, freq_ > 380)
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
			data_ = np.nan_to_num((data_ - np.nanmean(data_[:,:,global_off_bins_bool_arr], axis=-1,keepdims=True))/np.nanstd(data_[:,:,global_off_bins_bool_arr], axis=-1,keepdims=True), nan=0, neginf=0,posinf=0)
			d_mean_ = data_.mean(-1)
			unflag_freq_360_380 = np.logical_or(freq_ < 360, freq_ > 380)
			unflag_freq = np.logical_and(d_mean_.std(0) > (np.median(d_mean_.std(0))*0.1) , d_mean_.std(0) < (np.median(d_mean_.std(0)) +15*robust_std(d_mean_.std(0)) ) )
			unflag_freq = np.logical_and(unflag_freq, unflag_freq_360_380)
			unflag_time = np.sum(data_, axis = (1,2)) != 0
		else:
			data_, freq_, mjds_, dm_ = pfd_data(file_)
			data_ = data_[:,0]
			unflag_freq_360_380 = np.logical_or(freq_ < 360, freq_ > 380)
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
	noise = 1e3*np.array(noise) # Converted to mJy
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
	freq_arr_, flux_arr_ = np.where(snr3_bool * width_mask, fcrunch10_freq, np.nan), flux#fcrunch10_freq[snr3_bool * width_mask], flux
	error_y = error  # noise_
	#return flux, noise_, error, freq_arr_, flux_arr_
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
			for n_pow_law_ind in range(1,int(len(freq_arr_[~np.isnan(freq_arr_)])/2)):
				#n_pow_law_ind = 1
				params_0, bounds_0 = initial_guess_n_bounds_pow_law_func( n_pow_law_ind, freq_arr_[~np.isnan(freq_arr_)], flux_arr_[~np.isnan(flux_arr_)])
				print('TRIED THIS STEP : 1 ', params_0)
				#param_, _ = curve_fit(lambda nu_,*param_0: fit_func_st_line(nu_, n_pow_law_ind, *param_0), log_freq, log_flux, p0 = params_0, bounds = bounds_0, maxfev = 5000, nan_policy='omit')
				param_, _ = curve_fit(lambda nu_,*param_0: fit_func_power_law(nu_, n_pow_law_ind, *param_0), freq_arr_, flux_arr_, p0 = params_0, sigma = noise_[~np.isnan(noise_)] , bounds = bounds_0, maxfev = 5000, absolute_sigma=True, nan_policy='omit')
				print('TRIED THIS STEP : 2 ', param_)
				#param_, _ = curve_fit(lambda nu_,*param_0: fit_func_st_line(nu_, n_pow_law_ind, *param_0), log_freq, log_flux, p0 = params_0, sigma = error_y/(np.log(10)*flux) , bounds = bounds_0, maxfev = 5000, nan_policy='omit')
				Chi2 = ( ((flux_arr_[~np.isnan(flux_arr_)] -  fit_func_power_law(freq_arr_[~np.isnan(flux_arr_)], n_pow_law_ind, *param_))/noise_[~np.isnan(noise_)])**2 ).sum()
				print('TRIED THIS STEP : 3')
				#Chi2 = ( ((10**log_flux -  10**fit_func_st_line(log_freq, n_pow_law_ind, *param_))/noise_)**2 ).sum()
				#print('number power law = ', n_pow_law_ind, ' ; chi2 = ',Chi2)
				aicc_ = Chi2 + 2*len(param_)*len(freq_arr_[~np.isnan(freq_arr_)])/(len(freq_arr_[~np.isnan(freq_arr_)]) - len(param_) - 1)
				AICC.append(aicc_)
				param_arr.append(param_)
				param_err_arr.append(_)
				del params_0, bounds_0, Chi2, param_, _, aicc_
				n_pow_law = int(np.argmin(AICC) + 1)
				opt_param = param_arr[np.argmin(AICC)]
				opt_param_err = np.diag(param_err_arr[np.argmin(AICC)])
				
		except:
			print('THE FIT WAS BAD')
			pass
	
	####################################################################
	if bool(save_plot + show_plots) :
		fig = plt.figure(figsize=(15,10))
		label_size = 15
		N_div = 10
		N_tick = len(freq_arr_)//N_div
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
		subp4.errorbar(freq_arr_, flux_arr_, yerr = error_y, fmt = '.')
		if allow_fit:
			try:
				n_pow_law = int(np.argmin(AICC) + 1)
				opt_param = param_arr[np.argmin(AICC)]
				opt_param_err = np.diag(param_err_arr[np.argmin(AICC)])
				break_arr, spectral_index_arr = opt_param[: n_pow_law -1], opt_param[n_pow_law - 1: -1]
				break_err_arr, spectral_index_err_arr = opt_param_err[: n_pow_law -1], opt_param_err[n_pow_law - 1: -1]
				subp4.plot(np.linspace(freq_arr_[np.nanargmin(freq_arr_)], freq_arr_[np.nanargmax(freq_arr_)], 1000), fit_func_power_law(np.linspace(freq_arr_[np.nanargmin(freq_arr_)], freq_arr_[np.nanargmax(freq_arr_)], 1000), n_pow_law, *opt_param) )
				for i in range(n_pow_law): print("spectral index {0} +- {1} ".format(round(spectral_index_arr[i],3), round(spectral_index_err_arr[i],3)))
				subp4.set_title('Spectal Index:' + "{0} +- {1} ".format(np.around(spectral_index_arr,3), np.around(spectral_index_err_arr,3)) + '\n' + 'Breaks at (MHz):' + "{0} +- {1}".format(np.around(break_arr,3), np.around(break_err_arr,3)))
			except:
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
			return [opt_param, opt_param_err], snr, flux, noise_, error, np.rint(freq_arr_), on_off_mask, dm_
			
		except:
			return snr, flux, noise_, error, np.rint(freq_arr_), on_off_mask, dm_
			
	else:
		return snr, flux, noise_, error, np.rint(freq_arr_), on_off_mask, dm_
#####################################################################################################################
#####################################################################################################################


# This version mainly does (data - data_mean[off pulse])/data_std(off_pulse)[:,:,None] -> create the on-off mask. if mask_with_prim_comp_flags_v7 is used then the definition of standard deviation for a chunk in frequecy changes (1/np.sqrt(sum_along_freq(1/variance_along_time)))



def ugmrt_in_band_flux_spectra_v2_fit_variant_nan_show_only(Tsky, n_chan = 10, sigma_ = 10, thres = 3, beam_ = 'PA', primary_comp = True, show_plots = True, allow_fit = False, save_plot = False, n_ant = 23, **file_data):
	bandpass_correction = True
	robust_std = lambda x_ : 1.4826*np.median(abs(x_.flatten() - np.median(x_.flatten())))
	if len(file_data.keys()) >2:
		data_, freq_, mjds_ = file_data['DATA'], file_data['FREQ'], file_data['MJD']
		unflag_freq = np.sum(data_, axis = (0,2)) != 0		# for removing fully flagged channels from the data_
		unflag_freq_360_380 = np.logical_or(freq_ < 360, freq_ > 380)
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
			data_ = np.nan_to_num((data_ - np.nanmean(data_[:,:,global_off_bins_bool_arr], axis=-1,keepdims=True))/np.nanstd(data_[:,:,global_off_bins_bool_arr], axis=-1,keepdims=True), nan=0, neginf=0,posinf=0)
			d_mean_ = data_.mean(-1)
			unflag_freq_360_380 = np.logical_or(freq_ < 360, freq_ > 380)
			unflag_freq = np.logical_and(d_mean_.std(0) > (np.median(d_mean_.std(0))*0.1) , d_mean_.std(0) < (np.median(d_mean_.std(0)) +15*robust_std(d_mean_.std(0)) ) )
			unflag_freq = np.logical_and(unflag_freq, unflag_freq_360_380)
			unflag_time = np.sum(data_, axis = (1,2)) != 0
		else:
			data_, freq_, mjds_, dm_ = pfd_data(file_)
			data_ = data_[:,0]
			unflag_freq_360_380 = np.logical_or(freq_ < 360, freq_ > 380)
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
	flux_return_nan = np.where(snr3_bool * width_mask, snr * noise, np.nan) #(snr * noise)[snr3_bool * width_mask]
	noise_return_nan = np.where(snr3_bool * width_mask, noise, np.nan) #noise[snr3_bool * width_mask]
	error_return_nan = np.sqrt(noise_return_nan**2 + (0.1 * flux_return_nan)**2)
	freq_arr_return_nan, flux_arr_return_nan = fcrunch10_freq, flux_return_nan
	error_y_return_nan = error_return_nan  # noise_return_nan 
	flux = flux_arr_return_nan[~np.isnan(flux_return_nan)]
	noise_ = noise_return_nan[~np.isnan(noise_return_nan)]
	error = np.sqrt(noise_**2 + (0.1 * flux)**2)
	freq_arr_, flux_arr_ = fcrunch10_freq[snr3_bool * width_mask], flux
	error_y = error  # noise_
	#return flux, noise_, error, freq_arr_, flux_arr_
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
			for n_pow_law_ind in range(1,int(len(freq_arr_)/2)):
				#n_pow_law_ind = 1
				params_0, bounds_0 = initial_guess_n_bounds_pow_law_func( n_pow_law_ind, freq_arr_, flux_arr_)
				#param_, _ = curve_fit(lambda nu_,*param_0: fit_func_st_line(nu_, n_pow_law_ind, *param_0), log_freq, log_flux, p0 = params_0, bounds = bounds_0, maxfev = 5000, nan_policy='omit')
				param_, _ = curve_fit(lambda nu_,*param_0: fit_func_power_law(nu_, n_pow_law_ind, *param_0), freq_arr_, flux_arr_, p0 = params_0, sigma = noise_ , bounds = bounds_0, maxfev = 5000)
				#param_, _ = curve_fit(lambda nu_,*param_0: fit_func_st_line(nu_, n_pow_law_ind, *param_0), log_freq, log_flux, p0 = params_0, sigma = error_y/(np.log(10)*flux) , bounds = bounds_0, maxfev = 5000, nan_policy='omit')
				Chi2 = ( ((flux_arr_ -  fit_func_power_law(freq_arr_, n_pow_law_ind, *param_))/noise_)**2 ).sum()
				#Chi2 = ( ((10**log_flux -  10**fit_func_st_line(log_freq, n_pow_law_ind, *param_))/noise_)**2 ).sum()
				#print('number power law = ', n_pow_law_ind, ' ; chi2 = ',Chi2)
				aicc_ = Chi2 + 2*len(param_)*len(freq_arr_)/(len(freq_arr_) - len(param_) - 1)
				AICC.append(aicc_)
				param_arr.append(param_)
				param_err_arr.append(_)
				del params_0, bounds_0, Chi2, param_, _, aicc_
				n_pow_law = int(np.argmin(AICC) + 1)
				opt_param = param_arr[np.argmin(AICC)]
				opt_param_err = np.diag(param_err_arr[np.argmin(AICC)])
				
		except:
			print('THE FIT WAS BAD')
			pass
	
	####################################################################
	if bool(save_plot + show_plots) :
		fig = plt.figure(figsize=(15,10))
		label_size = 15
		N_div = 10
		N_tick = len(freq_arr_)//N_div
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
		subp4.errorbar(freq_arr_, flux_arr_, yerr = error_y, fmt = '.')
		if allow_fit:
			try:
				n_pow_law = int(np.argmin(AICC) + 1)
				opt_param = param_arr[np.argmin(AICC)]
				opt_param_err = np.diag(param_err_arr[np.argmin(AICC)])
				break_arr, spectral_index_arr = opt_param[: n_pow_law -1], opt_param[n_pow_law - 1: -1]
				break_err_arr, spectral_index_err_arr = opt_param_err[: n_pow_law -1], opt_param_err[n_pow_law - 1: -1]
				subp4.plot(np.linspace(freq_arr_[np.nanargmin(freq_arr_)], freq_arr_[np.nanargmax(freq_arr_)], 1000), fit_func_power_law(np.linspace(freq_arr_[np.nanargmin(freq_arr_)], freq_arr_[np.nanargmax(freq_arr_)], 1000), n_pow_law, *opt_param) )
				for i in range(n_pow_law): print("spectral index {0} +- {1} ".format(round(spectral_index_arr[i],3), round(spectral_index_err_arr[i],3)))
				subp4.set_title('Spectal Index:' + "{0} +- {1} ".format(np.around(spectral_index_arr,3), np.around(spectral_index_err_arr,3)) + '\n' + 'Breaks at (MHz):' + "{0} +- {1}".format(np.around(break_arr,3), np.around(break_err_arr,3)))
			except:
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
			return [opt_param, opt_param_err], snr, flux_return_nan, noise_return_nan , error_return_nan, np.rint(freq_arr_return_nan), on_off_mask, dm_
			
		except:
			return snr, flux_return_nan, noise_return_nan , error_return_nan, np.rint(freq_arr_return_nan), on_off_mask, dm_
			
	else:
		return snr, flux_return_nan, noise_return_nan , error_return_nan, np.rint(freq_arr_return_nan), on_off_mask, dm_
#####################################################################################################################


# This version mainly does (data - data_mean[off pulse]) -> create the on-off mask -> data/data_std(off_pulse)[:,:,None]. if mask_with_prim_comp_flags_v7 is used then the definition of standard deviation for a chunk in frequecy changes (1/np.sqrt(sum_along_freq(1/variance_along_time)))



def ugmrt_in_band_flux_spectra_v3(Tsky, n_chan = 10, sigma_ = 10, thres = 3, beam_ = 'PA', primary_comp = True, show_plots = True, allow_fit = False, save_plot = False, n_ant = 23, **file_data):
	bandpass_correction = True
	robust_std = lambda x_ : 1.4826*np.median(abs(x_.flatten() - np.median(x_.flatten())))
	if len(file_data.keys()) >2:
		data_, freq_, mjds_, dm_, file_ = file_data['DATA'], file_data['FREQ'], file_data['MJD'], file_data['DM'], file_data['FILE']
		unflag_freq = np.sum(data_, axis = (0,2)) != 0		# for removing fully flagged channels from the data_
		unflag_freq_360_380 = np.logical_or(freq_ < 360, freq_ > 380)
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
			data_ = np.nan_to_num(data_ - np.nanmean(data_[:,:,global_off_bins_bool_arr], axis=-1,keepdims=True))
			d_mean_ = np.nan_to_num(data_/np.nanstd(data_[:,:,global_off_bins_bool_arr], axis=-1,keepdims=True), nan=0, neginf=0,posinf=0).mean(-1)
			unflag_freq = np.logical_and(d_mean_.std(0) > (np.median(d_mean_.std(0))*0.1) , d_mean_.std(0) < (np.median(d_mean_.std(0)) +15*robust_std(d_mean_.std(0)) ) )
			unflag_time = np.sum(data_, axis = (1,2)) != 0
			unflag_freq_360_380 = np.logical_or(freq_ < 360, freq_ > 380)
			unflag_freq = np.logical_and(unflag_freq, unflag_freq_360_380)
		else:
			data_, freq_, mjds_, dm_ = pfd_data(file_)
			data_ = data_[:,0]
			
			d_mean_ = data_.mean(-1)
			unflag_freq = np.logical_and(d_mean_.std(0) > (np.median(d_mean_.std(0))*0.1) , d_mean_.std(0) < (np.median(d_mean_.std(0)) +15*robust_std(d_mean_.std(0)) ) )
			unflag_freq_360_380 = np.logical_or(freq_ < 360, freq_ > 380)
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
		on_weight_arr_f, on_weight_arr_p, chunk_ind_arr = mask_with_prim_comp_flags_v7(data_, unflag_freq, thres, n_chan, prim_comp_ = True)
	else:
		on_weight_arr_, chunk_ind_arr = mask_with_prim_comp_flags_v7(data_, unflag_freq, thres, n_chan, prim_comp_ = False)
		on_weight_arr_f = on_weight_arr_p = on_weight_arr_
	
	data_ = np.nan_to_num(data_/np.nanstd(data_[:,:,global_off_bins_bool_arr], axis=-1,keepdims=True), nan=0, neginf=0,posinf=0)
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
	flux = (snr * noise)[snr3_bool * width_mask]
	noise_ = noise[snr3_bool * width_mask]
	error = np.sqrt(noise_**2 + (0.1 * flux)**2)
	log_freq, log_flux = np.log10(fcrunch10_freq[snr3_bool * width_mask]), np.log10(flux)
	error_y = error  # noise_
	
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
			for n_pow_law_ind in range(1,int(len(log_freq)/2)):
				#n_pow_law_ind = 1
				params_0, bounds_0 = initial_guess_n_bounds_func( n_pow_law_ind, log_freq, log_flux)
				#param_, _ = curve_fit(lambda nu_,*param_0: fit_func_st_line(nu_, n_pow_law_ind, *param_0), log_freq, log_flux, p0 = params_0, bounds = bounds_0, maxfev = 5000)
				param_, _ = curve_fit(lambda nu_,*param_0: fit_func_st_line(nu_, n_pow_law_ind, *param_0), log_freq, log_flux, p0 = params_0, sigma = noise_/(np.log(10)*flux) , bounds = bounds_0, maxfev = 5000)
				#param_, _ = curve_fit(lambda nu_,*param_0: fit_func_st_line(nu_, n_pow_law_ind, *param_0), log_freq, log_flux, p0 = params_0, sigma = error_y/(np.log(10)*flux) , bounds = bounds_0, maxfev = 5000)
				Chi2 = ( ((log_flux -  fit_func_st_line(log_freq, n_pow_law_ind, *param_))/(noise_/(np.log(10)*flux)))**2 ).sum()
				#Chi2 = ( ((10**log_flux -  10**fit_func_st_line(log_freq, n_pow_law_ind, *param_))/noise_)**2 ).sum()
				#print('number power law = ', n_pow_law_ind, ' ; chi2 = ',Chi2)
				aicc_ = Chi2 + 2*len(param_)*len(log_freq)/(len(log_freq) - len(param_) - 1)
				AICC.append(aicc_)
				param_arr.append(param_)
				param_err_arr.append(_)
				del params_0, bounds_0, Chi2, param_, _, aicc_
				n_pow_law = int(np.argmin(AICC) + 1)
				opt_param = param_arr[np.argmin(AICC)]
				opt_param_err = np.diag(param_err_arr[np.argmin(AICC)])
		except:
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
				n_pow_law = int(np.argmin(AICC) + 1)
				opt_param = param_arr[np.argmin(AICC)]
				opt_param_err = np.diag(param_err_arr[np.argmin(AICC)])
				break_arr, spectral_index_arr = opt_param[: n_pow_law -1], opt_param[n_pow_law - 1: -1]
				break_err_arr, spectral_index_err_arr = opt_param_err[: n_pow_law -1], opt_param_err[n_pow_law - 1: -1]
				subp4.plot(10**np.linspace(log_freq[0], log_freq[-1], 1000), 10**fit_func_st_line(np.linspace(log_freq[0], log_freq[-1], 1000), n_pow_law, *opt_param) )
				for i in range(n_pow_law): print("spectral index {0} +- {1} ".format(round(spectral_index_arr[i],3), round(spectral_index_err_arr[i],3)))
				subp4.set_title('Spectal Index:' + "{0} +- {1} ".format(np.around(spectral_index_arr,3), np.around(spectral_index_err_arr,3)) + '\n' + 'Breaks at (MHz):' + "{0} +- {1}".format(np.around(10**break_arr,3), np.around(np.log(10)*break_err_arr*10**break_arr,3)))
			except:
				pass
		subp4.loglog()
		subp4.set_xticks(freq[::len(freq)//N_div + 1], freq.astype(int)[::len(freq)//N_div + 1],rotation=25)
		subp4.xaxis.set_minor_locator(ticker.AutoMinorLocator())
		subp4.tick_params(axis ='both', labelsize = label_size)
		####################################################################################
		#plt.
		
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
######################################################################################################################################################################################################

#	Note : Before running make sure the search_file are present in respective search_band for each of the Pulsar (for example: b3_files_mjd_freq_file.npy should be there in .../J1646-2142/good_pfd_b3)

start_loc = '/data/rahul/psr_file/ALL_MSP_PFD_fs2/ALL_MSP_PFD'#'/mnt/f/Studies/NCRA/PhD\ Project/Back_up/data/MSP_spectra'

#MSP_list = ['J1646-2142', 'J2144-5237', 'J1828+0625', 'J1536-4948', 'J1207-5050', 'J1120-3618', 'J0248+4230', 'J1242-4712', 'J2101-4802']
MSP_list = ['J1828+0625']
search_band = ['good_pfd_b3']#, 'good_pfd_b4', 'good_pfd_b5']
search_file = ['b3_files_mjd_freq_file.npy', 'b4_files_mjd_freq_file.npy', 'b5_files_mjd_freq_file.npy']

Tsky_info_dict = {'J1646-2142' : [355, 60, 17, 3], 'J2144-5237' : [145, 24, 7, 1], 'J1828+0625' : [503, 86, 24, 4], 'J1536-4948' : [742, 126,36, 6], 'J1207-5050' : [207, 35, 10, 1], 'J1120-3618' : [133, 22, 6, 1], 'J0248+4230' : [210, 35, 10, 1], 'J1242-4712' : [188, 32, 9, 1], 'J2101-4802' : [181, 31, 9, 1]}

band_func = lambda nu : 5 if 1000<=nu<=1460 else (4 if 550<=nu<=1000 else (3 if 250<=nu<=500 else (2 if 100<=nu<=250 else 'No Band info')))


#				Applying the method to various pulsars stored in a given file format : 'start_loc/MSP_name/good_pfd_b*' ; png will be stored in : 'start_loc/MSP_name/good_pfd_b*/plot_files_*'
for prime_comp in [False]:#, True]:
	if prime_comp:
		comp_tag = 'prim'
	else:
		comp_tag = 'all'
	
	for l in MSP_list:
		print('!!!!!!!!!!!!!!!!!!@@@@@@@@@@@@@@@@@@@@@@@##########################$$$$$$$$$$$$$$$$$$$$$$$$%%%%%%%%%%%%%%%%%%%%%%%%%^^^^^^^^^^^^^^^^^^^^^^&&&&&&&&&&&&&&&&&&&&&&')
		os.chdir(start_loc)
		tsky_b2, tsky_b3, tsky_b4, tsky_b5 = Tsky_info_dict[l]
		print('WORKING ON MSP : ', l)
		for band_i in range(len(search_band)):
			Table_info = []
			os.chdir(start_loc+'/'+l)
			if os.path.exists(search_band[band_i]+'/'+search_file[band_i]):
				print('Entered file :', search_band[band_i]+'/'+search_file[band_i])
				os.chdir(search_band[band_i])
				if not os.path.exists('plot_files'):
					os.mkdir('plot_files')
				files_mjd_freq_file = np.load(search_file[band_i], allow_pickle=True)
				group_mjd_full = group(files_mjd_freq_file)

				count = count_G = 0
				for G in group_mjd_full:
					print(count_G)
					mode_arr = list(np.unique(np.array(G)[:, 3]))
					try:
						mode_arr.remove('unknown')
					except:
						pass
					switch = True
					for g in G:
						
						mjd, F, f, pa_mode, n_ant  = g
						if band_func(F) == 2:
							#continue
							tsky = tsky_b2
						elif band_func(F) == 3:
							#continue
							tsky = tsky_b3
						elif band_func(F) == 4:
							#continue
							tsky = tsky_b4
						elif band_func(F) == 5:
							#continue
							tsky = tsky_b5
						
						if n_ant == 'TBD':
							if np.array(G).shape[0] == 1:
								n_ant = 24
							else:
								if sum(abs(np.diff(np.array(G)[:,1].astype(float))) >100).astype(bool):
									n_ant = 12
								else:
									n_ant = 24
						if not mode_arr:
							pa_mode = 'PA'
						
							
						print('file : ', f)
						print('n_ant : ', n_ant)
						print('Sky temp : ',tsky)
						print('Beam mode : ', pa_mode)
						n_chan = 15
						sigma = 5
						thres = 3
						try:
							print(count)
							SNR, Total_flux, noise, error, freq, on_weight_arr, dm = ugmrt_in_band_flux_spectra_v2(tsky, 'full', float(sigma), float(thres), pa_mode, prime_comp, False, False, False, n_ant, FILE = str(f))
							Total_flux = Total_flux *1e3
							error *=1e3
							spec_par, snr_nu, flux_nu, noise_nu, error_nu, nu, weight, dm = ugmrt_in_band_flux_spectra_v2(tsky, int(n_chan), float(sigma), float(thres), pa_mode, prime_comp, False, True, True, n_ant, FILE = str(f))
							print('step 1')
							
							breaks_point, breaks_point_error = spec_par[0][: int( (len(spec_par[0])/2) - 1)], spec_par[1][: int( (len(spec_par[1])/2) - 1)]
							break_frequencies, break_frequencies_error = np.around(10**breaks_point,3), np.around(np.log(10)*breaks_point_error*10**breaks_point,3)
							spectral_index, spectral_index_error = spec_par[0][int( (len(spec_par[0])/2) - 1) : -1], spec_par[1][int( (len(spec_par[1])/2) - 1) : -1]
							'''
							break_frequencies, break_frequencies_error = spec_par[0][: int( (len(spec_par[0])/2) - 1)], spec_par[1][: int( (len(spec_par[1])/2) - 1)]
							spectral_index, spectral_index_error = spec_par[0][int( (len(spec_par[0])/2) - 1) : -1], spec_par[1][int( (len(spec_par[1])/2) - 1) : -1]
							'''
							Table_info.append([f, round(SNR[0],3), round(Total_flux[0],3), round(error[0],3), list(break_frequencies), list(break_frequencies_error) ,list(spectral_index), list(spectral_index_error), dm, [flux_nu,error_nu, snr_nu, nu], mjd])
							print('step 2')
							del F, mjd, SNR, Total_flux, noise, error, noise_nu, error_nu, freq, on_weight_arr, dm, spec_par, snr_nu, flux_nu, nu, weight,breaks_point, breaks_point_error, break_frequencies, break_frequencies_error, spectral_index, spectral_index_error
							print('=============================================================')
							print('Completed')
							print('=============================================================')
							if switch:
								switch = False
						except:
							print(count + 10000)
							print('problem file : ' )
							print(f)
							pass
						print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
						count += 1
					print('#####################################################')	
					count_G += 1
				os.system('convert plot_files/*png ' + 'plot_files/b'+str(int(band_i + 3))+ '_v2_'+ str(comp_tag) +'_'+str(n_chan)+'_'+str(sigma)+'_'+str(thres)+'.pdf')
				df = {'File' : np.array(Table_info,object)[:,0], 'SNR' : np.array(Table_info,object)[:,1], 'Total_Flux' : np.array(Table_info,object)[:,2], 'Flux_error' : np.array(Table_info,object)[:,3], 'Break_Frequencies' : np.array(Table_info,object)[:,4], 'Break_Frequencies_error' : np.array(Table_info,object)[:,5], 'Spectral_indices' : np.array(Table_info,object)[:,6], 'Spectral_indices_error' : np.array(Table_info,object)[:,7], 'dm' : np.array(Table_info,object)[:,8], 'Flux_snr_subband_freq' : np.array(Table_info,object)[:,9], 'mjd' : np.array(Table_info,object)[:,10]}
				df = pandas.DataFrame(df)
				df.to_csv('plot_files/File_b'+str(int(band_i + 3))+'_'+ str(comp_tag) +'_SNR_Flux_error_Break_frequency_error_spectral_index_error.csv')
				os.system('rm plot_files/*png')
#####################################################################################################################

##################                  Creating a data-structure for all pulsars and band info
start_loc = '/data/rahul/psr_file/ALL_MSP_PFD_fs2/ALL_MSP_PFD'#'/mnt/f/Studies/NCRA/PhD\ Project/Back_up/data/MSP_spectra'

MSP_list = ['J1646-2142', 'J2144-5237', 'J1828+0625', 'J1536-4948', 'J1207-5050', 'J1120-3618', 'J0248+4230', 'J1242-4712', 'J2101-4802']
#MSP_list = ['J2101-4802']
search_band = ['good_pfd_b3', 'good_pfd_b4', 'good_pfd_b5']
search_file = ['b3_files_mjd_freq_file.npy', 'b4_files_mjd_freq_file.npy', 'b5_files_mjd_freq_file.npy']

Tsky_info_dict = {'J1646-2142' : [355, 60, 17, 3], 'J2144-5237' : [145, 24, 7, 1], 'J1828+0625' : [503, 86, 24, 4], 'J1536-4948' : [742, 126,36, 6], 'J1207-5050' : [207, 35, 10, 1], 'J1120-3618' : [133, 22, 6, 1], 'J0248+4230' : [210, 35, 10, 1], 'J1242-4712' : [188, 32, 9, 1], 'J2101-4802' : [181, 31, 9, 1]}

band_func = lambda nu : 5 if 1000<=nu<=1460 else (4 if 550<=nu<=1000 else (3 if 250<=nu<=500 else (2 if 100<=nu<=250 else 'No Band info')))

all_info_df = {}
for prime_comp in [False]:#, True]:
	if prime_comp:
		comp_tag = 'prim'
	else:
		comp_tag = 'all'
	
	for l in MSP_list:
		df_l = {}
		print('!!!!!!!!!!!!!!!!!!@@@@@@@@@@@@@@@@@@@@@@@##########################$$$$$$$$$$$$$$$$$$$$$$$$%%%%%%%%%%%%%%%%%%%%%%%%%^^^^^^^^^^^^^^^^^^^^^^&&&&&&&&&&&&&&&&&&&&&&')
		os.chdir(start_loc)
		tsky_b2, tsky_b3, tsky_b4, tsky_b5 = Tsky_info_dict[l]
		print('WORKING ON MSP : ', l)
		for band_i in range(len(search_band)):
			Table_info = []
			os.chdir(start_loc+'/'+l)
			if os.path.exists(search_band[band_i]+'/'+search_file[band_i]):
				print('Entered file :', search_band[band_i]+'/'+search_file[band_i])
				os.chdir(search_band[band_i])
				if not os.path.exists('plot_files'):
					os.mkdir('plot_files')
				files_mjd_freq_file = np.load(search_file[band_i], allow_pickle=True)
				group_mjd_full = group(files_mjd_freq_file)

				count = count_G = 0
				for G in group_mjd_full:
					print(count_G)
					mode_arr = list(np.unique(np.array(G)[:, 3]))
					try:
						mode_arr.remove('unknown')
					except:
						pass
					switch = True
					for g in G:
						
						mjd, F, f, pa_mode, n_ant  = g
						if band_func(F) == 2:
							#continue
							tsky = tsky_b2
						elif band_func(F) == 3:
							#continue
							tsky = tsky_b3
						elif band_func(F) == 4:
							#continue
							tsky = tsky_b4
						elif band_func(F) == 5:
							#continue
							tsky = tsky_b5
						
						if n_ant == 'TBD':
							if np.array(G).shape[0] == 1:
								n_ant = 24
							else:
								if sum(abs(np.diff(np.array(G)[:,1].astype(float))) >100).astype(bool):
									n_ant = 12
								else:
									n_ant = 24
						if not mode_arr:
							pa_mode = 'PA'
						
							
						print('file : ', f)
						print('n_ant : ', n_ant)
						print('Sky temp : ',tsky)
						print('Beam mode : ', pa_mode)
						n_chan = 15
						sigma = 5
						thres = 3
						try:
							print(count)
							SNR, Total_flux, noise, error, freq, on_weight_arr, dm = ugmrt_in_band_flux_spectra_v2(tsky, 'full', float(sigma), float(thres), pa_mode, prime_comp, False, False, False, n_ant, FILE = str(f))
							Total_flux = Total_flux *1e3
							error *=1e3
							spec_par, snr_nu, flux_nu, noise_nu, error_nu, nu, weight, dm = ugmrt_in_band_flux_spectra_v2(tsky, int(n_chan), float(sigma), float(thres), pa_mode, prime_comp, False, True, True, n_ant, FILE = str(f))
							print('step 1')
							
							breaks_point, breaks_point_error = spec_par[0][: int( (len(spec_par[0])/2) - 1)], spec_par[1][: int( (len(spec_par[1])/2) - 1)]
							break_frequencies, break_frequencies_error = np.around(10**breaks_point,3), np.around(np.log(10)*breaks_point_error*10**breaks_point,3)
							spectral_index, spectral_index_error = spec_par[0][int( (len(spec_par[0])/2) - 1) : -1], spec_par[1][int( (len(spec_par[1])/2) - 1) : -1]
							'''
							break_frequencies, break_frequencies_error = spec_par[0][: int( (len(spec_par[0])/2) - 1)], spec_par[1][: int( (len(spec_par[1])/2) - 1)]
							spectral_index, spectral_index_error = spec_par[0][int( (len(spec_par[0])/2) - 1) : -1], spec_par[1][int( (len(spec_par[1])/2) - 1) : -1]
							'''
							Table_info.append([f, round(SNR[0],3), Total_flux[0], error[0], list(break_frequencies), list(break_frequencies_error) ,list(spectral_index), list(spectral_index_error), dm, [flux_nu,error_nu, snr_nu, nu], mjd])
							print('step 2')
							del F, mjd, SNR, Total_flux, noise, error, noise_nu, error_nu, freq, on_weight_arr, dm, spec_par, snr_nu, flux_nu, nu, weight,breaks_point, breaks_point_error, break_frequencies, break_frequencies_error, spectral_index, spectral_index_error
							print('=============================================================')
							print('Completed')
							print('=============================================================')
							if switch:
								switch = False
						except:
							print(count + 10000)
							print('problem file : ' )
							print(f)
							pass
						print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
						count += 1
					print('#####################################################')	
					count_G += 1
				os.system('convert plot_files/*png ' + 'plot_files/b'+str(int(band_i + 3))+ '_v2_'+ str(comp_tag) +'_'+str(n_chan)+'_'+str(sigma)+'_'+str(thres)+'.pdf')
				df = {'File' : np.array(Table_info,object)[:,0], 'SNR' : np.array(Table_info,object)[:,1], 'Total_Flux' : np.array(Table_info,object)[:,2], 'Flux_error' : np.array(Table_info,object)[:,3], 'Break_Frequencies' : np.array(Table_info,object)[:,4], 'Break_Frequencies_error' : np.array(Table_info,object)[:,5], 'Spectral_indices' : np.array(Table_info,object)[:,6], 'Spectral_indices_error' : np.array(Table_info,object)[:,7], 'dm' : np.array(Table_info,object)[:,8], 'Flux_error_snr_subband_freq' : np.array(Table_info,object)[:,9], 'mjd' : np.array(Table_info,object)[:,10]}
				df = pandas.DataFrame(df)
				#df.to_csv('plot_files/File_b'+str(int(band_i + 3))+'_'+ str(comp_tag) +'_SNR_Flux_error_Break_frequency_error_spectral_index_error.csv')
				#os.system('rm plot_files/*png')
				df_l['band'+str(search_band[band_i][-1])] = df
		all_info_df[l] = df_l

# Save in binary format using pickle
import pickle
'''
filehandler_i = open(start_loc+'/all_info_df.bin', 'wb')
pickle.dump(all_info_df, filehandler_i)
filehandler_i.close()
'''
# Read from binary format using pickle

filehandler = open(start_loc+'/all_info_df.bin', 'rb')
all_info_df = pickle.load(filehandler)
filehandler.close()

#############################			Copying all the pdf files (start_loc/*/good_pfd_b*/plot_files/) to results directory (start_loc/all_plots_summary_plot)
'''
############					Method 1 (*incomplete)


for l in MSP_list:
	print('!!!!!!!!!!!!!!!!!!@@@@@@@@@@@@@@@@@@@@@@@##########################$$$$$$$$$$$$$$$$$$$$$$$$%%%%%%%%%%%%%%%%%%%%%%%%%^^^^^^^^^^^^^^^^^^^^^^&&&&&&&&&&&&&&&&&&&&&&')
	os.chdir(start_loc)
	print('WORKING ON MSP : ', l)
	for band_i in range(3):
		os.chdir(start_loc+'/'+l)
		if not os.path.exists(start_loc+'/all_plots_summary_plot'+'/'+l):
			os.mkdir(start_loc+'/all_plots_summary_plot'+'/'+l)
		if os.path.exists(search_band[band_i]+'/plot_files_version_1'):
			print('Entered file :', search_band[band_i]+'/plot_files_version_1')
			os.chdir(search_band[band_i]+'/plot_files_version_1')
			if os.path.exists('b'+str(int(band_i + 3))+ '_v2_prim_'+str(n_chan)+'_'+str(sigma)+'_'+str(thres)+'.pdf'):
				os.system('cp b'+str(int(band_i + 3))+ '_v2_prim_'+str(n_chan)+'_'+str(sigma)+'_'+str(thres)+'.pdf ' + start_loc+'/all_plots_summary_plot')
			
'''
############					Method 2

os.chdir(start_loc)
plot_collection = start_loc + '/all_plots_summary_plot'
all_pdfs = glob.glob('*/good_pfd*/plot_files/*pdf')
pdf_csv = glob.glob('*/good_pfd*/plot_files/*csv') + glob.glob('*/good_pfd*/plot_files/*pdf')
try:
	os.system('rm -r ' + plot_collection + '/*')
except:
	pass
os.chdir(start_loc)
for files in pdf_csv:
	a_ = files.split('/')
	a_.pop(2)
	MSP_name, band_info, pdf_exist = a_
	print('Working on : ', MSP_name,  band_info, pdf_exist)
	if not os.path.exists(plot_collection + '/' + MSP_name):
		os.mkdir(plot_collection + '/' + MSP_name)
	os.system('cp ' + files + ' ' + plot_collection + '/' + MSP_name)
	print('##############################################################')
	
###################################		Renaming the plots_files on corresponding file structure


for l in MSP_list:
	print('!!!!!!!!!!!!!!!!!!@@@@@@@@@@@@@@@@@@@@@@@##########################$$$$$$$$$$$$$$$$$$$$$$$$%%%%%%%%%%%%%%%%%%%%%%%%%^^^^^^^^^^^^^^^^^^^^^^&&&&&&&&&&&&&&&&&&&&&&')
	os.chdir(start_loc)
	tsky_b2, tsky_b3, tsky_b4, tsky_b5 = Tsky_info_dict[l]
	print('WORKING ON MSP : ', l)
	for band_i in range(3):
		os.chdir(start_loc+'/'+l)
		if os.path.exists(search_band[band_i]+'/plot_files'):
			os.system('mv ' + search_band[band_i]+'/plot_files ' + search_band[band_i]+'/plot_files_version_9')
			

all_psr_names = list(all_info_df.keys())

def dashboard_0(psr_name,band_no,all_df_file = all_info_df):
    df_ = all_df_file[str(psr_name)]['band'+str(band_no)]
    print('PSR : ', psr_name, ' ,Band : ', str(band_no))
    a_ = df_.Flux_error_snr_subband_freq
    freq_chunk_arr = np.array([i[-1].tolist() for i in a_])
    flux_chunk_arr = np.array([i[0].tolist() for i in a_])
    flux_error_chunk_arr = np.array([i[1].tolist() for i in a_])
    mjd_arr = df_.mjd
    Flux_arr = df_.Total_Flux
    Flux_error_arr = df_.Flux_error
    fig,ax = plt.subplots(2,2,figsize=(14,12))
    ax[0,0].errorbar(mjd_arr, Flux_arr, yerr=Flux_error_arr, fmt='.')
    ax[0,0].set_title('Total Flux (mJy) vs mjd')
    ax[0,1].errorbar(np.nanmean(freq_chunk_arr,axis=0),1e3*np.nanmean(flux_chunk_arr,axis=0),yerr=1e3*np.sqrt(np.nansum(flux_error_chunk_arr**2,axis=0))/(~np.isnan(flux_error_chunk_arr)).sum(0))
    ax[0,1].loglog()
    ax[0,1].set_xticks(np.nanmean(freq_chunk_arr,axis=0), np.nanmean(freq_chunk_arr,axis=0).astype(int),rotation=25)
    ax[0,1].xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax[0,1].set_title('In-band Spectra : Flux (mJy) per subband vs subband freq (MHz)')
    ax[1,0].hist(Flux_arr)
    ax[1,0].set_title('Histogram of Total flux(mJy)')
    ##############################################################################################################
    max_snr_file = df_.iloc[np.argmax(df_.SNR)].File
    file_name = psr_name+'/good_pfd_b'+str(band_no)+'/'+max_snr_file
    data, F, mjd, dm = pfd_data(str(file_name))
    data = data[:,0]
    profile = data.mean((0,1))
    ax[1,1].plot(profile)
    on_mask = data.mean((0,1))*np.sqrt(data.shape[0]*data.shape[1])/data.std((0,1)) >5
    ax[1,1].set_title('Max SNR Profile')
    ax[1,1].plot(np.arange(0,profile.shape[0])[on_mask], profile[on_mask],'o')
    plt.suptitle('PSR : ' + psr_name + ' ,Band : ' + str(band_no) + ' , Number of epochs : ' + str(df_.shape[0]))
    plt.show()
    
######################################################################################################
def dashboard_1(psr_name, band_no, all_df_file = all_info_df):
    df_ = all_df_file[str(psr_name)]['band'+str(band_no)]
    print('PSR : ', psr_name, ' ,Band : ', str(band_no))
    a_ = df_.Flux_error_snr_subband_freq
    #######################################################################
    full_freq_2d_arr = [i[-1].tolist() for i in a_]
    l0 = np.unique([i for y in full_freq_2d_arr for i in y])
    #grouping_thresh = np.diff(np.sort(np.unique([i[0] for i in full_freq_2d_arr]))).max()
    grouping_thresh = min([np.diff(i).min() for i in full_freq_2d_arr])/2
    full_freq_group = []		# in the end this will 1d array, that is it contains all the group of frequency chunk information
    sublist = [l0[0]]
    for i in range(1, len(l0)):
        if l0[i] - l0[i - 1] < grouping_thresh:
            sublist.append(l0[i])
        else:
            full_freq_group.append(sublist)
            sublist = [l0[i]]
    full_freq_group.append(sublist)
    full_freq_group_hist = [[0 for i in j] for j in full_freq_group]
    full_flux_2d_arr = np.empty( (0,len(full_freq_group)) )		# in the end this will 2d array, that is it contains flux for each frequency chunk over each epoch information
    full_flux_err_2d_arr = np.empty( (0,len(full_freq_group)) )		# in the end this will 2d array, that is it contains flux for each frequency chunk over each epoch information
    for i in a_:
        freq_chunk_ = i[-1].tolist()
        flux_chunk_ = i[0].tolist()
        flux_err_chunk_ = i[1].tolist()
        
        flux_row_nan = np.full(len(full_freq_group), np.nan)
        flux_err_row_nan = np.full(len(full_freq_group), np.nan)
        check_numbers = False
        for index1, freq_checks in enumerate(freq_chunk_):
            for index2, val in enumerate(full_freq_group):
                index_last = np.copy(index2)
                if freq_checks in val:
                    full_freq_group_hist[index2][val.index(freq_checks)] += 1
                    flux_row_nan[index2] = flux_chunk_[index1]
                    flux_err_row_nan[index2] = flux_err_chunk_[index1]
                    break
        full_flux_2d_arr = np.vstack((full_flux_2d_arr, flux_row_nan))
        full_flux_err_2d_arr = np.vstack((full_flux_err_2d_arr, flux_err_row_nan))
    hist_ind = [np.argmax(i) for i in full_freq_group_hist]
    freq_chunk_arr = np.array([np.take(i, j) for i , j in  zip(full_freq_group, hist_ind)])
    #######################################################################
    mjd_arr = df_.mjd
    Flux_arr = df_.Total_Flux
    Flux_error_arr = df_.Flux_error
    fig,ax = plt.subplots(2,2,figsize=(14,12))
    ax[0,0].errorbar(mjd_arr, Flux_arr, yerr=Flux_error_arr, fmt='.')
    ax[0,0].set_title('Total Flux (mJy) vs mjd')
    ax[0,0].semilogy()
    ax[0,1].errorbar(freq_chunk_arr,1e3*np.nanmean(full_flux_2d_arr,axis=0),yerr=1e3*np.sqrt(np.nansum(full_flux_err_2d_arr**2,axis=0))/(~np.isnan(full_flux_err_2d_arr)).sum(0))
    ax[0,1].loglog()
    ax[0,1].set_xticks(freq_chunk_arr, freq_chunk_arr.astype(int),rotation=25)
    ax[0,1].xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax[0,1].set_title('In-band Spectra : Flux (mJy) per subband vs subband freq (MHz)')
    ########################## Freedman-Diaconis?s Rule for histogram bin size
    q1 = Flux_arr.quantile(0.25)
    q3 = Flux_arr.quantile(0.75)
    iqr = q3 - q1
    bin_msp = (2 * iqr) / (len(Flux_arr) ** (1 / 3))
    bin_count = int(np.ceil((Flux_arr.max() - Flux_arr.min()) / bin_msp))
    ax[1,0].hist(Flux_arr, bins= bin_count)
    ax[1,0].set_title('Histogram of Total flux(mJy)')
    ##############################################################################################################
    max_snr_file = df_.iloc[np.argmax(df_.SNR)].File
    file_name = psr_name+'/good_pfd_b'+str(band_no)+'/'+max_snr_file
    data, F, mjd, dm = pfd_data(str(file_name))
    data = data[:,0]
    profile = data.mean((0,1))
    ax[1,1].plot(profile)
    on_mask = data.mean((0,1))*np.sqrt(data.shape[0]*data.shape[1])/data.std((0,1)) >5
    ax[1,1].set_title('Max SNR Profile')
    ax[1,1].plot(np.arange(0,profile.shape[0])[on_mask], profile[on_mask],'o')
    plt.suptitle('PSR : ' + psr_name + ' ,Band : ' + str(band_no) + ' , Number of epochs : ' + str(df_.shape[0]))
    plt.show()
    
######################################################################################################
def dashboard_1_fit_spectra(psr_name, band_no, all_df_file = all_info_df):
    df_ = all_df_file[str(psr_name)]['band'+str(band_no)]
    print('PSR : ', psr_name, ' ,Band : ', str(band_no))
    a_ = df_.Flux_error_snr_subband_freq
    #######################################################################
    full_freq_2d_arr = [i[-1].tolist() for i in a_]
    l0 = np.unique([i for y in full_freq_2d_arr for i in y])
    #grouping_thresh = np.diff(np.sort(np.unique([i[0] for i in full_freq_2d_arr]))).max()
    grouping_thresh = np.nanmin([np.nanmin(np.diff(i)) for i in full_freq_2d_arr])/2
    full_freq_group = []		# in the end this will 1d array, that is it contains all the group of frequency chunk information
    sublist = [l0[0]]
    for i in range(1, len(l0)):
        if l0[i] - l0[i - 1] < grouping_thresh:
            sublist.append(l0[i])
        else:
            full_freq_group.append(sublist)
            sublist = [l0[i]]
    full_freq_group.append(sublist)
    full_freq_group_hist = [[0 for i in j] for j in full_freq_group]
    full_flux_2d_arr = np.empty( (0,len(full_freq_group)) )		# in the end this will 2d array, that is it contains flux for each frequency chunk over each epoch information
    full_flux_err_2d_arr = np.empty( (0,len(full_freq_group)) )		# in the end this will 2d array, that is it contains flux for each frequency chunk over each epoch information
    for i in a_:
        freq_chunk_ = i[-1].tolist()
        flux_chunk_ = i[0].tolist()
        flux_err_chunk_ = i[1].tolist()
        
        flux_row_nan = np.full(len(full_freq_group), np.nan)
        flux_err_row_nan = np.full(len(full_freq_group), np.nan)
        check_numbers = False
        for index1, freq_checks in enumerate(freq_chunk_):
            for index2, val in enumerate(full_freq_group):
                index_last = np.copy(index2)
                if freq_checks in val:
                    full_freq_group_hist[index2][val.index(freq_checks)] += 1
                    flux_row_nan[index2] = flux_chunk_[index1]
                    flux_err_row_nan[index2] = flux_err_chunk_[index1]
                    break
        full_flux_2d_arr = np.vstack((full_flux_2d_arr, flux_row_nan))
        full_flux_err_2d_arr = np.vstack((full_flux_err_2d_arr, flux_err_row_nan))
    hist_ind = [np.argmax(i) for i in full_freq_group_hist]
    freq_chunk_arr = np.array([np.take(i, j) for i , j in  zip(full_freq_group, hist_ind)])
    #######################################################################
    mjd_arr = df_.mjd
    Flux_arr = df_.Total_Flux
    Flux_error_arr = df_.Flux_error
    fig,ax = plt.subplots(2,2,figsize=(14,12))
    ax[0,0].errorbar(mjd_arr, Flux_arr, yerr=Flux_error_arr, fmt='.')
    ax[0,0].set_title('Total Flux (mJy) vs mjd')
    ax[0,0].semilogy()
    AICC = []
    param_arr = []
    param_err_arr = []
    log_freq = np.log10(freq_chunk_arr[~np.all(np.isnan(full_flux_2d_arr),axis=0)])
    flux = 1e3*np.nanmean(full_flux_2d_arr,axis=0)[~np.all(np.isnan(full_flux_2d_arr),axis=0)]
    log_flux = np.log10(flux)
    noise_ = (1e3*np.sqrt(np.nansum(full_flux_err_2d_arr**2,axis=0))/(~np.isnan(full_flux_err_2d_arr)).sum(0))[~np.all(np.isnan(full_flux_2d_arr),axis=0)]
    #return log_freq, log_flux, noise_
    try:
        for n_pow_law_ind in range(1,int(len(log_freq)/2)):
            params_0, bounds_0 = initial_guess_n_bounds_func( n_pow_law_ind, log_freq, log_flux)
            print('THIS IS STEP ', n_pow_law_ind)
            param_, _ = curve_fit(lambda nu_,*param_0: fit_func_st_line(nu_, n_pow_law_ind, *param_0), log_freq, log_flux, p0 = params_0, sigma = noise_/(np.log(10)*flux) , bounds = bounds_0, maxfev = 5000)
            Chi2 = ( ((log_flux -  fit_func_st_line(log_freq, n_pow_law_ind, *param_))/(noise_/(np.log(10)*flux)))**2 ).sum()
            aicc_ = Chi2 + 2*len(param_)*len(log_freq)/(len(log_freq) - len(param_) - 1)
            AICC.append(aicc_)
            param_arr.append(param_)
            param_err_arr.append(_)
            del params_0, bounds_0, Chi2, param_, _, aicc_
    except:
        pass
    ax[0,1].errorbar(freq_chunk_arr,1e3*np.nanmean(full_flux_2d_arr,axis=0),yerr=1e3*np.sqrt(np.nansum(full_flux_err_2d_arr**2,axis=0))/(~np.isnan(full_flux_err_2d_arr)).sum(0))
    print(AICC)
    n_pow_law = int(np.argmin(AICC) + 1)
    opt_param = param_arr[np.argmin(AICC)]
    opt_param_err = np.diag(param_err_arr[np.argmin(AICC)])
    break_arr, spectral_index_arr = opt_param[: n_pow_law -1], opt_param[n_pow_law - 1: -1]
    break_err_arr, spectral_index_err_arr = opt_param_err[: n_pow_law -1], opt_param_err[n_pow_law - 1: -1]
    ax[0,1].plot(10**np.linspace(log_freq[0], log_freq[-1], 1000), 10**fit_func_st_line(np.linspace(log_freq[0], log_freq[-1], 1000), n_pow_law, *opt_param) )
    ax[0,1].loglog()
    ax[0,1].set_xticks(freq_chunk_arr, freq_chunk_arr.astype(int),rotation=25)
    ax[0,1].xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax[0,1].set_title('In-band Spectra : Flux (mJy) per subband vs subband freq (MHz)')
    ########################## Freedman-Diaconis?s Rule for histogram bin size
    q1 = Flux_arr.quantile(0.25)
    q3 = Flux_arr.quantile(0.75)
    iqr = q3 - q1
    bin_msp = (2 * iqr) / (len(Flux_arr) ** (1 / 3))
    bin_count = int(np.ceil((Flux_arr.max() - Flux_arr.min()) / bin_msp))
    ax[1,0].hist(Flux_arr, bins= bin_count)
    ax[1,0].set_title('Histogram of Total flux(mJy)')
    ##############################################################################################################
    max_snr_file = df_.iloc[np.argmax(df_.SNR)].File
    file_name = psr_name+'/good_pfd_b'+str(band_no)+'/'+max_snr_file
    data, F, mjd, dm = pfd_data(str(file_name))
    data = data[:,0]
    profile = data.mean((0,1))
    ax[1,1].plot(profile)
    on_mask = data.mean((0,1))*np.sqrt(data.shape[0]*data.shape[1])/data.std((0,1)) >5
    ax[1,1].set_title('Max SNR Profile')
    ax[1,1].plot(np.arange(0,profile.shape[0])[on_mask], profile[on_mask],'o')
    plt.suptitle('PSR : ' + psr_name + ' ,Band : ' + str(band_no) + ' , Number of epochs : ' + str(df_.shape[0]))
    plt.show()

######################################################################################################
def dashboard_1_fit_spectra_band34(psr_name, all_df_file = all_info_df):
    b34_Total_flux_arr = []
    b34_Total_flux_err_arr = []
    b34_mjd_arr = []
    b34_freq_arr = []
    b34_flux_arr = []
    b34_flux_err_arr = []
    
    for band_no in [3,4]:
        df_ = all_df_file[str(psr_name)]['band'+str(band_no)]
        print('PSR : ', psr_name, ' ,Band : ', str(band_no))
        a_ = df_.Flux_error_snr_subband_freq
        #######################################################################
        full_freq_2d_arr = [i[-1].tolist() for i in a_]
        l0 = np.unique([i for y in full_freq_2d_arr for i in y])
        l0 = l0[np.isfinite(l0)]
        #grouping_thresh = np.diff(np.sort(np.unique([i[0] for i in full_freq_2d_arr]))).max()
        grouping_thresh = np.nanmin([np.nanmin(np.diff(i)) for i in full_freq_2d_arr])/2
        full_freq_group = []		# in the end this will 1d array, that is it contains all the group of frequency chunk information
        sublist = [l0[0]]
        for i in range(1, len(l0)):
            if l0[i] - l0[i - 1] < grouping_thresh:
                sublist.append(l0[i])
            else:
                full_freq_group.append(sublist)
                sublist = [l0[i]]
        full_freq_group.append(sublist)
        full_freq_group_hist = [[0 for i in j] for j in full_freq_group]
        full_flux_2d_arr = np.empty( (0,len(full_freq_group)) )		# in the end this will 2d array, that is it contains flux for each frequency chunk over each epoch information
        full_flux_err_2d_arr = np.empty( (0,len(full_freq_group)) )		# in the end this will 2d array, that is it contains flux for each frequency chunk over each epoch information
        full_snr_2d_arr = np.empty( (0,len(full_freq_group)) )
        for i in a_:
            freq_chunk_ = i[-1].tolist()
            flux_chunk_ = i[0].tolist()
            flux_err_chunk_ = i[1].tolist()
            snr_chunk_ = i[2].tolist()

            flux_row_nan = np.full(len(full_freq_group), np.nan)
            flux_err_row_nan = np.full(len(full_freq_group), np.nan)
            snr_row_nan = np.full(len(full_freq_group), np.nan)
            check_numbers = False
            for index1, freq_checks in enumerate(freq_chunk_):
                for index2, val in enumerate(full_freq_group):
                    index_last = np.copy(index2)
                    if freq_checks in val:
                        full_freq_group_hist[index2][val.index(freq_checks)] += 1
                        flux_row_nan[index2] = flux_chunk_[index1]
                        flux_err_row_nan[index2] = flux_err_chunk_[index1]
                        snr_row_nan[index2] = snr_chunk_[index1]
                        break
            full_flux_2d_arr = np.vstack((full_flux_2d_arr, flux_row_nan))
            full_flux_err_2d_arr = np.vstack((full_flux_err_2d_arr, flux_err_row_nan))
            full_snr_2d_arr = np.vstack((full_snr_2d_arr, snr_row_nan))
        hist_ind = [np.argmax(i) for i in full_freq_group_hist]
        freq_chunk_arr = np.array([np.take(i, j) for i , j in  zip(full_freq_group, hist_ind)])
        b34_Total_flux_arr.extend(df_.Total_Flux.to_list())
        b34_Total_flux_err_arr.extend(df_.Flux_error.to_list())
        b34_mjd_arr.extend(list(df_.mjd))
        b34_freq_arr.extend(list(freq_chunk_arr))
        weight = 'SNR'
        if weight == 'None':
            b34_flux_arr.extend(list(1e3*np.nanmean(full_flux_2d_arr,axis=0)))
        elif weight == 'SNR':
            b34_flux_arr.extend(list(1e3*np.nansum(full_flux_2d_arr*full_snr_2d_arr,axis = 0)/np.nansum(full_snr_2d_arr,axis = 0)))
        b34_flux_err_arr.extend(list(1e3*np.sqrt(np.nansum(full_flux_err_2d_arr**2,axis=0))/(~np.isnan(full_flux_err_2d_arr)).sum(0)))
    #return b34_Total_flux_arr, b34_Total_flux_err_arr, b34_mjd_arr, b34_freq_arr, b34_flux_arr, b34_flux_err_arr
    #######################################################################
    fig,ax = plt.subplots(2,2,figsize=(14,12))
    ax[0,0].errorbar(b34_mjd_arr, b34_Total_flux_arr, yerr=b34_Total_flux_err_arr, fmt='.')
    ax[0,0].set_title('Total Flux (mJy) vs mjd')
    ax[0,0].semilogy()
    AICC = []
    param_arr = []
    param_err_arr = []
    log_freq = np.log10(b34_freq_arr)
    flux = 1e3*np.array(b34_flux_arr)
    log_flux = np.log10(flux)
    noise_ = np.array(b34_flux_err_arr)
    #return log_freq, log_flux, noise_
    try:
        for n_pow_law_ind in range(1,int(len(log_freq)/2)):
            params_0, bounds_0 = initial_guess_n_bounds_func( n_pow_law_ind, log_freq, log_flux)
            print('THIS IS STEP ', n_pow_law_ind)
            param_, _ = curve_fit(lambda nu_,*param_0: fit_func_st_line(nu_, n_pow_law_ind, *param_0), log_freq, log_flux, p0 = params_0, sigma = noise_/(np.log(10)*flux) , bounds = bounds_0, maxfev = 5000)
            Chi2 = ( ((log_flux -  fit_func_st_line(log_freq, n_pow_law_ind, *param_))/(noise_/(np.log(10)*flux)))**2 ).sum()
            aicc_ = Chi2 + 2*len(param_)*len(log_freq)/(len(log_freq) - len(param_) - 1)
            AICC.append(aicc_)
            param_arr.append(param_)
            param_err_arr.append(_)
            del params_0, bounds_0, Chi2, param_, _, aicc_
    except:
        print('SOME ERROR OCCURED')
        pass
    ax[0,1].errorbar(b34_freq_arr,b34_flux_arr,yerr=b34_flux_err_arr,fmt='.')
    print(param_arr)
    n_pow_law = int(np.argmin(AICC) + 1)
    opt_param = param_arr[np.argmin(AICC)]
    opt_param_err = np.diag(param_err_arr[np.argmin(AICC)])
    break_arr, spectral_index_arr = opt_param[: n_pow_law -1], opt_param[n_pow_law - 1: -1]
    break_err_arr, spectral_index_err_arr = opt_param_err[: n_pow_law -1], opt_param_err[n_pow_law - 1: -1]
    #ax[0,1].plot(10**np.linspace(log_freq[0], log_freq[-1], 1000), 10**fit_func_st_line(np.linspace(log_freq[0], log_freq[-1], 1000), n_pow_law, *opt_param) )
    ax[0,1].loglog()
    ax[0,1].set_xticks(np.array(b34_freq_arr)[::len(b34_freq_arr)//5], np.array(b34_freq_arr).astype(int)[::len(b34_freq_arr)//5],rotation=25)
    ax[0,1].xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax[0,1].set_title('In-band Spectra : Flux (mJy) per subband vs subband freq (MHz)')
    ########################## Freedman-Diaconis?s Rule for histogram bin size
    q1 = np.quantile(b34_Total_flux_arr, 0.25)
    q3 = np.quantile(b34_Total_flux_arr, 0.75)
    iqr = q3 - q1
    bin_msp = (2 * iqr) / (len(b34_Total_flux_arr) ** (1 / 3))
    bin_count = int(np.ceil((max(b34_Total_flux_arr) - min(b34_Total_flux_arr)) / bin_msp))
    ax[1,0].hist(b34_Total_flux_arr, bins= bin_count)
    ax[1,0].set_title('Histogram of Total flux(mJy)')
    ##############################################################################################################
    '''
    max_snr_file = df_.iloc[np.argmax(df_.SNR)].File
    file_name = psr_name+'/good_pfd_b'+str(band_no)+'/'+max_snr_file
    data, F, mjd, dm = pfd_data(str(file_name))
    data = data[:,0]
    profile = data.mean((0,1))
    ax[1,1].plot(profile)
    on_mask = data.mean((0,1))*np.sqrt(data.shape[0]*data.shape[1])/data.std((0,1)) >5
    ax[1,1].set_title('Max SNR Profile')
    ax[1,1].plot(np.arange(0,profile.shape[0])[on_mask], profile[on_mask],'o')
    '''
    plt.suptitle('PSR : ' + psr_name + ' ,Band 3 & 4 ' + ' , Number of epochs : ' + str(len(b34_mjd_arr)))
    plt.show()
    return AICC, param_arr, param_err_arr

######################################################################################################
def dashboard_2(psr_name,band_no,all_df_file = all_info_df):
    df_ = all_df_file[str(psr_name)]['band'+str(band_no)]
    print('PSR : ', psr_name, ' ,Band : ', str(band_no))
    a_ = df_.Flux_error_snr_subband_freq
    #######################################################################
    full_freq_2d_arr = [i[-1].tolist() for i in a_]
    l0 = np.unique([i for y in full_freq_2d_arr for i in y])
    #l0 = np.unique(full_freq_2d_arr)
    #grouping_thresh = np.diff(np.sort(np.unique([i[0] for i in full_freq_2d_arr]))).max()
    grouping_thresh = min([np.diff(i).min() for i in full_freq_2d_arr])/2
    full_freq_group = []		# in the end this will 1d array, that is it contains all the group of frequency chunk information
    sublist = [l0[0]]
    for i in range(1, len(l0)):
        if l0[i] - l0[i - 1] < grouping_thresh:
            sublist.append(l0[i])
        else:
            full_freq_group.append(sublist)
            sublist = [l0[i]]
    full_freq_group.append(sublist)
    full_freq_group_hist = [[0 for i in j] for j in full_freq_group]
    full_flux_2d_arr = np.empty( (0,len(full_freq_group)) )		# in the end this will 2d array, that is it contains flux for each frequency chunk over each epoch information
    full_flux_err_2d_arr = np.empty( (0,len(full_freq_group)) )		# in the end this will 2d array, that is it contains flux for each frequency chunk over each epoch information
    full_snr_2d_arr = np.empty( (0,len(full_freq_group)) )		# in the end this will 2d array, that is it contains flux for each frequency chunk over each epoch information
    for i in a_:
        freq_chunk_ = i[-1].tolist()
        flux_chunk_ = i[0].tolist()
        flux_err_chunk_ = i[1].tolist()
        snr_chunk_ = i[2].tolist()
        
        flux_row_nan = np.full(len(full_freq_group), np.nan)
        flux_err_row_nan = np.full(len(full_freq_group), np.nan)
        snr_row_nan = np.full(len(full_freq_group), np.nan)
        check_numbers = False
        for index1, freq_checks in enumerate(freq_chunk_):
            for index2, val in enumerate(full_freq_group):
                index_last = np.copy(index2)
                if freq_checks in val:
                    full_freq_group_hist[index2][val.index(freq_checks)] += 1
                    flux_row_nan[index2] = flux_chunk_[index1]
                    flux_err_row_nan[index2] = flux_err_chunk_[index1]
                    snr_row_nan[index2] = snr_chunk_[index1]
                    break
        full_flux_2d_arr = np.vstack((full_flux_2d_arr, flux_row_nan))
        full_flux_err_2d_arr = np.vstack((full_flux_err_2d_arr, flux_err_row_nan))
        full_snr_2d_arr = np.vstack((full_snr_2d_arr, snr_row_nan))
    hist_ind = [np.argmax(i) for i in full_freq_group_hist]
    freq_chunk_arr = np.array([np.take(i, j) for i , j in  zip(full_freq_group, hist_ind)])
    #######################################################################
    mjd_arr = df_.mjd
    Flux_arr = df_.Total_Flux
    Flux_error_arr = df_.Flux_error
    fig,ax = plt.subplots(2,2,figsize=(14,12))
    ax[0,0].errorbar(mjd_arr, Flux_arr, yerr=Flux_error_arr, fmt='.')
    ax[0,0].set_title('Total Flux (mJy) vs mjd')
    ax[0,0].semilogy()
    snr_weighted_full_flux_1d_arr = np.nansum(full_flux_2d_arr*full_snr_2d_arr**2,axis = 0)/np.nansum(full_snr_2d_arr**2,axis = 0)
    snr_weighted_full_flux_err_1d_arr = np.nansum(full_flux_2d_arr*full_snr_2d_arr**2,axis = 0)/np.nansum(full_snr_2d_arr**2,axis = 0)
    ax[0,1].errorbar(freq_chunk_arr,1e3*snr_weighted_full_flux_1d_arr,yerr=1e3*np.sqrt(np.nansum(full_flux_err_2d_arr**2,axis=0))/(~np.isnan(full_flux_err_2d_arr)).sum(0))
    ax[0,1].loglog()
    ax[0,1].set_xticks(freq_chunk_arr, freq_chunk_arr.astype(int),rotation=25)
    ax[0,1].xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax[0,1].set_title('In-band Spectra : Flux (mJy) per subband vs subband freq (MHz)')
    ########################## Freedman-Diaconis?s Rule for histogram bin size
    q1 = Flux_arr.quantile(0.25)
    q3 = Flux_arr.quantile(0.75)
    iqr = q3 - q1
    bin_msp = (2 * iqr) / (len(Flux_arr) ** (1 / 3))
    bin_count = int(np.ceil((Flux_arr.max() - Flux_arr.min()) / bin_msp))
    ax[1,0].hist(Flux_arr, bins= bin_count)
    ax[1,0].set_title('Histogram of Total flux(mJy)')
    ##############################################################################################################
    max_snr_file = df_.iloc[np.argmax(df_.SNR)].File
    file_name = psr_name+'/good_pfd_b'+str(band_no)+'/'+max_snr_file
    data, F, mjd, dm = pfd_data(str(file_name))
    data = data[:,0]
    profile = data.mean((0,1))
    ax[1,1].plot(profile)
    on_mask = data.mean((0,1))*np.sqrt(data.shape[0]*data.shape[1])/data.std((0,1)) >5
    ax[1,1].set_title('Max SNR Profile')
    ax[1,1].plot(np.arange(0,profile.shape[0])[on_mask], profile[on_mask],'o')
    plt.suptitle('PSR : ' + psr_name + ' ,Band : ' + str(band_no) + ' , Number of epochs : ' + str(df_.shape[0]))
    plt.show()


def dashboard_time_series_hist(psr_name, band_no, scale = 'linear', all_df_file = all_info_df):
    df_ = all_info_df[psr_name]['band'+str(band_no)]
    dm_ = np.median(df_.dm)
    Flux_arr = df_.Total_Flux.to_numpy(float)
    Flux_error_arr = df_.Flux_error.to_numpy(float)
    if scale == 'log':
        Flux_arr_scaled = np.log10(Flux_arr)
        Flux_error_arr_scaled = np.log10(1 + Flux_error_arr/Flux_arr)
    else:
        Flux_arr_scaled = Flux_arr
        Flux_error_arr_scaled = Flux_error_arr
    mjd_arr = df_.mjd
    fig, ax = plt.subplots(1, 2, sharey = True)
    plt.subplots_adjust(wspace=0.0)
    ###############################################
    ax[0].errorbar(mjd_arr, Flux_arr_scaled, yerr=Flux_error_arr_scaled, fmt='.')
    #ax[0].set_title('Total Flux (mJy) vs mjd')
    ###############################################
    q1 = np.quantile(Flux_arr_scaled, 0.25)
    q3 = np.quantile(Flux_arr_scaled, 0.75)
    iqr = q3 - q1
    bin_msp = (2 * iqr) / (len(Flux_arr_scaled) ** (1 / 3))
    bin_count = int(np.ceil((Flux_arr_scaled.max() - Flux_arr_scaled.min()) / bin_msp))
    ax[1].hist(Flux_arr_scaled, bins= bin_count, orientation='horizontal')
    plt.suptitle('PSR : ' + psr_name + ' Band : ' + str(band_no) + ' Number of epochs : ' + str(df_.shape[0]) + ' DM :' + str(dm_))
    plt.show()

###################### Checks for the dashboard for each pulsar

for i in all_psr_names:
    print('WORKING on : ', i)
    try:
        dashboard_1(i,5)
        print('COMPLETE')
    except:
        print('NOT COMPLETE')
    print('##################################################################################################################################################')


hist_msp = ['J1828+0625','J1646-2142','J2144-5237','J1120-3618']

def plot_hist_0(msp_list, band_n):
    fig, ax = plt.subplots(len(msp_list)//2, 2, figsize=(14,12))
    for ind, msp_ in enumerate(msp_list):
        df_ = all_info_df[msp_]['band'+str(band_n)]
        dm_ = np.median(df_.dm)
        Flux_arr = df_.Total_Flux
        q1 = Flux_arr.quantile(0.25)
        q3 = Flux_arr.quantile(0.75)
        iqr = q3 - q1
        bin_msp = (2 * iqr) / (len(Flux_arr) ** (1 / 3))
        bin_count = int(np.ceil((Flux_arr.max() - Flux_arr.min()) / bin_msp))
        ax[ind//2, ind%2].hist(Flux_arr, bins= bin_count)
        ax[ind//2, ind%2].set_title('PSR : ' + msp_ + ' ,Band : ' + str(band_n) + ' , Number of epochs : ' + str(df_.shape[0]) + ' , DM :' + str(dm_))
        ax[ind//2, ind%2].set_xlabel('Total Flux (mJy)', fontsize = 10)
        ax[ind//2, ind%2].set_ylabel('Number of epochs', fontsize = 10)
    plt.suptitle('Histogram of Total flux(mJy)')
    plt.show()



def plot_hist_1(msp_list, band_n, size_):
    fig = plt.subplot()
    if not isinstance(msp_list, list):
        msp_list = [msp_list]

    for ind, msp_ in enumerate(msp_list):
        plt.subplot(int(len(msp_list)//2 + (len(msp_list) == 1)), 1 + (len(msp_list) != 1), ind + 1)
        df_ = all_info_df[msp_]['band'+str(band_n)]
        dm_ = np.median(df_.dm)
        Flux_arr = df_.Total_Flux
        q1 = Flux_arr.quantile(0.25)
        q3 = Flux_arr.quantile(0.75)
        iqr = q3 - q1
        bin_msp = (2 * iqr) / (len(Flux_arr) ** (1 / 3))
        bin_count = int(np.ceil((Flux_arr.max() - Flux_arr.min()) / bin_msp))
        label_ = 'PSR : ' + msp_ + '\n Band : ' + str(band_n) + '\n Number of epochs : ' + str(df_.shape[0]) + '\n DM :' + str(dm_)
        plt.hist(Flux_arr, bins= bin_count, label = label_)
        plt.legend(fontsize = size_)
        plt.xlabel('Total Flux (mJy)', fontsize = size_)
        plt.ylabel('Number of epochs', fontsize = size_)
        plt.tick_params(axis='both',labelsize = size_)

    plt.suptitle('Histogram of Total flux(mJy)', fontsize = size_)
    plt.show()

plot_hist_1(hist_msp,3,12)



def plot_mjd_fig_0(msp_list, band_n, size_):
    fig, ax = plt.subplots(len(msp_list)//2, 2, figsize=(14,12))
    fig.subplots_adjust(wspace=0.3, hspace=0.4)
    for ind, msp_ in enumerate(msp_list):
        df_ = all_info_df[msp_]['band'+str(band_n)]
        Flux_arr = df_.Total_Flux
        Flux_error_arr = df_.Flux_error
        mjd_arr = df_.mjd
        dm_ = np.median(df_.dm)
        q1 = Flux_arr.quantile(0.25)
        q3 = Flux_arr.quantile(0.75)
        iqr = q3 - q1
        bin_msp = (2 * iqr) / (len(Flux_arr) ** (1 / 3))
        bin_count = int(np.ceil((Flux_arr.max() - Flux_arr.min()) / bin_msp))
        label_ = 'PSR : ' + msp_ + ' ,Band : ' + str(band_n) + ' ,Number of epochs : ' + str(df_.shape[0]) + '\n DM :' + str(dm_)
        ax[ind//2, ind%2].errorbar(mjd_arr, Flux_arr, yerr=Flux_error_arr, fmt='.')
        ax[ind//2, ind%2].set_title(label_, fontsize = size_)
        ax[ind//2, ind%2].set_ylabel('Total Flux (mJy)', fontsize = size_)
        ax[ind//2, ind%2].set_xlabel('MJD', fontsize = size_)
        ax[ind//2, ind%2].tick_params(axis='both',labelsize = size_)
        #ax[ind//2, ind%2].semilogy()
    plt.show()

def plot_mjd_fig_1(msp_list, band_n, size_):
    fig = plt.subplot()
    if not isinstance(msp_list, list):
        msp_list = [msp_list]
    for ind, msp_ in enumerate(msp_list):
        ax = plt.subplot(int(len(msp_list)//2 + (len(msp_list) == 1)), 1 + (len(msp_list) != 1), ind + 1)
        try:
            df_ = all_info_df[msp_]['band'+str(band_n)]
        except:
            continue
        Flux_arr = df_.Total_Flux
        Flux_error_arr = df_.Flux_error
        mjd_arr = df_.mjd
        dm_ = np.median(df_.dm)
        q1 = Flux_arr.quantile(0.25)
        q3 = Flux_arr.quantile(0.75)
        iqr = q3 - q1
        bin_msp = (2 * iqr) / (len(Flux_arr) ** (1 / 3))
        bin_count = int(np.ceil((Flux_arr.max() - Flux_arr.min()) / bin_msp))
        label_ = 'PSR : ' + msp_ + '\nNumber of epochs : ' + str(df_.shape[0]) + '\nDM :' + str(dm_)
        plt.errorbar(mjd_arr, Flux_arr, yerr=Flux_error_arr, fmt='.', label = label_)
        plt.legend(fontsize = size_)
        plt.ylabel('Total Flux (mJy)', fontsize = size_)
        plt.xlabel('MJD', fontsize = size_)
        plt.tick_params(axis='both',labelsize = size_)
        #plt.set_xticks(mjd_arr, Time(mjd_arr.astype(float),format='mjd').strftime("%y-%b-%d").astype(str), rotation=30)
        plt.tick_params(axis='x', rotation=20)
        ax.label_outer()
        plt.semilogy()
        del Flux_arr, Flux_error_arr, mjd_arr, dm_, q1, q3, iqr, bin_msp, bin_count, label_
    plt.suptitle('Total flux vs MJD for ' + 'Band : ' + str(band_n), fontsize = size_)
    #plt.tight_layout(pad=3)
    plt.show()



plot_mjd_fig(hist_msp,3,15)

############################################################################################################################

def fit_structure_a(x_, a, x0, c):
    return 2*(1 - np.exp(-(x_/x0)**a)) + c

def fit_structure_b(x_, a, x0, c, k):
    return 2*k*(1 - np.exp(-(x_/x0)**a)) + c

def fit_structure_c(x_, a, x0, c, k):
    '''
    a = alpha also known as the slope of the structure function
    x0= some factor times refractive time scale
    c = dc level of structure function of linear scale
    k = difference between the noise and saturation level of structure function
    '''
    return 10**(2*k*(1-np.exp(-(x_/x0)**a)) + c)

def fit_structure_d(x_, a, x0, c, k):
    '''
    a = alpha also known as the slope of the structure function
    x0= some factor times refractive time scale
    c = dc level of structure function of linear scale
    k = difference between the noise and saturation level of structure function
    '''
    return 2*k*(1-np.exp(-(x_/x0)**a)) + c

# structure_function_total_flux_0 uses Flux_error_rms_1d for errors

def structure_function_total_flux_0(msp_cand, band_n):
    mjd_ = all_info_df[msp_cand]['band'+str(band_n)].mjd.to_numpy(int)
    fav_ind = np.triu_indices(len(mjd_), k=1)
    mjd_diff = abs(mjd_[:, None] - mjd_[None, :])[fav_ind]

    Flux_1d = all_info_df[msp_cand]['band'+str(band_n)].Total_Flux.to_numpy(float)
    Flux_error_1d = all_info_df[msp_cand]['band'+str(band_n)].Flux_error.to_numpy(float)
    Flux_error_rms_1d = np.sqrt(Flux_error_1d**2 - (0.1*Flux_1d)**2)
    print('step -1')
    Flux_diff_2d = np.triu(Flux_1d[:, None] - Flux_1d[None, :], k=1)
    Flux_error_diff_2d = np.triu(Flux_error_1d[:, None] - Flux_error_1d[None, :], k=1)
    Flux_error_rms_diff_2d = np.triu(Flux_error_rms_1d[None, :] - Flux_error_rms_1d[None, :], k=1)
    print('step -2')
    mat1 = (Flux_diff_2d**2)[fav_ind]
    mat2 = abs(Flux_error_rms_1d[:, None]**2 + Flux_error_rms_1d[None, :]**2)[fav_ind]
    mat3 = 2*np.triu(Flux_error_rms_diff_2d * Flux_diff_2d, k=1)[fav_ind]
    mat4 = -2*np.triu(Flux_error_rms_1d[:, None] * Flux_error_rmall_info_df,s_1d[None, :], k=1)[fav_ind]
    #mat5 = 
    mat6 = 4*((Flux_diff_2d * Flux_error_rms_1d[:,None])**2)[fav_ind]
    mat7 = mat4**2
    print('step 1')
    bins=np.geomspace(mjd_diff.min(), mjd_diff.max(), int(len(np.unique(mjd_diff))/10))
    mjd_hist_bin_array = []		# 2d array with rows , columns = mjd, delay bins of the histogram
    #l = ['a'+str(i) for i in range(len(mjd_))]
    for mjd_val in mjd_:
        bool_arr = mjd_val == mjd_
        mat_2d_mjd_ = bool_arr[:,None] + bool_arr[None, :]
        mjd_hist_, _ = np.histogram(mjd_diff, bins=bins, weights=np.triu(mat_2d_mjd_.astype(int), k=1)[fav_ind])
        mjd_hist_bin_array.append(mjd_hist_**2)
    mjd_hist_bin_array = np.array(mjd_hist_bin_array)
    error_term1_n = np.sum(Flux_error_rms_1d[:, None]**4 * mjd_hist_bin_array , axis=0)
    y_structure_function, x = np.histogram(mjd_diff, bins=bins, weights=(mat1 + mat2 + mat3 + mat4))
    print('step 2')
    y_N_p, x_ = np.histogram(mjd_diff, bins=bins)
    print('step 3')
    y_error_sf, x = np.histogram(mjd_diff, bins=bins, weights= mat6+mat7 )
    final_y_error = (error_term1_n + y_error_sf)/(y_N_p**2)
    #plt.plot((x[1:] + x[:-1])/2, y/y_, 'o')
    plt.errorbar((x[1:] + x[:-1])/2, y_structure_function/(y_N_p * Flux_1d.mean()**2), yerr = final_y_error,fmt='.')
    print('step 4')
    plt.loglog()
    plt.show()


def histedges_equalA(x_, nbin):
        pow = 0.5
        dx = np.diff(np.sort(x_))
        tmp = np.cumsum(dx ** pow)
        tmp = np.pad(tmp, (1, 0), 'constant')
        return np.interp(np.linspace(0, tmp.max(), nbin + 1), tmp, np.sort(x_))





# structure_function_total_flux_1 uses Flux_error_1d for errors

def structure_function_total_flux_1a(msp_cand, band_n, fit=True, n_bin = 10, save_fig = True, mjd_mask = None):
    '''
    Fits fit_structure_a in log space that is : log D(tau) vs log tau space
    ''' 
    if mjd_mask is None:
        mjd_ = all_info_df[msp_cand]['band'+str(band_n)].mjd.to_numpy(int)
        Flux_1d = all_info_df[msp_cand]['band'+str(band_n)].Total_Flux.to_numpy(float)
        Flux_error_1d = all_info_df[msp_cand]['band'+str(band_n)].Flux_error.to_numpy(float)
    else:
        mjd_ = all_info_df[msp_cand]['band'+str(band_n)].mjd.to_numpy(int)[mjd_mask]
        Flux_1d = all_info_df[msp_cand]['band'+str(band_n)].Total_Flux.to_numpy(float)[mjd_mask]
        Flux_error_1d = all_info_df[msp_cand]['band'+str(band_n)].Flux_error.to_numpy(float)[mjd_mask]

    fav_ind = np.triu_indices(len(mjd_), k=1)
    mjd_diff = abs(mjd_[:, None] - mjd_[None, :])[fav_ind]

    Flux_error_rms_1d = np.sqrt(Flux_error_1d**2 - (0.1*Flux_1d)**2)
    
    Flux_diff_2d = np.triu(Flux_1d[:, None] - Flux_1d[None, :], k=1)
    Flux_error_diff_2d = np.triu(Flux_error_1d[:, None] - Flux_error_1d[None, :], k=1)
    Flux_error_rms_diff_2d = np.triu(Flux_error_rms_1d[None, :] - Flux_error_rms_1d[None, :], k=1)
    
    mat1 = (Flux_diff_2d**2)[fav_ind]
    mat2 = abs(Flux_error_1d[:, None]**2 + Flux_error_1d[None, :]**2)[fav_ind]
    mat3 = 2*np.triu(Flux_error_diff_2d * Flux_diff_2d, k=1)[fav_ind]
    mat4 = -2*np.triu(Flux_error_1d[:, None] * Flux_error_1d[None, :], k=1)[fav_ind]
    mat6 = 4*((Flux_diff_2d * Flux_error_1d[:,None])**2)[fav_ind]
    mat7 = mat4**2

    if isinstance(n_bin, int):
        #bins=np.geomspace(1, mjd_diff.max(), int(len(mjd_diff)/n_bin))
        #bins=np.geomspace(1, mjd_diff.max(), int(n_bin +1) )
        #	bins = np.percentile(mjd_diff, np.linspace(0.1, 100, int(n_bin+1)))		# bins edges with same number of elements in each bins in linear space
        #	bins = 10**np.percentile(np.log10(mjd_diff[mjd_diff != 0]), np.linspace(0.1, 100, int(n_bin+1)))		# bins edges with same number of elements in each bins in log space
        bins=np.geomspace(mjd_diff[mjd_diff != 0].min(), mjd_diff.max(), int(n_bin +1) )
        #bins=np.linspace(mjd_diff[mjd_diff != 0].min(), mjd_diff.max(), int(n_bin +1) )
    else:
    	#bins=np.geomspace(1, mjd_diff.max(), int(2*len(mjd_)+1))
    	bins=np.geomspace(mjd_diff[mjd_diff != 0].min(), mjd_diff.max(), int(2*len(mjd_)+1))
        #bins=np.linspace(mjd_diff[mjd_diff != 0].min(), mjd_diff.max(), int(len(mjd_)+1) )
    #bins = histedges_equalA(mjd_diff, int(2*len(mjd_)+1))
    mjd_hist_bin_array = []		# 2d array with rows , columns = mjd, delay bins of the histogram
    for mjd_val in mjd_:
        bool_arr = mjd_val == mjd_
        mat_2d_mjd_ = bool_arr[:,None] + bool_arr[None, :]
        mjd_hist_, _ = np.histogram(mjd_diff, bins=bins, weights=np.triu(mat_2d_mjd_.astype(int), k=1)[fav_ind])
        mjd_hist_bin_array.append(mjd_hist_**2)
        
    mjd_hist_bin_array = np.array(mjd_hist_bin_array); print(mjd_hist_bin_array.shape)
    error_term1_n = np.sum(Flux_error_1d[:, None]**4 * mjd_hist_bin_array , axis=0)
    y_structure_function, x = np.histogram(mjd_diff, bins=bins, weights=mat1)
    y_N_p, x = np.histogram(mjd_diff, bins=bins)
    y_error_sf, x = np.histogram(mjd_diff, bins=bins, weights= mat6+mat7 )
    final_y_error = np.sqrt((error_term1_n + y_error_sf)/((np.median(Flux_1d)**2)*(y_N_p**2)))
    y_structure_function /= (y_N_p * np.median(Flux_1d)**2)
    remove_1_bins_info = False
    if remove_1_bins_info:
        final_y_error[y_N_p == 1] = np.nan
        y_structure_function[y_N_p == 1] = np.nan
        y_N_p[y_N_p == 1] = 0
    #print('np.median(Flux_1d) : ', np.median(Flux_1d))
    #print('y_error_sf : ', y_error_sf)
    #print('error_term1_n : ',error_term1_n)
    print('final_y_error : ',final_y_error)
    print('y_N_p : ',y_N_p)
    print('y_structure_function : ',  y_structure_function)
    print('x : ', x)
    #return bins, y_structure_function, final_y_error


    #plt.plot((x[1:] + x[:-1])/2, y/y_, 'o')
    #plt.errorbar((x[1:] + x[:-1])/2, y_structure_function/(y_N_p * Flux_1d.mean()**2), yerr = final_y_error,fmt='.')

    #plt.hist(mjd_diff, bins=bins); plt.plot(bins, 10*np.ones_like(bins), 'o')
    
    if fit:
        #log_delay_, log_struct_func_, log_struct_func_error = np.log10((x[1:] + x[:-1])/2), np.log10(y_structure_function), np.log10(1 + final_y_error/y_structure_function)
        log_delay_, log_struct_func_, log_struct_func_error = np.sqrt(np.log10(x[1:]) * np.log10(x[:-1])), np.log10(y_structure_function), np.log10(1 + final_y_error/y_structure_function)
        #delay_, struct_func_ = (x[1:] + x[:-1])/2, y_structure_function
        p_20 = [1, log_delay_.mean(), np.nanmin(log_struct_func_)]
        print('p_20 : ', p_20)
        #p_20 = [1, delay_.mean(), np.nanmin(struct_func_), 1]
        #p_2, _ = curve_fit(fit_structure_a, log_delay_, log_struct_func_, p0 = p_20, sigma = (final_y_error/(np.log(10)*10**log_struct_func_))[~np.isnan(final_y_error)] , bounds = [[0, 0, -np.inf], [np.inf]*3], maxfev = 5000, nan_policy='omit')
        #p_2, _ = curve_fit(fit_structure_a, log_delay_, log_struct_func_, p0 = p_20, sigma = log_struct_func_error[~np.isnan(final_y_error)] , bounds = [[0, 0, -np.inf], [np.inf]*3], maxfev = 5000, nan_policy='omit')
        p_2, _ = curve_fit(fit_structure_a, log_delay_, log_struct_func_, p0 = p_20, bounds = [[0, 0, -np.inf], [np.inf]*3], maxfev = 5000, nan_policy='omit')
        #p_2, _ = curve_fit(fit_structure_a, delay_, struct_func_, p0 = p_20, bounds = [[0, 0, -np.inf, 0], [np.inf]*4], maxfev = 5000, nan_policy='omit')
        #print('Parameter : x0 (in days) :', 10**p_2[1], r'$\alpha$ : ', p_2[0], 'c : ', p_2[2])
        print(r'Parameter : $\alpha$ :' + "{0} +- {1} ".format(round(p_2[0], 3), round(np.diag(_)[0],3)) + ' ; x0 : ' + "{0} +- {1} ".format(round(10**p_2[1], 3), round(((10**np.diag(_)[1]) - 1)*10**p_2[1],3)) + ' ; c : ' + "{0} +- {1} ".format(round(p_2[2], 3), round(np.diag(_)[2],3)))
        plt.plot(10**log_delay_[~np.isnan(final_y_error)],10**fit_structure_a(log_delay_[~np.isnan(final_y_error)], *p_2))
        ###plt.plot(np.logspace(0,3),10**fit_structure_a(np.linspace(0,3), *p_2))
        Tau_r = 10**p_2[1]
        #Tau_r_error = (np.diag(_)[1])*np.log(10)*Tau_r
        Tau_r_error = ((10**np.diag(_)[1]) - 1)*Tau_r
        Chi_2 = np.sum(((log_struct_func_[~np.isnan(final_y_error)] - fit_structure_a(log_delay_[~np.isnan(final_y_error)], *p_2))/(log_struct_func_error[~np.isnan(final_y_error)]))**2)/(np.nansum(~np.isnan(final_y_error)) - 3)
        #plt.plot(delay_[~np.isnan(final_y_error)], fit_structure_a(delay_[~np.isnan(final_y_error)], *p_2))
    plt.errorbar((x[1:] + x[:-1])/2, y_structure_function, yerr = final_y_error,fmt='.')
    plt.vlines(Tau_r, ymin=10**np.nanmin(log_struct_func_), ymax=10**np.nanmax(log_struct_func_))
    plt.fill([min(max(Tau_r - Tau_r_error, x[0]),Tau_r), min(max(Tau_r - Tau_r_error, x[0]), Tau_r), max(min(Tau_r + Tau_r_error, x[-1]),Tau_r), max(min(Tau_r + Tau_r_error,x[-1]),Tau_r)], [10**np.nanmin(log_struct_func_), 10**np.nanmax(log_struct_func_), 10**np.nanmax(log_struct_func_), 10**np.nanmin(log_struct_func_)], alpha = 0.5)
    my_title = 'MSP : ' + str(msp_cand) + ' , no of epochs : ' + str(len(mjd_)) + ' , no of delays : ' + str(len(bins) - 1) + '\nParameter : ' + r'$\tau_r$ (in days) : ' + str(round(Tau_r, 2)) + r'$\pm$' + str(round(Tau_r_error, 2)) + r' ; $\alpha$ : ' + str(round(p_2[0], 2))+ r'$\pm$' + str(round(np.diag(_)[0], 2))
    Bhaswati_title = str(msp_cand) + ', no of epochs : ' + str(len(mjd_)) + '\n' + r'$\tau_r$ (in days) : ' + str(round(Tau_r, 2)) + r'$\pm$' + str(round(Tau_r_error, 2)) + r'; $\alpha_0$ : ' + str(round(p_2[0], 2))+ r'$\pm$' + str(round(np.diag(_)[0], 2))
    plt.title(Bhaswati_title)
    #plt.title('MSP : ' + str(msp_cand) + ' , no of epochs : ' + str(len(mjd_)) + ' , no of delays : ' + str(len(bins) - 1) + '\nParameter : ' + r'$\tau_r$ (in days) : ' + str(round(Tau_r, 2)) + r'$\pm$' + str(round(Tau_r_error, 2)) + r' ; $\alpha$ : ' + str(round(p_2[0], 2))+ r'$\pm$' + str(round(np.diag(_)[0], 2)) + r' ; $\chi^2_{red}$ : ' + str(round(Chi_2, 3)))
    plt.xlabel(r'$\tau$')
    plt.ylabel(r'$D(\tau)$')
    plt.loglog()
    if save_fig:
        file_name = str((msp_cand.split('-')*bool(len(msp_cand.split('-'))- 1) + msp_cand.split('+')*bool(len(msp_cand.split('+'))- 1))[0]) + '_struct_func'
        plt.savefig(file_name, dpi=200)
    plt.show()

def structure_function_total_flux_1b(msp_cand, band_n, fit=True, n_bin = 10, save_fig = True, mjd_mask = None):
    '''
    Fits fit_structure_a in log space that is : D(tau) vs tau space
    ''' 
    if mjd_mask is None:
        mjd_ = all_info_df[msp_cand]['band'+str(band_n)].mjd.to_numpy(int)
        Flux_1d = all_info_df[msp_cand]['band'+str(band_n)].Total_Flux.to_numpy(float)
        Flux_error_1d = all_info_df[msp_cand]['band'+str(band_n)].Flux_error.to_numpy(float)
    else:
        mjd_ = all_info_df[msp_cand]['band'+str(band_n)].mjd.to_numpy(int)[mjd_mask]
        Flux_1d = all_info_df[msp_cand]['band'+str(band_n)].Total_Flux.to_numpy(float)[mjd_mask]
        Flux_error_1d = all_info_df[msp_cand]['band'+str(band_n)].Flux_error.to_numpy(float)[mjd_mask]

    fav_ind = np.triu_indices(len(mjd_), k=1)
    mjd_diff = abs(mjd_[:, None] - mjd_[None, :])[fav_ind]

    Flux_error_rms_1d = np.sqrt(Flux_error_1d**2 - (0.1*Flux_1d)**2)
    
    Flux_diff_2d = np.triu(Flux_1d[:, None] - Flux_1d[None, :], k=1)
    Flux_error_diff_2d = np.triu(Flux_error_1d[:, None] - Flux_error_1d[None, :], k=1)
    Flux_error_rms_diff_2d = np.triu(Flux_error_rms_1d[None, :] - Flux_error_rms_1d[None, :], k=1)
    
    mat1 = (Flux_diff_2d**2)[fav_ind]
    mat2 = abs(Flux_error_1d[:, None]**2 + Flux_error_1d[None, :]**2)[fav_ind]
    mat3 = 2*np.triu(Flux_error_diff_2d * Flux_diff_2d, k=1)[fav_ind]
    mat4 = -2*np.triu(Flux_error_1d[:, None] * Flux_error_1d[None, :], k=1)[fav_ind]
    mat6 = 4*((Flux_diff_2d * Flux_error_1d[:,None])**2)[fav_ind]
    mat7 = mat4**2

    if isinstance(n_bin, int):
        #bins=np.geomspace(1, mjd_diff.max(), int(len(mjd_diff)/n_bin))
        bins=np.geomspace(1, mjd_diff.max(), int(n_bin +1) )
        #bins=np.geomspace(mjd_diff[mjd_diff != 0].min(), mjd_diff.max(), int(n_bin +1) )
        #bins=np.linspace(mjd_diff[mjd_diff != 0].min(), mjd_diff.max(), int(n_bin +1) )
    else:
    	bins=np.geomspace(1, mjd_diff.max(), int(2*len(mjd_)+1))
    	#bins=np.geomspace(mjd_diff[mjd_diff != 0].min(), mjd_diff.max(), int(2*len(mjd_)+1))
        #bins=np.linspace(mjd_diff[mjd_diff != 0].min(), mjd_diff.max(), int(len(mjd_)+1) )
    #bins = histedges_equalA(mjd_diff, int(2*len(mjd_)+1))
    mjd_hist_bin_array = []		# 2d array with rows , columns = mjd, delay bins of the histogram
    for mjd_val in mjd_:
        bool_arr = mjd_val == mjd_
        mat_2d_mjd_ = bool_arr[:,None] + bool_arr[None, :]
        mjd_hist_, _ = np.histogram(mjd_diff, bins=bins, weights=np.triu(mat_2d_mjd_.astype(int), k=1)[fav_ind])
        mjd_hist_bin_array.append(mjd_hist_**2)
        
    mjd_hist_bin_array = np.array(mjd_hist_bin_array); print(mjd_hist_bin_array.shape)
    error_term1_n = np.sum(Flux_error_1d[:, None]**4 * mjd_hist_bin_array , axis=0)
    y_structure_function, x = np.histogram(mjd_diff, bins=bins, weights=mat1)
    y_N_p, x = np.histogram(mjd_diff, bins=bins)
    y_error_sf, x = np.histogram(mjd_diff, bins=bins, weights= mat6+mat7 )
    final_y_error = np.sqrt((error_term1_n + y_error_sf)/((np.median(Flux_1d)**2)*(y_N_p**2)))
    y_structure_function /= (y_N_p * np.median(Flux_1d)**2)

    #final_y_error[y_N_p == 1] = np.nan
    #y_structure_function[y_N_p == 1] = np.nan
    #y_N_p[y_N_p == 1] = 0
    #print('np.median(Flux_1d) : ', np.median(Flux_1d))
    #print('y_error_sf : ', y_error_sf)
    #print('error_term1_n : ',error_term1_n)
    print('final_y_error : ',final_y_error)
    print('y_N_p : ',y_N_p)
    print('y_structure_function : ',  y_structure_function)
    print('x : ', x)
    #return bins, y_structure_function, final_y_error


    #plt.plot((x[1:] + x[:-1])/2, y/y_, 'o')
    #plt.errorbar((x[1:] + x[:-1])/2, y_structure_function/(y_N_p * Flux_1d.mean()**2), yerr = final_y_error,fmt='.')

    #plt.hist(mjd_diff, bins=bins); plt.plot(bins, 10*np.ones_like(bins), 'o')
    
    if fit:
        #log_delay_, log_struct_func_, log_struct_func_error = np.log10((x[1:] + x[:-1])/2), np.log10(y_structure_function), np.log10(1 + final_y_error/y_structure_function)
        #log_delay_, log_struct_func_, log_struct_func_error = np.sqrt(np.log10(x[1:]) * np.log10(x[:-1])), np.log10(y_structure_function), np.log10(1 + final_y_error/y_structure_function)
        delay_ = (x[1:] + x[:-1])/2
        #p_20 = [1, np.nanmean(delay_), np.nanmin(y_structure_function), np.nanmax(y_structure_function) - np.nanmin(y_structure_function)]
        #p_20 = [1, np.nanmean(log_delay_), np.nanmin(y_structure_function), np.nanmax(y_structure_function) - np.nanmin(y_structure_function)]
        #p_20 = [1, np.nanmean(delay_), np.nanmin(y_structure_function), (np.nanmax(y_structure_function) - np.nanmin(y_structure_function))/(np.nanmax(delay_) - np.nanmin(delay_))]
        #a0_p0, x0_p0, k0_p0 = 
        p_20 = [1, delay_[np.argmax(np.diff(y_structure_function[~np.isfinite(y_structure_function)]))], np.nanmin(y_structure_function), (np.nanmax(y_structure_function) - np.nanmin(y_structure_function))/(np.nanmax(delay_) - np.nanmin(delay_))]
        #p_2, _ = curve_fit(fit_structure_b, log_delay_, log_struct_func_, p0 = p_20, sigma = (final_y_error/(np.log(10)*10**log_struct_func_))[~np.isnan(final_y_error)] , bounds = [[0, 0, -np.inf], [np.inf]*3], maxfev = 5000, nan_policy='omit')
        #p_2, _ = curve_fit(fit_structure_b, log_delay_, log_struct_func_, p0 = p_20, sigma = log_struct_func_error[~np.isnan(final_y_error)] , bounds = [[0, 0, -np.inf], [np.inf]*3], maxfev = 5000, nan_policy='omit')
        #p_2, _ = curve_fit(fit_structure_b, log_delay_, log_struct_func_, p0 = p_20, bounds = [[0]*4, [np.inf]*4], maxfev = 5000, nan_policy='omit')
        p_2, _ = curve_fit(fit_structure_b, delay_, y_structure_function, p0 = p_20, sigma = final_y_error[~np.isnan(final_y_error)] ,bounds = [[0, 0, 0, 0], [100, 1e4, np.inf, np.inf]], maxfev = 5000, nan_policy='omit')
        #p_2, _ = curve_fit(fit_structure_b, delay_, y_structure_function, p0 = p_20 ,bounds = [[0, 0, 0, 0], [100, 1e4, np.inf, np.inf]], maxfev = 5000, nan_policy='omit')
        #print('Parameter : x0 (in days) :', p_2[1], r'$\alpha$ : ', p_2[0], 'c : ', p_2[2])
        print(r'Parameter : $\alpha$ :' + "{0} +- {1} ".format(round(p_2[0], 3), round(np.diag(_)[0],3)) + ' ; x0 : ' + "{0} +- {1} ".format(round(p_2[1], 3), round(np.diag(_)[1],3)) + ' ; c : ' + "{0} +- {1} ".format(round(p_2[2], 3), round(np.diag(_)[2],3)) + ' ; k : ' + "{0} +- {1} ".format(round(p_2[0], 3), round(np.diag(_)[0],3)))
        #plt.plot(10**log_delay_[~np.isnan(final_y_error)], 10**fit_structure_b(log_delay_[~np.isnan(final_y_error)], *p_2))
        plt.plot(delay_[~np.isnan(final_y_error)], fit_structure_b(delay_[~np.isnan(final_y_error)], *p_2))
        ###plt.plot(np.logspace(0,3),fit_structure_b(np.linspace(0,3), *p_2))
        #Tau_r = 10**p_2[1]
        #Tau_r_error = ((10**np.diag(_)[1]) - 1)*Tau_r
        #Chi_2 = np.sum(((10**log_struct_func_[~np.isnan(final_y_error)] - 10**fit_structure_b(log_delay_[~np.isnan(final_y_error)], *p_2))/(log_struct_func_error[~np.isnan(final_y_error)]))**2)/(np.nansum(~np.isnan(final_y_error)) - 4)
        Tau_r = p_2[1]
        Tau_r_error = np.diag(_)[1]
        Chi_2 = np.sum(((y_structure_function[~np.isnan(final_y_error)] - fit_structure_b(delay_[~np.isnan(final_y_error)], *p_2))/(final_y_error[~np.isnan(final_y_error)]))**2)/(np.nansum(~np.isnan(final_y_error)) - 4)
        #plt.plot(delay_[~np.isnan(final_y_error)], fit_structure_a(delay_[~np.isnan(final_y_error)], *p_2))
    #plt.errorbar(10**log_delay_, 10**log_struct_func_, yerr = final_y_error,fmt='.')
    plt.errorbar(delay_,y_structure_function, yerr = final_y_error,fmt='.')
    #plt.vlines(Tau_r, ymin=10**np.nanmin(log_struct_func_), ymax=10**np.nanmax(log_struct_func_))
    plt.vlines(Tau_r, ymin=np.nanmin(y_structure_function), ymax=np.nanmax(y_structure_function))
    #plt.fill([min(max(Tau_r - Tau_r_error, x[0]),Tau_r), min(max(Tau_r - Tau_r_error, x[0]), Tau_r), max(min(Tau_r + Tau_r_error, x[-1]),Tau_r), max(min(Tau_r + Tau_r_error,x[-1]),Tau_r)], [10**np.nanmin(log_struct_func_), 10**np.nanmax(log_struct_func_), 10**np.nanmax(log_struct_func_), 10**np.nanmin(log_struct_func_)], alpha = 0.5)
    plt.fill([min(max(Tau_r - Tau_r_error, x[0]),Tau_r), min(max(Tau_r - Tau_r_error, x[0]), Tau_r), max(min(Tau_r + Tau_r_error, x[-1]),Tau_r), max(min(Tau_r + Tau_r_error,x[-1]),Tau_r)], [np.nanmin(y_structure_function), np.nanmax(y_structure_function), np.nanmax(y_structure_function), np.nanmin(y_structure_function)], alpha = 0.5)
    plt.title('MSP : ' + str(msp_cand) + ' , no of epochs : ' + str(len(mjd_)) + ' , no of delays : ' + str(len(bins) - 1) + '\nParameter : ' + r'$\tau_r$ (in days) : ' + str(round(Tau_r, 2)) + r'$\pm$' + str(round(Tau_r_error, 2)) + r' ; $\alpha$ : ' + str(round(p_2[0], 2))+ r'$\pm$' + str(round(np.diag(_)[0], 2)) + r' ; $m_{w}$ : ' + str(round( np.sqrt(p_2[-1]/2), 3)) + r'$\pm$' + str(round(np.diag(_)[-1], 2)))
    #plt.title('MSP : ' + str(msp_cand) + ' , no of epochs : ' + str(len(mjd_)) + ' , no of delays : ' + str(len(bins) - 1) + '\nParameter : ' + r'$\tau_r$ (in days) : ' + str(round(Tau_r, 2)) + r'$\pm$' + str(round(Tau_r_error, 2)) + r' ; $\alpha$ : ' + str(round(p_2[0], 2))+ r'$\pm$' + str(round(np.diag(_)[0], 2)) + r' ; $\chi^2_{red}$ : ' + str(round(Chi_2, 3)))
    plt.xlabel(r'$log_{10}(\tau)$')
    plt.ylabel(r'$log_{10}D(\tau)$')
    plt.loglog()
    if save_fig:
        file_name = str((msp_cand.split('-')*bool(len(msp_cand.split('-'))- 1) + msp_cand.split('+')*bool(len(msp_cand.split('+'))- 1))[0]) + '_struct_func'
        plt.savefig(file_name, dpi=200)
    plt.show()


def structure_function_total_flux_1_broken_straight_lines(msp_cand, band_n, fit=True, model = 'flat_slope', n_bin = 10):
    '''
    Fits fit_structure_a in log space that is : log D(tau) vs log tau space
    ''' 
    mjd_ = all_info_df[msp_cand]['band'+str(band_n)].mjd.to_numpy(int)
    fav_ind = np.triu_indices(len(mjd_), k=1)
    mjd_diff = abs(mjd_[:, None] - mjd_[None, :])[fav_ind]
    
    Flux_1d = all_info_df[msp_cand]['band'+str(band_n)].Total_Flux.to_numpy(float)
    Flux_error_1d = all_info_df[msp_cand]['band'+str(band_n)].Flux_error.to_numpy(float)
    Flux_error_rms_1d = np.sqrt(Flux_error_1d**2 - (0.1*Flux_1d)**2)
    
    Flux_diff_2d = np.triu(Flux_1d[:, None] - Flux_1d[None, :], k=1)
    Flux_error_diff_2d = np.triu(Flux_error_1d[:, None] - Flux_error_1d[None, :], k=1)
    Flux_error_rms_diff_2d = np.triu(Flux_error_rms_1d[None, :] - Flux_error_rms_1d[None, :], k=1)
    
    mat1 = (Flux_diff_2d**2)[fav_ind]
    mat2 = abs(Flux_error_1d[:, None]**2 + Flux_error_1d[None, :]**2)[fav_ind]
    mat3 = 2*np.triu(Flux_error_diff_2d * Flux_diff_2d, k=1)[fav_ind]
    mat4 = -2*np.triu(Flux_error_1d[:, None] * Flux_error_1d[None, :], k=1)[fav_ind]
    mat6 = 4*((Flux_diff_2d * Flux_error_1d[:,None])**2)[fav_ind]
    mat7 = mat4**2

    if isinstance(n_bin, int):
        #bins=np.geomspace(1, mjd_diff.max(), int(len(mjd_diff)/n_bin))
        bins=np.geomspace(1, mjd_diff.max(), int(n_bin +1) )
        #bins=np.geomspace(mjd_diff.min(), mjd_diff.max(), int(n_bin +1) )
    else:
    	bins=np.geomspace(1, mjd_diff.max(), int(2*len(mjd_)+1))
    	#bins=np.geomspace(mjd_diff.min(), mjd_diff.max(), int(2*len(mjd_)+1))
    #bins = histedges_equalA(mjd_diff, int(2*len(mjd_)+1))
    mjd_hist_bin_array = []		# 2d array with rows , columns = mjd, delay bins of the histogram
    for mjd_val in mjd_:
        bool_arr = mjd_val == mjd_
        mat_2d_mjd_ = bool_arr[:,None] + bool_arr[None, :]
        mjd_hist_, _ = np.histogram(mjd_diff, bins=bins, weights=np.triu(mat_2d_mjd_.astype(int), k=1)[fav_ind])
        mjd_hist_bin_array.append(mjd_hist_**2)
        
    mjd_hist_bin_array = np.array(mjd_hist_bin_array); print(mjd_hist_bin_array.shape)
    error_term1_n = np.sum(Flux_error_1d[:, None]**4 * mjd_hist_bin_array , axis=0)
    y_structure_function, x = np.histogram(mjd_diff, bins=bins, weights=mat1)
    y_N_p, x = np.histogram(mjd_diff, bins=bins)
    y_error_sf, x = np.histogram(mjd_diff, bins=bins, weights= mat6+mat7 )
    final_y_error = np.sqrt((error_term1_n + y_error_sf)/((np.median(Flux_1d)**2)*(y_N_p**2)))
    y_structure_function /= (y_N_p * np.median(Flux_1d)**2)

    #final_y_error[y_N_p == 1] = np.nan
    #y_structure_function[y_N_p == 1] = np.nan
    #y_N_p[y_N_p == 1] = 0
    #print('np.median(Flux_1d) : ', np.median(Flux_1d))
    #print('y_error_sf : ', y_error_sf)
    #print('error_term1_n : ',error_term1_n)
    print('final_y_error : ',final_y_error)
    print('y_N_p : ',y_N_p)
    print('y_structure_function : ',  y_structure_function)
    print('x : ', x)
    #return bins, y_structure_function, final_y_error


    #plt.plot((x[1:] + x[:-1])/2, y/y_, 'o')
    #plt.errorbar((x[1:] + x[:-1])/2, y_structure_function/(y_N_p * Flux_1d.mean()**2), yerr = final_y_error,fmt='.')

    #plt.hist(mjd_diff, bins=bins); plt.plot(bins, 10*np.ones_like(bins), 'o')
    #model = 'flat_slope'
    if fit:
        #log_delay_, log_struct_func_, log_struct_func_error = np.log10((x[1:] + x[:-1])/2), np.log10(y_structure_function), np.log10(1 + final_y_error/y_structure_function)
        #log_delay_, log_struct_func_, log_struct_func_error = np.log10((x[1:] + x[:-1])/2), np.log10(y_structure_function), np.log10(1 + final_y_error/y_structure_function)
        log_delay_, log_struct_func_, log_struct_func_error = np.sqrt(np.log10(x[1:]) * np.log10(x[:-1])), np.log10(y_structure_function), np.log10(1 + final_y_error/y_structure_function)
        #delay_, struct_func_ = (x[1:] + x[:-1])/2, y_structure_function
        if model == 'flat_slope':
            p_20 = [np.nanmean(log_delay_), 0, 1, np.nanmin(log_struct_func_)]
            #p_20 = [1, delay_.mean(), np.nanmin(struct_func_), 1]
            #p_2, _ = curve_fit(fit_func_st_line, log_delay_, log_struct_func_, p0 = p_20, sigma = (final_y_error/(np.log(10)*10**log_struct_func_))[~np.isnan(final_y_error)] , bounds = [[0, 0, -np.inf], [np.inf]*3], maxfev = 5000, nan_policy='omit')
            #p_2, _ = curve_fit(fit_func_st_line, log_delay_, log_struct_func_, p0 = p_20, sigma = log_struct_func_error[~np.isnan(final_y_error)] , bounds = [[0, 0, -np.inf], [np.inf]*3], maxfev = 5000, nan_policy='omit')
            p_2, _ = curve_fit(lambda nu_,*param_0: fit_func_st_line(nu_, 2, *param_0), log_delay_, log_struct_func_, p0 = p_20, bounds = [[-np.inf, 0, 0, -np.inf], [np.inf, 1e-12, 1e3, np.inf]], maxfev = 5000, nan_policy='omit')
            #p_2, _ = curve_fit(fit_func_st_line, delay_, struct_func_, p0 = p_20, bounds = [[0, 0, -np.inf, 0], [np.inf]*4], maxfev = 5000, nan_policy='omit')
            print('Parameter : x0 (in days) :', 10**p_2[0], r'$\alpha$ : ', p_2[0], 'c : ', p_2[2])
        if model == 'slope_flat':
            p_20 = [np.nanmean(log_delay_), 1, 0, np.nanmean(log_struct_func_)]
            #p_20 = [1, delay_.mean(), np.nanmin(struct_func_), 1]
            #p_2, _ = curve_fit(fit_func_st_line, log_delay_, log_struct_func_, p0 = p_20, sigma = (final_y_error/(np.log(10)*10**log_struct_func_))[~np.isnan(final_y_error)] , bounds = [[0, 0, -np.inf], [np.inf]*3], maxfev = 5000, nan_policy='omit')
            #p_2, _ = curve_fit(fit_func_st_line, log_delay_, log_struct_func_, p0 = p_20, sigma = log_struct_func_error[~np.isnan(final_y_error)] , bounds = [[0, 0, -np.inf], [np.inf]*3], maxfev = 5000, nan_policy='omit')
            p_2, _ = curve_fit(lambda nu_,*param_0: fit_func_st_line(nu_, 2, *param_0), log_delay_, log_struct_func_, p0 = p_20, bounds = [[-np.inf, 0, 0, -np.inf], [np.inf, 1e3, 1e-12, np.inf]], maxfev = 5000, nan_policy='omit')
            #p_2, _ = curve_fit(fit_func_st_line, delay_, struct_func_, p0 = p_20, bounds = [[0, 0, -np.inf, 0], [np.inf]*4], maxfev = 5000, nan_policy='omit')
            print('Parameter : x0 (in days) :', 10**p_2[0], r'$\alpha$ : ', p_2[0], 'c : ', p_2[2])
        plt.plot(10**log_delay_[~np.isnan(final_y_error)],10**fit_func_st_line(log_delay_[~np.isnan(final_y_error)],2, *p_2))
        Tau_r = 10**p_2[0]
        #Tau_r_error = (np.diag(_)[1])*np.log(10)*Tau_r
        Tau_r_error = ((10**np.diag(_)[0]) - 1)*Tau_r
        Chi_2 = np.sum(((log_struct_func_[~np.isnan(final_y_error)] - fit_func_st_line(log_delay_[~np.isnan(final_y_error)],2, *p_2))/(log_struct_func_error[~np.isnan(final_y_error)]))**2)/(np.nansum(~np.isnan(final_y_error)) - 3)
        #plt.plot(delay_[~np.isnan(final_y_error)], fit_func_st_line(delay_[~np.isnan(final_y_error)],2, *p_2))
    plt.errorbar((x[1:] + x[:-1])/2, y_structure_function, yerr = final_y_error,fmt='.')
    plt.vlines(Tau_r, ymin=10**np.nanmin(log_struct_func_), ymax=10**np.nanmax(log_struct_func_))
    plt.fill([Tau_r - Tau_r_error, Tau_r - Tau_r_error, Tau_r + Tau_r_error, Tau_r + Tau_r_error], [10**np.nanmin(log_struct_func_), 10**np.nanmax(log_struct_func_), 10**np.nanmax(log_struct_func_), 10**np.nanmin(log_struct_func_)], alpha = 0.5)
    #plt.title('MSP : ' + str(msp_cand) + ' , no of epochs : ' + str(len(mjd_)) + ' , no of delays : ' + str(len(bins) - 1) + '\nParameter : ' + r'$\tau_r$ (in days) : ' + str(round(Tau_r, 2)) + r'$\pm$' + str(round(Tau_r_error, 2)) + r' ; $\alpha$ : ' + str(round(p_2[0], 2))+ r'$\pm$' + str(round(np.diag(_)[0], 2)))
    plt.title('MSP : ' + str(msp_cand) + ' , no of epochs : ' + str(len(mjd_)) + ' , no of delays : ' + str(len(bins) - 1) + '\nParameter : ' + r'$\tau_r$ (in days) : ' + str(round(Tau_r, 2)) + r'$\pm$' + str(round(Tau_r_error, 2)) + r' ; $\alpha$ : ' + str(round(p_2[0], 2))+ r'$\pm$' + str(round(np.diag(_)[0], 2)) + r' ; $\chi^2_{red}$ : ' + str(round(Chi_2, 3)))
    plt.xlabel(r'$log_{10}(\tau)$')
    plt.ylabel(r'$log_{10}D(\tau)$')
    plt.loglog()
    file_name = str((msp.split('-')*bool(len(msp.split('-'))- 1) + msp.split('+')*bool(len(msp.split('+'))- 1))[0]) + '_struct_func'
    #plt.savefig(file_name, dpi=200)
    plt.show()



def structure_function_total_flux_2a(msp_cand, band_n, fit=True, n_bin = 10):
    '''
    Fits fit_structure_a in in linear space that is : D(tau) vs tau space
    ''' 
    mjd_ = all_info_df[msp_cand]['band'+str(band_n)].mjd.to_numpy(int)
    fav_ind = np.triu_indices(len(mjd_), k=1)
    mjd_diff = abs(mjd_[:, None] - mjd_[None, :])[fav_ind]
    
    Flux_1d = all_info_df[msp_cand]['band'+str(band_n)].Total_Flux.to_numpy(float)
    Flux_error_1d = all_info_df[msp_cand]['band'+str(band_n)].Flux_error.to_numpy(float)
    Flux_error_rms_1d = np.sqrt(Flux_error_1d**2 - (0.1*Flux_1d)**2)
    
    Flux_diff_2d = np.triu(Flux_1d[:, None] - Flux_1d[None, :], k=1)
    Flux_error_diff_2d = np.triu(Flux_error_1d[:, None] - Flux_error_1d[None, :], k=1)
    Flux_error_rms_diff_2d = np.triu(Flux_error_rms_1d[None, :] - Flux_error_rms_1d[None, :], k=1)
    
    mat1 = (Flux_diff_2d**2)[fav_ind]
    mat2 = abs(Flux_error_1d[:, None]**2 + Flux_error_1d[None, :]**2)[fav_ind]
    mat3 = 2*np.triu(Flux_error_diff_2d * Flux_diff_2d, k=1)[fav_ind]
    mat4 = -2*np.triu(Flux_error_1d[:, None] * Flux_error_1d[None, :], k=1)[fav_ind]
    mat6 = 4*((Flux_diff_2d * Flux_error_1d[:,None])**2)[fav_ind]
    mat7 = mat4**2

    if isinstance(n_bin, int):
        #bins=np.geomspace(1, mjd_diff.max(), int(len(mjd_diff)/n_bin))
        bins=np.geomspace(1, mjd_diff.max(), int(n_bin +1) )
    else:
    	bins=np.geomspace(1, mjd_diff.max(), int(len(mjd_)+1))
    mjd_hist_bin_array = []		# 2d array with rows , columns = mjd, delay bins of the histogram
    for mjd_val in mjd_:
        bool_arr = mjd_val == mjd_
        mat_2d_mjd_ = bool_arr[:,None] + bool_arr[None, :]
        mjd_hist_, _ = np.histogram(mjd_diff, bins=bins, weights=np.triu(mat_2d_mjd_.astype(int), k=1)[fav_ind])
        mjd_hist_bin_array.append(mjd_hist_**2)
        
    mjd_hist_bin_array = np.array(mjd_hist_bin_array); print(mjd_hist_bin_array.shape)
    error_term1_n = np.sum(Flux_error_1d[:, None]**4 * mjd_hist_bin_array , axis=0)
    y_structure_function, x = np.histogram(mjd_diff, bins=bins, weights=mat1)
    y_N_p, x = np.histogram(mjd_diff, bins=bins)
    y_error_sf, x = np.histogram(mjd_diff, bins=bins, weights= mat6+mat7 )
    final_y_error = np.sqrt((error_term1_n + y_error_sf)/((np.median(Flux_1d)**2)*(y_N_p**2)))
    y_structure_function /= (y_N_p * np.median(Flux_1d)**2)
    #final_y_error[y_N_p == 1] = np.nan
    #y_structure_function[y_N_p == 1] = np.nan
    #y_N_p[y_N_p == 1] = 0
    #final_y_error[y_N_p == 1] = np.nan
    #y_structure_function[y_N_p == 1] = np.nan
    
    #print('np.median(Flux_1d) : ', np.median(Flux_1d))
    #print('y_error_sf : ', y_error_sf)
    #print('error_term1_n : ',error_term1_n)
    print('final_y_error : ',final_y_error)
    print('y_N_p : ',y_N_p)
    print('y_structure_function : ',  y_structure_function)
    #print('x : ', x)
    #return bins, y_structure_function, final_y_error


    #plt.plot((x[1:] + x[:-1])/2, y/y_, 'o')
    #plt.errorbar((x[1:] + x[:-1])/2, y_structure_function/(y_N_p * Flux_1d.mean()**2), yerr = final_y_error,fmt='.')

    #plt.hist(mjd_diff, bins=bins); plt.plot(bins, 10*np.ones_like(bins), 'o')
    
    if fit:
        delay_ = (x[1:] + x[:-1])/2
        p_20 = [1, delay_.mean(), np.nanmin(y_structure_function) ]
        #p_20 = [1, 1, np.nanmin(y_structure_function) ]
        p_2, _ = curve_fit(fit_structure_a, delay_, y_structure_function, p0 = p_20 , sigma = final_y_error[~np.isnan(final_y_error)] , bounds = [[0, 0, 0],[1e3, np.inf, np.inf]], maxfev = 5000, nan_policy='omit')
        #p_2, _ = curve_fit(fit_structure_a, delay_, y_structure_function, p0 = p_20 , bounds = [[0, 0, 0],[1e3, np.inf, np.inf]], maxfev = 5000, nan_policy='omit')
        plt.plot(delay_[~np.isnan(final_y_error)], fit_structure_a(delay_[~np.isnan(final_y_error)], *p_2))
        
        Tau_r = p_2[1]
        #Tau_r_error = (np.diag(_)[1])*np.log(10)*Tau_r
        Tau_r_error = np.diag(_)[1]
        Chi_2 = np.sum(((y_structure_function[~np.isnan(final_y_error)] - fit_structure_a(delay_[~np.isnan(final_y_error)], *p_2))/(final_y_error[~np.isnan(final_y_error)]))**2)/(np.nansum(~np.isnan(final_y_error)) - 3)
        #plt.plot(delay_[~np.isnan(final_y_error)], fit_structure_a(delay_[~np.isnan(final_y_error)], *p_2))
    plt.errorbar((x[1:] + x[:-1])/2, y_structure_function, yerr = final_y_error,fmt='.')
    plt.vlines(Tau_r, ymin=np.nanmin(y_structure_function), ymax=np.nanmax(y_structure_function))
    plt.fill([Tau_r - Tau_r_error, Tau_r - Tau_r_error, Tau_r + Tau_r_error, Tau_r + Tau_r_error], [np.nanmin(y_structure_function), np.nanmax(y_structure_function), np.nanmax(y_structure_function), np.nanmin(y_structure_function)], alpha = 0.5)
    #plt.title('MSP : ' + str(msp_cand) + ' , no of epochs : ' + str(len(mjd_)) + ' , no of delays : ' + str(len(bins) - 1) + '\nParameter : ' + r'$\tau_r$ (in days) : ' + str(round(Tau_r, 2)) + r'$\pm$' + str(round(Tau_r_error, 2)) + r' ; $\alpha$ : ' + str(round(p_2[0], 2))+ r'$\pm$' + str(round(np.diag(_)[0], 2)))
    plt.title('MSP : ' + str(msp_cand) + ' , no of epochs : ' + str(len(mjd_)) + ' , no of delays : ' + str(len(bins) - 1) + '\nParameter : ' + r'$\tau_r$ (in days) : ' + str(round(Tau_r, 2)) + r'$\pm$' + str(round(Tau_r_error, 2)) + r' ; $\alpha$ : ' + str(round(p_2[0], 2))+ r'$\pm$' + str(round(np.diag(_)[0], 2)) + r' ; $\chi^2_{red}$ : ' + str(round(Chi_2, 3)))
    plt.xlabel(r'$log_{10}(\tau)$')
    plt.ylabel(r'$log_{10}D(\tau)$')
    plt.loglog()
    plt.show()



def structure_function_total_flux_2c(msp_cand, band_n, fit=True, n_bin = 10):
    '''
    Fits fit_structure_d in in linear space that is : D(tau) vs tau space
    ''' 
    mjd_ = all_info_df[msp_cand]['band'+str(band_n)].mjd.to_numpy(int)
    fav_ind = np.triu_indices(len(mjd_), k=1)
    mjd_diff = abs(mjd_[:, None] - mjd_[None, :])[fav_ind]
    
    Flux_1d = all_info_df[msp_cand]['band'+str(band_n)].Total_Flux.to_numpy(float)
    Flux_error_1d = all_info_df[msp_cand]['band'+str(band_n)].Flux_error.to_numpy(float)
    Flux_error_rms_1d = np.sqrt(Flux_error_1d**2 - (0.1*Flux_1d)**2)
    
    Flux_diff_2d = np.triu(Flux_1d[:, None] - Flux_1d[None, :], k=1)
    Flux_error_diff_2d = np.triu(Flux_error_1d[:, None] - Flux_error_1d[None, :], k=1)
    Flux_error_rms_diff_2d = np.triu(Flux_error_rms_1d[None, :] - Flux_error_rms_1d[None, :], k=1)
    
    mat1 = (Flux_diff_2d**2)[fav_ind]
    mat2 = abs(Flux_error_1d[:, None]**2 + Flux_error_1d[None, :]**2)[fav_ind]
    mat3 = 2*np.triu(Flux_error_diff_2d * Flux_diff_2d, k=1)[fav_ind]
    mat4 = -2*np.triu(Flux_error_1d[:, None] * Flux_error_1d[None, :], k=1)[fav_ind]
    mat6 = 4*((Flux_diff_2d * Flux_error_1d[:,None])**2)[fav_ind]
    mat7 = mat4**2

    if isinstance(n_bin, int):
        #bins=np.geomspace(1, mjd_diff.max(), int(len(mjd_diff)/n_bin))
        bins=np.geomspace(1, mjd_diff.max(), int(n_bin +1) )
    else:
    	bins=np.geomspace(1, mjd_diff.max(), int(2*len(mjd_)+1))
    mjd_hist_bin_array = []		# 2d array with rows , columns = mjd, delay bins of the histogram
    for mjd_val in mjd_:
        bool_arr = mjd_val == mjd_
        mat_2d_mjd_ = bool_arr[:,None] + bool_arr[None, :]
        mjd_hist_, _ = np.histogram(mjd_diff, bins=bins, weights=np.triu(mat_2d_mjd_.astype(int), k=1)[fav_ind])
        mjd_hist_bin_array.append(mjd_hist_**2)
        
    mjd_hist_bin_array = np.array(mjd_hist_bin_array); print(mjd_hist_bin_array.shape)
    error_term1_n = np.sum(Flux_error_1d[:, None]**4 * mjd_hist_bin_array , axis=0)
    y_structure_function, x = np.histogram(mjd_diff, bins=bins, weights=mat1)
    y_N_p, x = np.histogram(mjd_diff, bins=bins)
    y_error_sf, x = np.histogram(mjd_diff, bins=bins, weights= mat6+mat7 )
    final_y_error = np.sqrt((error_term1_n + y_error_sf)/((np.median(Flux_1d)**2)*(y_N_p**2)))
    y_structure_function /= (y_N_p * np.median(Flux_1d)**2)
    #final_y_error[y_N_p == 1] = np.nan
    #y_structure_function[y_N_p == 1] = np.nan
    #y_N_p[y_N_p == 1] = 0
    #final_y_error[y_N_p == 1] = np.nan
    #y_structure_function[y_N_p == 1] = np.nan
    
    #print('np.median(Flux_1d) : ', np.median(Flux_1d))
    #print('y_error_sf : ', y_error_sf)
    #print('error_term1_n : ',error_term1_n)
    print('final_y_error : ',final_y_error)
    print('y_N_p : ',y_N_p)
    print('y_structure_function : ',  y_structure_function)
    #print('x : ', x)
    #return bins, y_structure_function, final_y_error


    #plt.plot((x[1:] + x[:-1])/2, y/y_, 'o')
    #plt.errorbar((x[1:] + x[:-1])/2, y_structure_function/(y_N_p * Flux_1d.mean()**2), yerr = final_y_error,fmt='.')

    #plt.hist(mjd_diff, bins=bins); plt.plot(bins, 10*np.ones_like(bins), 'o')
    
    if fit:
        delay_ = (x[1:] + x[:-1])/2
        p_20 = [1, delay_.mean(), np.nanmin(y_structure_function), np.nanmax(y_structure_function) - np.nanmin(y_structure_function)]
        #p_20 = [1, 1, np.nanmin(y_structure_function) ]
        #p_2, _ = curve_fit(fit_structure_d, delay_, y_structure_function, p0 = p_20 , sigma = final_y_error[~np.isnan(final_y_error)] , bounds = [[0, 0, 0, 0],[1e3, np.inf, np.inf, np.inf]], maxfev = 5000, nan_policy='omit')
        p_2, _ = curve_fit(fit_structure_d, delay_, y_structure_function, p0 = p_20 , bounds = [[0, 0, 0, 0 ],[1e3, np.inf, np.inf, np.inf]], maxfev = 5000, nan_policy='omit')
        plt.plot(delay_[~np.isnan(final_y_error)], fit_structure_d(delay_[~np.isnan(final_y_error)], *p_2))
        
        Tau_r = p_2[1]
        #Tau_r_error = (np.diag(_)[1])*np.log(10)*Tau_r
        Tau_r_error = np.diag(_)[1]
        Chi_2 = np.sum(((y_structure_function[~np.isnan(final_y_error)] - fit_structure_d(delay_[~np.isnan(final_y_error)], *p_2))/(final_y_error[~np.isnan(final_y_error)]))**2)/(np.nansum(~np.isnan(final_y_error)) - 3)
        #plt.plot(delay_[~np.isnan(final_y_error)], fit_structure_d(delay_[~np.isnan(final_y_error)], *p_2))
    plt.errorbar((x[1:] + x[:-1])/2, y_structure_function, yerr = final_y_error,fmt='.')
    plt.vlines(Tau_r, ymin=np.nanmin(y_structure_function), ymax=np.nanmax(y_structure_function))
    plt.fill([Tau_r - Tau_r_error, Tau_r - Tau_r_error, Tau_r + Tau_r_error, Tau_r + Tau_r_error], [np.nanmin(y_structure_function), np.nanmax(y_structure_function), np.nanmax(y_structure_function), np.nanmin(y_structure_function)], alpha = 0.5)
    #plt.title('MSP : ' + str(msp_cand) + ' , no of epochs : ' + str(len(mjd_)) + ' , no of delays : ' + str(len(bins) - 1) + '\nParameter : ' + r'$\tau_r$ (in days) : ' + str(round(Tau_r, 2)) + r'$\pm$' + str(round(Tau_r_error, 2)) + r' ; $\alpha$ : ' + str(round(p_2[0], 2))+ r'$\pm$' + str(round(np.diag(_)[0], 2)))
    plt.title('MSP : ' + str(msp_cand) + ' , no of epochs : ' + str(len(mjd_)) + ' , no of delays : ' + str(len(bins) - 1) + '\nParameter : ' + r'$\tau_r$ (in days) : ' + str(round(Tau_r, 2)) + r'$\pm$' + str(round(Tau_r_error, 2)) + r' ; $\alpha$ : ' + str(round(p_2[0], 2))+ r'$\pm$' + str(round(np.diag(_)[0], 2)) + r' ; $\chi^2_{red}$ : ' + str(round(Chi_2, 3)))
    plt.xlabel(r'$log_{10}(\tau)$')
    plt.ylabel(r'$log_{10}D(\tau)$')
    plt.loglog()
    plt.show()

def freq_subgrouping(msp_cand, band_n):
    a_ = all_info_df[msp_cand]['band'+str(band_n)].Flux_error_snr_subband_freq
    
    flux_error_snr_freq_1d_arr = np.array([ i for y in a_ for i in np.array(y).T if ~np.isnan(i[-1])]).T
    #######################################################################
    f_bins = np.min([len(i[-1]) for i in a_])
    print(f_bins)
    bins = histedges_equalA(flux_error_snr_freq_1d_arr[-1],bins=f_bins)
    hist_count, freq_bins = np.histogram(flux_error_snr_freq_1d_arr[-1], bins=f_bins)
    hist_flux, freq_bins = np.histogram(flux_error_snr_freq_1d_arr[-1], bins=f_bins, weights=flux_error_snr_freq_1d_arr[0])
    hist_flux /= hist_count
    hist_flux_error, freq_bins = np.histogram(flux_error_snr_freq_1d_arr[-1], bins=f_bins, weights=flux_error_snr_freq_1d_arr[1]**2)
    hist_flux_error = np.sqrt(hist_flux_error)/hist_count
    return flux_error_snr_freq_1d_arr, hist_flux, freq_bins, hist_flux_error,bins

def freq_subgrouping_per_epoch(msp_cand, band_n):
    a_ = all_info_df[msp_cand]['band'+str(band_n)][['Flux_error_snr_subband_freq', 'mjd']]
    
    all_epoch_freq_1d_arr = np.array([ i[-1] for y in a_.Flux_error_snr_subband_freq for i in np.array(y).T if ~np.isnan(i[-1])])
    #######################################################################
    f_bins = np.median([np.sum(~np.isnan(i[-1])) for i in a_.Flux_error_snr_subband_freq]).astype(int)
    f_bins = histedges_equalA(all_epoch_freq_1d_arr, int(f_bins))
    full_flux_error_snr_freq_nd_arr = []
    mjd_arr = []
    weights = 'SNR'
    for epoch_ind in range(len(a_)):
        epoch_i = a_.iloc[epoch_ind].Flux_error_snr_subband_freq
        epoch_i = np.array(epoch_i)[:, ~np.isnan(epoch_i[-1])]
        #print(epoch_i)
        freq_bin_center = (f_bins[1:] + f_bins[:-1])/2
        if weights == 'normal':
            hist_count, freq_bins = np.histogram(epoch_i[-1], bins=f_bins)
            hist_flux_sum, freq_bins = np.histogram(epoch_i[-1], bins=f_bins, weights=epoch_i[0])
            hist_flux_mean = hist_flux_sum/hist_count
            
            hist_flux_error_quad_sum, freq_bins = np.histogram(epoch_i[-1], bins=f_bins, weights=epoch_i[1]**2)
            hist_flux_error_mean = np.sqrt(hist_flux_error_quad_sum)/hist_count

            full_flux_error_snr_freq_nd_arr.append([hist_flux_mean, hist_flux_error_mean])
            mjd_arr.append(a_.iloc[epoch_ind].mjd)
        ######################################		SNR weighted mean flux
        elif weights == 'SNR':
            hist_count_snr_dinominator, freq_bins = np.histogram(epoch_i[-1], bins=f_bins, weights = epoch_i[2]**2)
            hist_snr_weighted_flux_sum, freq_bins = np.histogram(epoch_i[-1], bins=f_bins, weights=epoch_i[0] * epoch_i[2]**2)
            hist_SNR_weighted_flux_mean = hist_snr_weighted_flux_sum/hist_count_snr_dinominator

            hist_SNR_weighted_flux_error_quad_sum, freq_bins = np.histogram(epoch_i[-1], bins=f_bins, weights=(epoch_i[1]**2)*(epoch_i[2]**4))
            hist_SNR_weighted_flux_error_mean = np.sqrt(hist_SNR_weighted_flux_error_quad_sum)/hist_count_snr_dinominator

            full_flux_error_snr_freq_nd_arr.append([hist_SNR_weighted_flux_mean, hist_SNR_weighted_flux_error_mean])
            mjd_arr.append(a_.iloc[epoch_ind].mjd)
    return np.array(full_flux_error_snr_freq_nd_arr).transpose(0,2,1), freq_bin_center, np.array(mjd_arr, dtype=int)	# transpose makes it an array with (i_epoch, j_frequency); freq_bin_center contains frequency information, mjd array




def structure_function_freq_flux_2(msp_cand, band_n, fit = True, n_bin = 10): 
    Flux_and_error_2d_epoch_freq, freq_,  mjd_0 = freq_subgrouping_per_epoch(msp_cand, band_n)
    fig = plt.subplot()
    plt.subplots_adjust(hspace=0,wspace=0)
    for freq_ind in range(len(freq_)):
        print('###################################################################################################')
        Flux_1d = Flux_and_error_2d_epoch_freq[:,freq_ind,0]
        no_nan_ind = ~np.isnan(Flux_1d)
        if np.sum(no_nan_ind) < 2:
            print('No epoch observed at this frequency : ', int(freq_[freq_ind]) , ' MHz')
            continue
        Flux_1d = Flux_1d[no_nan_ind]
        Flux_error_1d = Flux_and_error_2d_epoch_freq[:,freq_ind,1]
        Flux_error_1d = Flux_error_1d[no_nan_ind]
        mjd_ = mjd_0[no_nan_ind]
        
        fav_ind = np.triu_indices(len(mjd_), k=1)
        mjd_diff = abs(mjd_[:, None] - mjd_[None, :])[fav_ind]
        Flux_error_rms_1d = np.sqrt(Flux_error_1d**2 - (0.1*Flux_1d)**2)

        
        Flux_diff_2d = np.triu(Flux_1d[:, None] - Flux_1d[None, :], k=1)
        Flux_error_diff_2d = np.triu(Flux_error_1d[:, None] - Flux_error_1d[None, :], k=1)
        Flux_error_rms_diff_2d = np.triu(Flux_error_rms_1d[None, :] - Flux_error_rms_1d[None, :], k=1)
        #return Flux_diff_2d, Flux_error_diff_2d
        mat1 = (Flux_diff_2d**2)[fav_ind]
        mat2 = abs(Flux_error_1d[:, None]**2 + Flux_error_1d[None, :]**2)[fav_ind]
        mat3 = 2*np.triu(Flux_error_diff_2d * Flux_diff_2d, k=1)[fav_ind]
        mat4 = -2*np.triu(Flux_error_1d[:, None] * Flux_error_1d[None, :], k=1)[fav_ind]
        #mat5 = 
        mat6 = 4*((Flux_diff_2d * Flux_error_1d[:,None])**2)[fav_ind]
        mat7 = mat4**2
        #return mat1, mat2, mat3, mat4, mat6, mat7
        #print(mat1.shape, mat2.shape, mat3.shape, mat4.shape, mat6.shape, mat7.shape)
        #bins=np.geomspace(mjd_diff.min(), mjd_diff.max(), 20)
        #bins=np.geomspace(mjd_diff.min(), mjd_diff.max(), int(len(np.unique(mjd_diff))/15))
        #bins=np.geomspace(1, mjd_diff.max(), int(len(mjd_diff)/10))
        if isinstance(n_bin, int):
            #bins=np.geomspace(1, mjd_diff.max(), int(len(mjd_diff)/n_bin))
            bins=np.geomspace(1, mjd_diff.max(), int(n_bin +1) )
        else:
    	    bins=np.geomspace(1, mjd_diff.max(), int(2*len(mjd_)+1))

        #bins = histedges_equalA(mjd_diff, 15)
        mjd_hist_bin_array = []		# 2d array with rows , columns = mjd, delay bins of the histogram
        for mjd_val in mjd_:
            bool_arr = mjd_val == mjd_
            mat_2d_mjd_ = bool_arr[:,None] + bool_arr[None, :]
            mjd_hist_, _ = np.histogram(mjd_diff, bins=bins, weights=np.triu(mat_2d_mjd_.astype(int), k=1)[fav_ind])
            mjd_hist_bin_array.append(mjd_hist_**2)
            
        mjd_hist_bin_array = np.array(mjd_hist_bin_array); print(mjd_hist_bin_array.shape)
        error_term1_n = np.sum(Flux_error_1d[:, None]**4 * mjd_hist_bin_array , axis=0)
        #plt.imshow(mjd_hist_bin_array); plt.show()
        #y_structure_function, x = np.histogram(mjd_diff, bins=bins, weights=(mat1 + mat2 + mat3 + mat4))
        y_structure_function, x = np.histogram(mjd_diff, bins=bins, weights=mat1)
        y_N_p, x_ = np.histogram(mjd_diff, bins=bins)
        y_error_sf, x = np.histogram(mjd_diff, bins=bins, weights= mat6+mat7 )
        final_y_error = np.sqrt((error_term1_n + y_error_sf)/(np.median(Flux_1d)**4*y_N_p**2))
        y_structure_function /= (y_N_p * np.median(Flux_1d)**2)
        '''
        print('np.median(Flux_1d) : ', np.median(Flux_1d))
        print('y_error_sf : ', y_error_sf)
        print('error_term1_n : ',error_term1_n)
        print('final_y_error : ',final_y_error)
        print('y_N_p : ',y_N_p)
        print('y_structure_function/(y_N_p * np.median(Flux_1d)**2) : ', y_structure_function/(y_N_p * np.median(Flux_1d)**2) )
        print('x : ', x)
        
        
        if fit:
            model = '3part'
            log_delay_, log_struct_func_ = np.log10((x[1:] + x[:-1])/2), np.log10(y_structure_function/(y_N_p * np.median(Flux_1d)**2))
            if model == '2part':
                
                #p_20 = [log_delay_.mean(), (log_struct_func_[~np.isnan(final_y_error)][-1] - log_struct_func_[~np.isnan(final_y_error)][0])/(log_delay_[-1] - log_delay_[0]), 0, log_struct_func_[~np.isnan(log_struct_func_)][-1] ]
                p_20 = [log_delay_.mean(), 1,0, np.nanmin(log_struct_func_)  ]
                #p_2, _ = curve_fit(lambda nu_,*param_0: fit_func_st_line(nu_, 2, *param_0), log_delay_, log_struct_func_, p0 = p_20, sigma = (final_y_error/(np.log(10)*10**log_struct_func_))[~np.isnan(final_y_error)] , bounds = [[log_delay_.min(), -100, 0, -np.inf],[log_delay_.max(), 100, 1e-10, np.inf]], maxfev = 5000, nan_policy='omit')
                p_2, _ = curve_fit(lambda nu_,*param_0: fit_func_st_line(nu_, 2, *param_0), log_delay_, log_struct_func_, p0 = p_20, bounds = [[log_delay_.min(), -100, 0, -np.inf],[log_delay_.max(), 100, 1e-10, np.inf]], maxfev = 5000, nan_policy='omit')
                #plt.errorbar(10**log_delay_, 10**log_struct_func_, yerr=final_y_error, fmt='.')
                plt.plot(10**log_delay_,10**fit_func_st_line(log_delay_, 2, *p_2))
            #########################################################
            if model == '3part':
                p_30 = [(2*log_delay_.min() + log_delay_.max())/3, (log_delay_.min() + 2*log_delay_.max())/3, 0, 1, 0, np.nanmin(log_struct_func_)  ]
                #p_3, _ = curve_fit(lambda nu_,*param_0: fit_func_st_line(nu_, 3, *param_0), log_delay_, log_struct_func_, p0 = p_30, sigma = (final_y_error/(np.log(10)*10**log_struct_func_))[~np.isnan(final_y_error)] , bounds = [[log_delay_.min(), log_delay_.min(), 0, -100, 0, np.nanmin(log_struct_func_)],[log_delay_.max(), log_delay_.max(),1e-10, 100, 1e-10, np.nanmax(log_struct_func_)]], maxfev = 5000, nan_policy='omit')
                p_3, _ = curve_fit(lambda nu_,*param_0: fit_func_st_line(nu_, 3, *param_0), log_delay_, log_struct_func_, p0 = p_30, bounds = [[log_delay_.min(), log_delay_.min(), 0, -np.inf, 0, -np.inf],[log_delay_.max(), log_delay_.max(),1e-10, np.inf, 1e-10, np.inf]], maxfev = 5000, nan_policy='omit')
                #plt.errorbar(10**log_delay_, 10**log_struct_func_, yerr=final_y_error, fmt='.')
                plt.plot(10**log_delay_,10**fit_func_st_line(log_delay_, 3, *p_3))
        plt.title('MSP : ' + str(msp_cand) + ' , no of epochs : ' + str(len(mjd_)) + ' , no of delays : ' + str(len(bins) - 1) + ' , Freq : ' + str(int(freq_[freq_ind])))
        plt.loglog()
        plt.show()
        print('################################################################################################')
        '''
        if freq_ind == 0:
            ax = plt.subplot(int(len(freq_)//2 + (len(freq_)%2 == 1)), 2, freq_ind + 1)
        else:
            plt.subplot(int(len(freq_)//2 + (len(freq_)%2 == 1)), 2, freq_ind + 1, sharex=ax,sharey=ax)
        if fit:
            log_delay_, log_struct_func_ = np.log10((x[1:] + x[:-1])/2), np.log10(y_structure_function)
            p_20 = [1, np.nanmean(log_delay_), np.nanmin(log_struct_func_)]
            try:
                p_2, _ = curve_fit(fit_structure_a, log_delay_, log_struct_func_, p0 = p_20, bounds = [[0, 0, -np.inf], [np.inf]*3], maxfev = 5000, nan_policy='omit')
                Tau_r = 10**p_2[1]
                Tau_r_error = ((10**np.diag(_)[1]) - 1)*Tau_r
                #label = 'No of delays : ' + str(len(bins) - 1) + ' FREQ : ' + str(round(freq_[freq_ind])) + ' , Parameter : ' + r'$\tau_r$ (in days) : ' + str(round(Tau_r, 2)) + r'$\pm$' + str(round(Tau_r_error, 2)) + r' ; $\alpha$ : ' + str(round(p_2[0], 2))+ r'$\pm$' + str(round(np.diag(_)[0], 2))
                label = ' FREQ : ' + str(round(freq_[freq_ind])) + ' , Parameter : ' + r'$\tau_r$ (in days) : ' + str(round(Tau_r, 2)) + r'$\pm$' + str(round(Tau_r_error, 2)) + r' ; $\alpha$ : ' + str(round(p_2[0], 2))+ r'$\pm$' + str(round(np.diag(_)[0], 2))
                print('Parameter : x0 (in days) :', 10**p_2[1], r'$\alpha$ : ', p_2[0], 'c : ', p_2[2])
                plt.plot(10**log_delay_[~np.isnan(final_y_error)],10**fit_structure_a(log_delay_[~np.isnan(final_y_error)], *p_2), label=label)

            except:
                print('FIT NOT POSSIBLE')
                continue
            
            plt.vlines(Tau_r, ymin=10**np.nanmin(log_struct_func_), ymax=10**np.nanmax(log_struct_func_))
            plt.fill([Tau_r - Tau_r_error, Tau_r - Tau_r_error, Tau_r + Tau_r_error, Tau_r + Tau_r_error], [10**np.nanmin(log_struct_func_), 10**np.nanmax(log_struct_func_), 10**np.nanmax(log_struct_func_), 10**np.nanmin(log_struct_func_)], alpha = 0.5)
        plt.errorbar((x[1:] + x[:-1])/2, y_structure_function, yerr = final_y_error,fmt='.')
        plt.legend()
        plt.xlabel(r'$log_{10}(\tau)$')
        plt.ylabel(r'$log_{10}D(\tau)$')
        plt.loglog()
    plt.suptitle('MSP : ' + str(msp_cand) + ' , no of epochs : ' + str(len(mjd_)))
    plt.show()

def time_series_freq(msp_cand, band_n):
    a_= all_info_df[msp_cand]['band'+str(band_n)]
    Flux_total, Flux_total_error, mjd = a_.Total_Flux, a_.Flux_error, a_.mjd
    Flux_total_normalised = Flux_total/np.nanmean(Flux_total)
    Flux_total_error_normalised = Flux_total_error/np.nanmean(Flux_total)
    plt.figure(figsize=(14,7))
    plt.errorbar(mjd, Flux_total_normalised, yerr = Flux_total_error_normalised, fmt ='x', label='mean flux')
    Flux_and_error_2d_epoch_freq, freq_,  mjd_0 = freq_subgrouping_per_epoch(msp_cand, band_n)
    
    for i in range(len(freq_)):
        f_flux_norm, f_flux_err_norm = Flux_and_error_2d_epoch_freq[:, i, 0]/np.nanmean(Flux_and_error_2d_epoch_freq[:, i, 0]), Flux_and_error_2d_epoch_freq[:, i, 1]/np.nanmean(Flux_and_error_2d_epoch_freq[:, i, 0])
        plt.errorbar(mjd_0, f_flux_norm + 5*(i+1), yerr=f_flux_err_norm, fmt='.-')#, label='F = ' + str(np.rint(freq_[i])))
    plt.yticks(5*np.arange(1, len(freq_) + 1), np.round(freq_))
    plt.legend()
    plt.legend(bbox_to_anchor=(0.75, 1.0), ncol=2)
    plt.title('MSP : '+msp_cand)
    plt.xlabel('Subband Frequency (in MHz)')
    plt.ylabel('epoch (in MJD)')
    plt.savefig(str(msp_cand)+'_freq_epoch_time_series', dpi=200)


Table_info = {}
for msp in np.sort(all_psr_names):
    msp_info = []
    for band_i in [3,4]:
        try:
            t = all_info_df[msp]['band'+str(band_i)].Total_Flux.to_numpy(float)
            t_error = all_info_df[msp]['band'+str(band_i)].Flux_error.to_numpy(float)
            print(np.nanmean(t), np.sqrt(np.nansum(t_error**2))/(np.nansum(~np.isnan(t_error))), round(np.nanstd(t)/(np.nanmean(t)),3), round(np.nanmax(t)/(np.nanmedian(t)), 3))
            msp_info.extend([np.nanmean(t), np.sqrt(np.nansum(t_error**2))/(np.sum(~np.isnan(t_error))), round(np.nanstd(t)/(np.nanmean(t)),3), round(np.nanmax(t)/(np.nanmedian(t)), 3)])
            
        except:
            msp_info.extend([np.nan]*4)
    Table_info[msp] = msp_info

error_in_bracket = lambda val_, val_err_, return_val_ : [round(val_, -int(np.floor(np.log10(val_err_)))), int(val_err_/(10**np.floor(np.log10(val_err_))))] if return_val_ else print( round(val_, -int(np.floor(np.log10(val_err_)))) , int(val_err_/(10**np.floor(np.log10(val_err_))))) 


def error_in_bracket(val_, val_err_, return_val_):
    a_,b_ = round(val_, -1-int(np.floor(np.log10(val_err_)))), int(val_err_/(10**np.floor(np.log10(val_err_))))
    if return_val_:
        return a_, b_
    else:
        print(a_, '('+str(b_)+')')


########	Section that calculates robust modulation and normal modulation index
Mod_ind_collection, Robust_Mod_ind_collection1, Robust_Mod_ind_collection2 = [], [], []
for msp in np.sort(all_psr_names):
    try:
        df_ = all_info_df[msp]['band3']
        Flux = df_.Total_Flux
        if len(df_) > 7:
            robust_modulation_index = 0.9183*(np.quantile(Flux, 0.75) - np.quantile(Flux, 0.25))/np.median(Flux)
            modulation_ind = np.std(Flux)/np.mean(Flux)
            dm = df_.dm.mean()
            Mod_ind_collection.append(Flux.std()/Flux.mean())
            Robust_Mod_ind_collection1.append(robust_modulation_index)
            Robust_Mod_ind_collection2.append(np.max(Flux)/np.median(Flux))
            #plt.scatter(modulation_ind, dm)
            print('MSP : ', msp, ' modulation -> ', round(modulation_ind,3), ' robust modulation -> ', robust_modulation_index)
        else:
            print('MSP : ', msp)
    except:
        print('MSP : ', msp, ' modulation ->  -')
        pass


############	Plot and save spectral nature for good fitted epochs


band_ = 3
for msp_ in all_psr_names:
    if not 'band'+str(band_) in all_info_df[msp_].keys():
        continue
    df_ = all_info_df[msp_]['band'+str(band_)]
    epoch_count = 0
    for i in range(len(df_)):
        nu0, nu1 = np.nanmin(df_.Flux_error_snr_subband_freq[i][-1]), np.nanmax(df_.Flux_error_snr_subband_freq[i][-1])
        log_tot_flux_i = np.log10(df_.Total_Flux[i])
        log_nu = np.linspace(np.log10(nu0), np.log10(nu1), 100)
        opt_param = list(np.log10(df_.Break_Frequencies[i])) + df_.Spectral_indices[i] + [0]
        n_pow_law = int(len(opt_param)/2)
        if n_pow_law == 1 or np.any(np.abs(df_.Spectral_indices[i]) < np.abs(df_.Spectral_indices_error[i])) or np.any(np.abs(df_.Spectral_indices_error[i]) <1e-10):
            continue
        epoch_count += 1
        log_flux -= np.nanmax(log_flux)
        #log_flux /= abs(log_flux.min())
        plt.plot(log_nu, log_flux)# + i)# 100*time_line[i])
    plt.title(msp_ + '  Band ' + str(band_) + ' , No of epochs : ' + str(epoch_count) + '/' + str(len(df_)))
    plt.xticks(log_nu[::10], (10**log_nu[::10]).astype(int))
    #plt.scatter(opt_param[: n_pow_law - 1], fit_func_st_line(opt_param[: n_pow_law - 1], n_pow_law, *opt_param) - log_flux[0])
    #plt.savefig(str(msp_)+'_best_fit_spectra_turnover.png')
    plt.show()

#######			Plot of model structure function 
tau_arr = np.logspace(0,2)

label_size = 7

plt.plot(tau_arr, 10**fit_structure_a(np.log10(tau_arr), 3, 1, -1.5))
#plt.annotate('Noise regime', (0,0), xycoords='data',ha='center',fontsize=10, va='bottom', rotation = 0, rotation_mode='anchor', transform_rotates_text=True)
plt.annotate(xy=(0,0), xycoords='axes points', text='Noise regime', fontsize= 2*label_size, va='bottom', rotation = 0, rotation_mode='anchor', transform_rotates_text=True)
plt.annotate(xy=(10, 0.15), xycoords='data', text='Structure \n regime', fontsize= 2*label_size, va='bottom', rotation = 2, rotation_mode='anchor', transform_rotates_text=True)
plt.annotate(xy=(40,1.6), xycoords='data', text='Saturation \n regime', fontsize= 2*label_size, va='bottom', rotation = 0, rotation_mode='anchor', transform_rotates_text=True)
plt.loglog()
plt.ylim(0.0225,3.5)
plt.xlabel(r'Time lag $\tau$ (Days)', fontsize= 2*label_size)
plt.ylabel(r'Structure function $D(\tau)$', fontsize= 2*label_size)
plt.show()





