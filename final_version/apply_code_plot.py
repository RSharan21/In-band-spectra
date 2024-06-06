import psrchive, glob, os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, fsolve
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
from argparse import ArgumentParser

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

########################################################################################################################################

# arr_2d should be actual data (without baseline subtraction)

def off_region_array_from_actual_data_2(arr_2d):
	
	box_car_len = int(np.round(0.15*arr_2d.shape[-1])) # int(0.15*arr_2d.shape[-1])
	start_b = np.convolve(arr_2d.mean(0),np.ones(box_car_len),mode='valid').argmin() -1
	mask = np.zeros(arr_2d.shape[-1],dtype=bool)
	print(start_b)
	if len(mask[start_b : start_b + box_car_len]):
		mask[start_b : start_b + box_car_len] = True
	else:
		start_b = np.convolve(np.roll(arr_2d.mean(0), box_car_len ),np.ones(box_car_len),mode='valid').argmin() -1	
		mask = np.zeros(arr_2d.shape[-1],dtype=bool)
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
###############################################################################################################################################
# Model is a either 1 power law or 2 smoothened power law (Eqn 2 from Sieber 1973; Synchrotron self absorption)
def initial_guess_model2(n_, x_, y_):
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
		range_ = np.logical_and(breaking_points[i]<=x_, x_<breaking_points[i+1])
		param_init, _ = curve_fit(lambda x_vals_, *params_0_: fit_func_power_law(x_vals_, 1, *params_0_), x_[range_],y_[range_], p0=[-1,1], bounds=[[-np.inf,0], [np.inf, np.inf]], maxfev = 5000)
		if i==0:
			c0 = param_init[-1]
		init_guess.append(param_init[0])
	init_guess.append(c0)
	#############################################################################
	lower_bounds = [x_.min()]*(n_ - 1) + [-np.inf]*(n_ + 1)
	if n_ == 1:
		return init_guess, [[0]*(n_ - 1) + [-np.inf]*n_ + [0], [np.inf]*2*n_ ]
	else:
		init_guess[1] = 2.5
		return init_guess, [[0]*(n_ - 1) + [2.499999999] + [-np.inf]*(n_ - 1) + [0], [np.inf]*(n_ - 1) + [2.5000001] + [np.inf]*n_]

def model2(x_, n_, *args):
	if n_ == 1:
		return args[-1]*(x_**args[0])
	if n_ == 2:
		return args[-1]*((x_/args[0])**args[1])*(1 - np.exp(-(x_/args[0])**(args[2] - args[1])))

def derivative_model2(x_, *args):
	return 1 + np.exp(-(x_/args[0])**(args[2] - args[1]))*(((args[2]/args[1]) - 1)* ((x_/args[0])**(args[2] - args[1])) - 1)

def error_prop(args, args_error):
	para_space_values = []
	for i in np.linspace(args[0] - args_error[0]/2, args[0] + args_error[0]/2, 100):
		for j in np.linspace(args[1] - args_error[1]/2, args[1] + args_error[1]/2, 100):
			for k in np.linspace(args[2] - args_error[2]/2, args[2] + args_error[2]/2, 100):
				for l in np.linspace(args[3] - args_error[3]/2, args[3] + args_error[3]/2, 100):
					para_space_values.append(fsolve(derivative_model2, args[0], args=tuple([i,j,k,l])))
	plt.hist(para_space_values)
	plt.show()

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
				
			else:
				data_, freq_, mjds_, dm_= pfd_data_b(file_)
				data_ = data_[:,0]
				global_off_bins_bool_arr = off_region_array_from_actual_data(data_)
				#####			Bandshape correction from the data
				
				data_ = np.nan_to_num((data_ - np.nanmean(data_[:,:,global_off_bins_bool_arr], axis=-1,keepdims=True))/np.nanstd(data_[:,:,global_off_bins_bool_arr], axis=-1,keepdims=True), nan=0, neginf=0,posinf=0)
				d_mean_ = data_.mean(-1)
				unflag_freq = np.sum(data_, axis = (0,2)) != 0		# for removing fully flagged channels from the data_
				unflag_freq = np.logical_and(d_mean_.std(0) > (np.median(d_mean_.std(0))*0.1) , d_mean_.std(0) < (np.median(d_mean_.std(0)) +15*robust_std(d_mean_.std(0)) ) )
				unflag_time = np.sum(data_, axis = (1,2)) != 0
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
	profile = data_.mean(axis=(0,1))
	snr3_bool = np.nan_to_num(snr, nan=0, posinf=0.0, neginf=0.0) > sigma
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
				params_0, bounds_0 = initial_guess_n_bounds_func( n_pow_law_ind, log_freq, log_flux)
				param_, _ = curve_fit(lambda nu_,*param_0: fit_func_st_line(nu_, n_pow_law_ind, *param_0), log_freq, log_flux, p0 = params_0, sigma = noise_/flux , bounds = bounds_0, maxfev = 5000)
				Chi2 = ( ((log_flux -  fit_func_st_line(log_freq, n_pow_law_ind, *param_))/(noise_/(np.log(10)*flux)))**2 ).sum()
				aicc_ = Chi2 + 2*len(param_)*len(log_freq)/(len(log_freq) - len(param_) - 1)
				AICC.append(aicc_)
				param_arr.append(param_)
				param_err_arr.append(_)
				del params_0, bounds_0, Chi2, param_, _, aicc_
				n_pow_law = int(np.argmin(AICC) + 1)
				opt_param = param_arr[np.argmin(AICC)]
				
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
				for i in range(n_pow_law): print("spectral index {0} +- {1} ".format(around(opt_param[n_pow_law - 1: -1][i],3), around(opt_param_err[n_pow_law - 1: -1][i],3)))
				subp4.set_title('Spectal Index:' + "{0} +- {1} ".format(np.around(opt_param[n_pow_law - 1: -1],3), np.around(opt_param_err[n_pow_law - 1: -1],3)))
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

	profile = data_.mean(axis=(0,1))
	snr3_bool = np.nan_to_num(snr, nan=0,posinf=0.0, neginf=0.0) > sigma_
	flux = np.where(snr3_bool * width_mask, snr * noise, np.nan)#(snr * noise)[snr3_bool * width_mask]
	noise_ = np.where(snr3_bool * width_mask, noise, np.nan)#noise[snr3_bool * width_mask]
	error = np.sqrt(noise_**2 + (0.1 * flux)**2)
	log_freq, log_flux = np.where(snr3_bool * width_mask, np.log10(fcrunch10_freq), np.nan), np.log10(flux)
	error_y = error  # noise_
	
	if allow_fit:
		AICC = []
		param_arr = []
		param_err_arr = []
		try:
			for n_pow_law_ind in range(1,int(len(log_freq[~np.isnan(log_freq)])/2)):
				params_0, bounds_0 = initial_guess_n_bounds_func( n_pow_law_ind, log_freq[~np.isnan(log_freq)], log_flux[~np.isnan(log_flux)])
				param_, _, infodict_, mesg_, ier_ = curve_fit(lambda nu_,*param_0: fit_func_st_line(nu_, n_pow_law_ind, *param_0), log_freq, log_flux, p0 = params_0, sigma = noise_[~np.isnan(noise_)]/(np.log(10)*flux[~np.isnan(log_flux)]) , bounds = bounds_0, maxfev = 5000,absolute_sigma=True,full_output=True, nan_policy='omit')
				Chi2 = ( ((log_flux[~np.isnan(log_flux)] -  fit_func_st_line(log_freq[~np.isnan(log_freq)], n_pow_law_ind, *param_))/(noise_[~np.isnan(noise_)]/(np.log(10)*flux[~np.isnan(log_flux)])))**2 ).sum()

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


# This version mainly does (data - data_mean[off pulse])/data_std(off_pulse)[:,:,None] -> create the on-off mask. if mask_with_prim_comp_flags_v7 is used then the definition of standard deviation for a chunk in frequecy changes (1/np.sqrt(sum_along_freq(1/variance_along_time)))



def ugmrt_in_band_flux_spectra_v2_fit_variant(Tsky, n_chan = 10, sigma_ = 10, thres = 3, beam_ = 'PA', primary_comp = True, show_plots = True, allow_fit = False, save_plot = False, n_ant = 23, **file_data):
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
	profile = data_.mean(axis=(0,1))
	snr3_bool = np.nan_to_num(snr, nan=0,posinf=0.0, neginf=0.0) > sigma_
	flux = (snr * noise)[snr3_bool * width_mask]
	noise_ = noise[snr3_bool * width_mask]
	error = np.sqrt(noise_**2 + (0.1 * flux)**2)
	freq_arr_, flux_arr_ = fcrunch10_freq[snr3_bool * width_mask], flux
	error_y = error  # noise_
	
	if allow_fit:
		AICC = []
		param_arr = []
		param_err_arr = []
		try:
			for n_pow_law_ind in range(1,int(len(freq_arr_)/2)):

				params_0, bounds_0 = initial_guess_n_bounds_pow_law_func( n_pow_law_ind, freq_arr_, flux_arr_)
				param_, _ = curve_fit(lambda nu_,*param_0: fit_func_power_law(nu_, n_pow_law_ind, *param_0), freq_arr_, flux_arr_, p0 = params_0, sigma = noise_ , bounds = bounds_0, maxfev = 5000,absolute_sigma=True, nan_policy='omit')
				Chi2 = ( ((flux_arr_ -  fit_func_power_law(freq_arr_, n_pow_law_ind, *param_))/noise_)**2 ).sum()
				aicc_ = Chi2 + 2*len(param_)*len(freq_arr_)/(len(freq_arr_) - len(param_) - 1)
				AICC.append(aicc_)
				param_arr.append(param_)
				param_err_arr.append(_)
				del params_0, bounds_0, Chi2, param_, _, aicc_
				n_pow_law = int(np.argmin(AICC) + 1)
				opt_param = param_arr[np.argmin(AICC)]
				
		except:
			print('FITTING PROCESS FAILED')
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
				subp4.plot(np.linspace(freq_arr_[0], freq_arr_[-1], 1000), fit_func_power_law(np.linspace(freq_arr_[0], freq_arr_[-1], 1000), n_pow_law, *opt_param) )
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

def plot_in_band_spectra(file_, Tsky, n_chan = 10, sigma_ = 5, thres = 3, beam_ = 'PA', primary_comp = True, show_plots = True, allow_fit = False, save_plot = False, n_ant = 23, return_values = True):
	global Total_flux, SNR, on_weight_arr
	SNR, Total_flux, noise, error_nu, freq, on_weight_arr, dm_ = ugmrt_in_band_flux_spectra_v2(Tsky, 'full', sigma_, thres, beam_, primary_comp, False, False, False, n_ant, FILE = file_)
	Total_flux = Total_flux*1e3
	spec, snr_nu, flux_nu, noise_nu, error_nu, nu, weight, dm_ = ugmrt_in_band_flux_spectra_v2_fit_variant(Tsky, n_chan, sigma_, thres, beam_, primary_comp, show_plots, allow_fit, save_plot, n_ant, FILE = file_)
	del Total_flux, SNR, on_weight_arr
	if return_values:
		return spec, snr_nu, flux_nu, noise_nu, error_nu, nu, weight, dm_



####################################################################################################################
###########			Function needed for searching for eclipses

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





if __name__ == "__main__":
	
	parser = ArgumentParser()
	parser.add_argument("-Tsky")
	parser.add_argument("-n_chan")
	parser.add_argument("-sigma_")
	parser.add_argument("-thres")
	parser.add_argument("-beam_")
	parser.add_argument("-primary_comp", action="store_const", const=True, default=False)
	parser.add_argument("-show_plots", action="store_const", const=True, default=False)	# default is false when -show_plots NOT mentioned, And True when mentioned in the command line
	parser.add_argument("-allow_fit", action="store_const", const=True, default=False)	
	parser.add_argument("-save_plot", action="store_const", const=True, default=False)
	parser.add_argument("-n_ant")
	parser.add_argument("-file_data")
	parser.add_argument("-return_values")
	args = parser.parse_args()
	plot_in_band_spectra(str(args.file_data), float(args.Tsky), int(args.n_chan), float(args.sigma_), float(args.thres), str(args.beam_), bool(args.primary_comp), bool(args.show_plots), bool(args.allow_fit), bool(args.save_plot), int(args.n_ant), bool(args.return_values))
	#ugmrt_in_band_flux_spectra_v2_fit_variant(float(args.Tsky), int(args.n_chan), float(args.sigma_), float(args.thres), str(args.beam_), bool(args.primary_comp), bool(args.show_plots), bool(args.allow_fit), bool(args.save_plot), int(args.n_ant), FILE=str(args.file_data))
	
    


