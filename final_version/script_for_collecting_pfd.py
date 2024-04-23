import psrchive, glob, os
import matplotlib.pyplot as plt
import numpy as np
import re


file_name = glob.glob('*/*pfd')

psr_nam = []
for file_ in file_name:
	psr_nam.append(file_.split('/')[0])
psr_nam = list(np.unique(psr_nam))

count = 0
with open('all_pfd_file_sep.txt','w') as f:
	for psr_nam_ in psr_nam:
		print(psr_nam_)
		f.write('###############################################################################################################################\n')
		f.write('###############################################################################################################################\n')
		f.write('\n')
		file_pa = [];file_ia = [];file_unnamed = []
		for file_ in file_name:
			if file_.split('/')[0] == psr_nam_:
				count += 1
				if 'pa' in [t for s in file_.split('_') for t in s.split('.')]:
					file_pa.append(file_.split('/')[-1])
				elif 'ia' in [t for s in file_.split('_') for t in s.split('.')]:
					file_ia.append(file_.split('/')[-1])
				else:
					file_unnamed.append(file_.split('/')[-1])
		'''
		f.write('PA beam Data :========================================================================'+'\n')
		for i in file_pa:
			f.writelines(i+'\n')
		f.write('IA beam Data :========================================================================'+'\n')
		for i in file_ia:
			f.writelines(i+'\n')
		'''
		f.write('Unnamed beam Data :========================================================================'+'\n')
		for i in file_unnamed:
			f.writelines(i+'\n')
f.close()

psr_date_arr = []; bad_date_format = []
for file_ in file_name:
	for psr_nam_ in psr_nam:
		if file_.split('/')[0] == psr_nam_:
			try:
				if re.findall(r'\d{2}[A-Za-z]+\d{4}', file_):
					psr_date_arr.append([psr_nam_, re.findall(r'\d{2}[A-Za-z]+\d{4}', file_)[0]])
				elif re.findall(r'\d{1}[A-Za-z]+\d{4}', file_):
					psr_date_arr.append([psr_nam_, re.findall(r'\d{1}[A-Za-z]+\d{4}', file_)[0]])
				elif re.findall(r'\d{1}[A-Za-z]+\d{1}k\d{2}', file_):
					psr_date_arr.append([psr_nam_,re.findall(r'\d{1}[A-Za-z]+\d{1}k\d{2}', file_)[0]])
				else:
					psr_date_arr.append([psr_nam_, re.findall(r'\d{2}[A-Za-z]+\d{1}k\d{2}', file_)[0]])
			except:
				bad_date_format.append(file_)

def return_psr_date(text_file_name):
	#psr_nam_ = text_file_name.split('/')[0]
	try:
		if re.findall(r'\d{2}[A-Za-z]+\d{4}', text_file_name):
			return re.findall(r'\d{2}[A-Za-z]+\d{4}', text_file_name)[0]
		elif re.findall(r'\d{1}[A-Za-z]+\d{4}', text_file_name):
			return re.findall(r'\d{1}[A-Za-z]+\d{4}', text_file_name)[0]
		elif re.findall(r'\d{1}[A-Za-z]+\d{1}k\d{2}', text_file_name):
			return re.findall(r'\d{1}[A-Za-z]+\d{1}k\d{2}', text_file_name)[0]
		else:
			return re.findall(r'\d{2}[A-Za-z]+\d{1}k\d{2}', text_file_name)[0]
	except:
		return text_file_name

def return_psr_mode_date(text_file_name):
	#psr_nam_ = text_file_name.split('/')[0]

	if 'pa' in [t for s in text_file_name.split('_') for t in s.split('.')]:
		mode = 'pa'
	elif 'ia' in [t for s in text_file_name.split('_') for t in s.split('.')]:
		mode = 'ia'
	else:
		mode = 'unnamed'


	try:
		if re.findall(r'\d{2}[A-Za-z]+\d{4}', text_file_name):
			return [mode, re.findall(r'\d{2}[A-Za-z]+\d{4}', text_file_name)[0]]
		elif re.findall(r'\d{1}[A-Za-z]+\d{4}', text_file_name):
			return [mode, re.findall(r'\d{1}[A-Za-z]+\d{4}', text_file_name)[0]]
		elif re.findall(r'\d{1}[A-Za-z]+\d{1}k\d{2}', text_file_name):
			return [mode, re.findall(r'\d{1}[A-Za-z]+\d{1}k\d{2}', text_file_name)[0]]
		else:
			return [mode, re.findall(r'\d{2}[A-Za-z]+\d{1}k\d{2}', text_file_name)[0]]
	except:
		return [mode, text_file_name]

with open('all_pfd_file_sep.txt','w') as f:
	all_unnamed_files=[]
	count = 0
	for psr_nam_ in psr_nam:
		print(psr_nam_)
		psr_date_arr_ = []
		file_pa = [];file_ia = [];file_unnamed = []
		for file_ in file_name:
			if file_.split('/')[0] == psr_nam_:
				
				if 'pa' in [t for s in file_.split('_') for t in s.split('.')]:
					file_pa.append(file_.split('/')[-1])
				elif 'ia' in [t for s in file_.split('_') for t in s.split('.')]:
					file_ia.append(file_.split('/')[-1])
				else:
					file_unnamed.append(file_.split('/')[-1])
				######### Till here the pa ia and unnamed were separated out for each psr
		break
		####### From here, unique dates for files for each psrs is being searched
		#
		for fi in file_unnamed:
			arr_ = return_psr_date(fi)
			if not arr_ == fi:
				psr_date_arr_.append(arr_)
		# Unique dates for each psr
		uniq_data, uniq_ind = np.unique(psr_date_arr_, return_index=True, axis =0)
		u_data = np.take(file_unnamed, uniq_ind)
		all_unnamed_files.append(u_data)
		f.write('Unnamed beam Data :========================================================================'+'\n')
		for i in u_data:
			f.writelines(i+'\n')





