#MATRAS GROUP. Physics Department, UNIVERSITY OF JAEN.
#
#
#CIADCast
#
#First, you have to include a new tracer in WRF model, in my case, in WRF-Solar.
#In Registry/Registry.EM add our new field “tr_ci” as part of “TRACER” array.
#
#state   real    tr_ci      ikjftb  tracer        1         -     irhusdf=(bdy_interp:dt)    "tr_ci"          "Traza Cloud Index" ""
#
#package   tracer_test3  tracer_opt==3       -             tracer:tr_ci
#
#Now you have to recompile WRF.
#Then, in namelist.input file, include the new settings for tracer option
#
#&dynamics
# tracer_opt = 3,
#


######Paths
rundates = ''#text file with periods of three hours to run.
MSG_ci_path = ''#folder with MSG processed cloud index as variable name 'cloud_index_n' in files called cifilename_+date
MSG_cth_path = ''#folder with MSG CTH product
latlons_wrfinput_d01 = ''#netcdf with wrf latitudes and longitudes for cth interpolation.
latlons_CTH = ''#netcdf with satellite latitudes and longitudes for cth interpolation.
wps_path = ''
gfs_path = ''
wrf_path = ''
output_path = ''
######


import os, glob
import numpy   as np
import netCDF4 as nc
from scipy.interpolate import griddata
from datetime import datetime, timedelta
import matplotlib.dates as dt

fec_anterior = datetime(2015, 3, 1, 0, 0)#A previous date

archivo = open(rundates,'r')
fechasClusterGFS = archivo.readlines()
archivo.close()
fechasClusterGFS = np.array([datetime.strptime(fechasClusterGFS[i][:-5],'%Y%m%d_%H') for i in range(len(fechasClusterGFS))])

fics_n = os.listdir(MSG_ci_path)
fics_n.sort()
fics_n = np.array([datetime.strptime(fics_n[i][-15:-3],'%Y%m%d%H%M') for i in range(len(fics_n))])

fics_cth = os.listdir(MSG_cth_path)
fics_cth.sort()
fics_cth_datetime = np.array([datetime.strptime(fics_cth[k][4:-3],'%Y%m%d%H%M') for k in range(len(fics_cth))])

wrf = nc.Dataset(latlons_wrfinput_d01,'r')
lat_wrf = wrf.variables['lat_wrf'][:]
lon_wrf = wrf.variables['lon_wrf'][:]
wrf.close()

dat = nc.Dataset(latlons_CTH,'r')
lat_sat = dat.variables['lat_sat'][:]
lon_sat = dat.variables['lon_sat'][:]
dat.close()

for f in np.arange(len(fechasClusterGFS)):
	if f < 763:#At this date num_metgrid_levels changed.
		NumMetgridLevels = 27
	else:
		NumMetgridLevels = 32
	#Select the dates 
	fechas = fics_n[(fics_n>=fechasClusterGFS[f]) & (fics_n<(fechasClusterGFS[f]+timedelta(hours=3)))]
	if len(fechas) != 0:
		for fec in fechas:
			#If it is a new day, start from the beginning.
			if (fec - timedelta(minutes=fec.minute,hours=(fec.hour%6)+6)) != (fec_anterior - timedelta(minutes=fec_anterior.minute,hours=(fec_anterior.hour%6)+6)):
				#Run WPS with 6 hours of spin-up
				os.chdir(wps_path)
				os.system('rm -f FILE\:201*')
				os.system('rm -f GRIBFILE.AA*')
				os.system('rm -f met_em.d0*')
				fecIni = fec - timedelta(minutes=fec.minute,hours=(fec.hour%6)+6)
				fecFin = fecIni + timedelta(hours=18)
				os.system("echo WPS "+datetime.strftime(fecIni,'%Y%m%d_%H%M'))
				os.system("sed 's/year1/"+str(fecIni.year)+"/g' namelist.wps.template > namelist.wps")
				os.system("sed -i 's/mes1/"+str(fecIni.month).zfill(2)+"/g' namelist.wps")
				os.system("sed -i 's/dia1/"+str(fecIni.day).zfill(2)+"/g' namelist.wps")
				os.system("sed -i 's/hora1/"+str(fecIni.hour).zfill(2)+"/g' namelist.wps")
				os.system("sed -i 's/year2/"+str(fecFin.year)+"/g' namelist.wps")
				os.system("sed -i 's/mes2/"+str(fecFin.month).zfill(2)+"/g' namelist.wps")
				os.system("sed -i 's/dia2/"+str(fecFin.day).zfill(2)+"/g' namelist.wps")
				os.system("sed -i 's/hora2/"+str(fecFin.hour).zfill(2)+"/g' namelist.wps")
				os.system("echo now "+datetime.strftime(datetime.now(),'%Y%m%d_%H%M%S_%f'))
				os.system("./link_grib.csh "+gfs_path+"/gfs_4_"+datetime.strftime(fecIni,'%Y%m%d_%H')+"00_0*")
				os.system("echo now "+datetime.strftime(datetime.now(),'%Y%m%d_%H%M%S_%f'))
				os.system("./ungrib.exe >& ungrib.log")
				os.system("tail -1 ungrib.log")
				os.system("./metgrid.exe >& metgrid.log")
				os.system("tail -1 metgrid.log")
				#Run WRF up to the first satellite date.
				os.chdir(wrf_path)
				os.system('rm -f met_em.d0*')
				os.system('ln -sf '+wps_path+'/met_em.d0* .')
				os.system('rm -f rsl.*')
				os.system('rm -f wrfinput_d0*')
				os.system('rm -f wrfout_d0*')
				os.system('rm -f wrfrst_d0*')
				os.system("sed 's/runHinserta/"+str(fec.hour-fecIni.hour)+"/g' namelist.input.template > namelist.input")
				os.system("sed -i 's/runMinserta/"+str(fec.minute+11)+"/g' namelist.input")
				os.system("sed -i 's/year1/"+str(fecIni.year)+"/g' namelist.input")
				os.system("sed -i 's/mes1/"+str(fecIni.month).zfill(2)+"/g' namelist.input")
				os.system("sed -i 's/dia1/"+str(fecIni.day).zfill(2)+"/g' namelist.input")
				os.system("sed -i 's/hora1/"+str(fecIni.hour).zfill(2)+"/g' namelist.input")
				os.system("sed -i 's/minuto1/"+str(fecIni.minute).zfill(2)+"/g' namelist.input")
				os.system("sed -i 's/year2/"+str(fecFin.year)+"/g' namelist.input")
				os.system("sed -i 's/mes2/"+str(fecFin.month).zfill(2)+"/g' namelist.input")
				os.system("sed -i 's/dia2/"+str(fecFin.day).zfill(2)+"/g' namelist.input")
				os.system("sed -i 's/hora2/"+str(fecFin.hour).zfill(2)+"/g' namelist.input")
				os.system("sed -i 's/HistoryInterval/10000/g' namelist.input")
				os.system("sed -i 's/RestartToF/.false./g' namelist.input")
				os.system("sed -i 's/RestartInterval/"+str((fec.hour-fecIni.hour)*60+fec.minute+11)+"/g' namelist.input")
				os.system("sed -i 's/NumMetgridLevels/"+str(NumMetgridLevels)+"/g' namelist.input")
				os.system("sed -i 's/radtime/30/g' namelist.input")
				os.system("mpirun -n 48 ./real.exe -parallel")
				os.system("tail rsl.out.0000")
				os.system("mpirun -n 48 ./wrf.exe -parallel")
				os.system("tail rsl.out.0000")
				
				try:	
					os.system("echo Inserto N "+datetime.strftime(fec + timedelta(minutes=11),'%Y%m%d_%H%M'))
				
					dat = nc.Dataset(MSG_ci_path+'/'+cifilename_'+datetime.strftime(fec,'%Y%m%d%H%M')+'.nc','r')
					n = dat.variables['cloud_index_n'][:]
					n[n==1e+30] = 0
					lat = dat.variables['latitude'][:]
					lon = dat.variables['longitude'][:]
					dat.close()

					#Interpolation of CTH to WRF grid
					dat = nc.Dataset(MSG_cth_path+'/'+fics_cth[np.where(abs(fics_cth_datetime-fec)==(abs(fics_cth_datetime-fec)).min())[0][0]],'r')
					cth_data = dat.variables['cth_data'][:]
					cth_mask = dat.variables['cth_mask'][:]
					dat.close()

					cth_data_wrf = griddata(np.c_[lon_sat.flatten(),lat_sat.flatten()],cth_data.flatten(),(lon_wrf,lat_wrf),method='nearest')
					cth_mask_wrf = griddata(np.c_[lon_sat.flatten(),lat_sat.flatten()],cth_mask.flatten(),(lon_wrf,lat_wrf),method='nearest')

					for core in range(48):#WRF restart domain in splitted into as many pieces as parallel cores
						wrfrst = nc.Dataset('wrfrst_d01_'+datetime.strftime(fec+timedelta(minutes=11),'%Y-%m-%d_%H:%M')+':00_00'+str(core).zfill(2),'a')
						lat_rst = wrfrst.variables['XLAT'][0]
						lon_rst = wrfrst.variables['XLONG'][0]
						ZNW = wrfrst.variables['ZNW'][0]
						deta=-np.diff(ZNW)
						MU=wrfrst.variables['MUB'][0]+wrfrst.variables['MU_2'][0]
						PHP = wrfrst.variables['PHP'][0]
						Z = np.array(PHP[:]/9.81)
						n_wrf = wrfrst.variables['tr_ci'][0]
						n_wrf[:] = 0.0
						n_rst = griddata(np.c_[lon.flatten(),lat.flatten()],n.flatten(),(lon_rst,lat_rst),method='nearest')
						cth_wrf_sinnan = griddata(np.c_[lon_wrf[cth_mask_wrf==0],lat_wrf[cth_mask_wrf==0]],cth_data_wrf[cth_mask_wrf==0],(lon_rst,lat_rst),method='nearest')
						for i in range(cth_wrf_sinnan.shape[0]):
							for j in range(cth_wrf_sinnan.shape[1]):
								#For each grid column, pick the vertical level closest to the CTH and insert the cloud index as mass mixing ratio.
								r = abs(Z[:,i,j]-cth_wrf_sinnan[i,j])
								capa_nube = np.where(r==r.min())[0][0]
								n_wrf[capa_nube,i,j] = n_rst[i,j]/(MU[i,j]*deta[capa_nube])
						n_sat=wrfrst.variables['tr_ci']
						n_sat[0]=n_wrf
						wrfrst.close()

					#Continue running WRF with restart, 6 hours more.
					os.system('rm -f wrfout_d0*')		
					os.system("sed 's/runHinserta/6/g' namelist.input.template > namelist.input")
					os.system("sed -i 's/runMinserta/0/g' namelist.input")
					os.system("sed -i 's/year1/"+str(fec.year)+"/g' namelist.input")
					os.system("sed -i 's/mes1/"+str(fec.month).zfill(2)+"/g' namelist.input")
					os.system("sed -i 's/dia1/"+str(fec.day).zfill(2)+"/g' namelist.input")
					os.system("sed -i 's/hora1/"+str(fec.hour).zfill(2)+"/g' namelist.input")
					os.system("sed -i 's/minuto1/"+str(fec.minute+11)+"/g' namelist.input")
					os.system("sed -i 's/year2/"+str(fecFin.year)+"/g' namelist.input")
					os.system("sed -i 's/mes2/"+str(fecFin.month).zfill(2)+"/g' namelist.input")
					os.system("sed -i 's/dia2/"+str(fecFin.day).zfill(2)+"/g' namelist.input")
					os.system("sed -i 's/hora2/"+str(fecFin.hour).zfill(2)+"/g' namelist.input")
					os.system("sed -i 's/HistoryInterval/15/g' namelist.input")
					os.system("sed -i 's/RestartToF/.true./g' namelist.input")
					os.system("sed -i 's/RestartInterval/10000/g' namelist.input")
					os.system("sed -i 's/NumMetgridLevels/"+str(NumMetgridLevels)+"/g' namelist.input")
					os.system("sed -i 's/radtime/5/g' namelist.input")
					os.system("mpirun -n 48 ./wrf.exe -parallel")
					os.system("tail rsl.out.0000")
				
					wrf = nc.Dataset('wrfout_d01_'+datetime.strftime(fec+timedelta(minutes=11),'%Y-%m-%d_%H:%M:%S'),'r')
					SWDOWN = wrf.variables['SWDOWN'][:]
					SWDDNI = wrf.variables['SWDDNI'][:]
					n_wrf = wrf.variables['tr_ci'][:]
					ZNW = wrf.variables['ZNW'][0]
					deta = -np.diff(ZNW)
					MU = wrf.variables['MUB'][:]+wrf.variables['MU'][:]
					SINALPHA = wrf.variables['SINALPHA'][0]
					COSALPHA = wrf.variables['COSALPHA'][0]
					U = wrf.variables['U'][:,:3]
					V = wrf.variables['V'][:,:3]
					Z = (wrf.variables['PH'][:,:4] + wrf.variables['PHB'][:,:4])/9.81
					U10 = wrf.variables['U10'][:]
					V10 = wrf.variables['V10'][:]
					wrf.close()
					#Save the cloud index de-normalized with the dry air mass of the corresponding grid-cell and integrated in the column.
					tr_ci = np.zeros(SWDOWN.shape)
					for i in range(tr_ci.shape[1]):
						for j in range(tr_ci.shape[2]):
							tr_ci[:,i,j] = ((n_wrf[:,:,i,j].T*MU[:,i,j]).T*deta).sum(axis=1)
					U = (U[:,:,:,:-1]+U[:,:,:,1:])/2 #from staggered to unstaggered
					V = (V[:,:,:-1,:]+V[:,:,1:,:])/2
					U_rotada = U*COSALPHA - V*SINALPHA
					V_rotada = V*COSALPHA + U*SINALPHA
					for t in range(len(Z)):
						Z[t] = Z[t]-Z[t,0]
					Z = (Z[:,:-1]+Z[:,1:])/2
					rootgrp = nc.Dataset(output_path+'/sim'+datetime.strftime(fec+timedelta(minutes=11),'%Y%m%d_%H%M')+'.nc', 'w', format='NETCDF4')
					dim0 = rootgrp.createDimension('dim0',SWDOWN.shape[0])
					dim1 = rootgrp.createDimension('dim1',SWDOWN.shape[1])
					dim2 = rootgrp.createDimension('dim2',SWDOWN.shape[2])
					dim3 = rootgrp.createDimension('dim3',3)
					SWDOWNnc = rootgrp.createVariable('SWDOWN','f8',('dim0','dim1','dim2'))
					SWDOWNnc[:] = SWDOWN
					SWDDNInc = rootgrp.createVariable('SWDDNI','f8',('dim0','dim1','dim2'))
					SWDDNInc[:] = SWDDNI
					tr_cinc = rootgrp.createVariable('tr_ci','f8',('dim0','dim1','dim2'))
					tr_cinc[:] = tr_ci
					U_rotadanc = rootgrp.createVariable('U','f8',('dim0','dim3','dim1','dim2'))
					U_rotadanc[:] = U_rotada
					V_rotadanc = rootgrp.createVariable('V','f8',('dim0','dim3','dim1','dim2'))
					V_rotadanc[:] = V_rotada
					Znc = rootgrp.createVariable('Z','f8',('dim0','dim3','dim1','dim2'))
					Znc[:] = Z
					U10nc = rootgrp.createVariable('U10','f8',('dim0','dim1','dim2'))
					U10nc[:] = U10
					V10nc = rootgrp.createVariable('V10','f8',('dim0','dim1','dim2'))
					V10nc[:] = V10
					rootgrp.close()

					#Finished this cycle, this is the previous date.
					fec_anterior = fec
					
				except:
					print('Error en WRFrestart1')
					fec_anterior = fec
					
			else:#If it is not a new day, continue restart to the next date and repeat the process.
				os.system("echo WRFrestart para "+datetime.strftime(fec,'%Y%m%d_%H%M'))
				try:	
					os.chdir(wrf_path)
					os.system('rm -f wrfout_d0*')		
					os.system("sed 's/runHinserta/"+str(int(((fec-fec_anterior).seconds)/3600))+"/g' namelist.input.template > namelist.input")
					os.system("sed -i 's/runMinserta/"+str(int((fec-fec_anterior).seconds/60)-int(((fec-fec_anterior).seconds)/3600)*60)+"/g' namelist.input")
					os.system("sed -i 's/year1/"+str(fec_anterior.year)+"/g' namelist.input")
					os.system("sed -i 's/mes1/"+str(fec_anterior.month).zfill(2)+"/g' namelist.input")
					os.system("sed -i 's/dia1/"+str(fec_anterior.day).zfill(2)+"/g' namelist.input")
					os.system("sed -i 's/hora1/"+str(fec_anterior.hour).zfill(2)+"/g' namelist.input")
					os.system("sed -i 's/minuto1/"+str(fec_anterior.minute+11)+"/g' namelist.input")
					os.system("sed -i 's/year2/"+str(fecFin.year)+"/g' namelist.input")
					os.system("sed -i 's/mes2/"+str(fecFin.month).zfill(2)+"/g' namelist.input")
					os.system("sed -i 's/dia2/"+str(fecFin.day).zfill(2)+"/g' namelist.input")
					os.system("sed -i 's/hora2/"+str(fecFin.hour).zfill(2)+"/g' namelist.input")
					os.system("sed -i 's/HistoryInterval/10000/g' namelist.input")
					os.system("sed -i 's/RestartToF/.true./g' namelist.input")
					os.system("sed -i 's/RestartInterval/"+str(int((fec-fec_anterior).seconds/60))+"/g' namelist.input")
					os.system("sed -i 's/NumMetgridLevels/"+str(NumMetgridLevels)+"/g' namelist.input")
					os.system("sed -i 's/radtime/30/g' namelist.input")
					os.system("mpirun -n 48 ./wrf.exe -parallel")
					os.system("tail rsl.out.0000")
					
					os.system("echo Inserto N "+datetime.strftime(fec + timedelta(minutes=11),'%Y%m%d_%H%M'))
					
					dat = nc.Dataset(MSG_ci_path+'/'+cifilename_'+datetime.strftime(fec,'%Y%m%d%H%M')+'.nc','r')
					n = dat.variables['cloud_index_n'][:]
					n[n==1e+30] = 0
					lat = dat.variables['latitude'][:]
					lon = dat.variables['longitude'][:]
					dat.close()

					dat = nc.Dataset(MSG_cth_path+'/'+fics_cth[np.where(abs(fics_cth_datetime-fec)==(abs(fics_cth_datetime-fec)).min())[0][0]],'r')
					cth_data = dat.variables['cth_data'][:]
					cth_mask = dat.variables['cth_mask'][:]
					dat.close()

					cth_data_wrf = griddata(np.c_[lon_sat.flatten(),lat_sat.flatten()],cth_data.flatten(),(lon_wrf,lat_wrf),method='nearest')
					cth_mask_wrf = griddata(np.c_[lon_sat.flatten(),lat_sat.flatten()],cth_mask.flatten(),(lon_wrf,lat_wrf),method='nearest')

					for core in range(48):
						wrfrst = nc.Dataset('wrfrst_d01_'+datetime.strftime(fec+timedelta(minutes=11),'%Y-%m-%d_%H:%M')+':00_00'+str(core).zfill(2),'a')
						lat_rst = wrfrst.variables['XLAT'][0]
						lon_rst = wrfrst.variables['XLONG'][0]
						ZNW = wrfrst.variables['ZNW'][0]
						deta=-np.diff(ZNW)
						MU=wrfrst.variables['MUB'][0]+wrfrst.variables['MU_2'][0]
						PHP = wrfrst.variables['PHP'][0]
						Z = np.array(PHP[:]/9.81)
						n_wrf = wrfrst.variables['tr_ci'][0]
						n_wrf[:] = 0.0
						n_rst = griddata(np.c_[lon.flatten(),lat.flatten()],n.flatten(),(lon_rst,lat_rst),method='nearest')
						cth_wrf_sinnan = griddata(np.c_[lon_wrf[cth_mask_wrf==0],lat_wrf[cth_mask_wrf==0]],cth_data_wrf[cth_mask_wrf==0],(lon_rst,lat_rst),method='nearest')
						for i in range(cth_wrf_sinnan.shape[0]):
							for j in range(cth_wrf_sinnan.shape[1]):
								r = abs(Z[:,i,j]-cth_wrf_sinnan[i,j])
								capa_nube = np.where(r==r.min())[0][0]
								n_wrf[capa_nube,i,j] = n_rst[i,j]/(MU[i,j]*deta[capa_nube])
						n_sat=wrfrst.variables['tr_ci']
						n_sat[0]=n_wrf
						wrfrst.close()

					os.system('rm -f wrfout_d0*')		
					os.system("sed 's/runHinserta/6/g' namelist.input.template > namelist.input")
					os.system("sed -i 's/runMinserta/0/g' namelist.input")
					os.system("sed -i 's/year1/"+str(fec.year)+"/g' namelist.input")
					os.system("sed -i 's/mes1/"+str(fec.month).zfill(2)+"/g' namelist.input")
					os.system("sed -i 's/dia1/"+str(fec.day).zfill(2)+"/g' namelist.input")
					os.system("sed -i 's/hora1/"+str(fec.hour).zfill(2)+"/g' namelist.input")
					os.system("sed -i 's/minuto1/"+str(fec.minute+11)+"/g' namelist.input")
					os.system("sed -i 's/year2/"+str(fecFin.year)+"/g' namelist.input")
					os.system("sed -i 's/mes2/"+str(fecFin.month).zfill(2)+"/g' namelist.input")
					os.system("sed -i 's/dia2/"+str(fecFin.day).zfill(2)+"/g' namelist.input")
					os.system("sed -i 's/hora2/"+str(fecFin.hour).zfill(2)+"/g' namelist.input")
					os.system("sed -i 's/HistoryInterval/15/g' namelist.input")
					os.system("sed -i 's/RestartToF/.true./g' namelist.input")
					os.system("sed -i 's/RestartInterval/10000/g' namelist.input")
					os.system("sed -i 's/NumMetgridLevels/"+str(NumMetgridLevels)+"/g' namelist.input")
					os.system("sed -i 's/radtime/5/g' namelist.input")
					os.system("mpirun -n 48 ./wrf.exe -parallel")
					os.system("tail rsl.out.0000")
					
					wrf = nc.Dataset('wrfout_d01_'+datetime.strftime(fec+timedelta(minutes=11),'%Y-%m-%d_%H:%M:%S'),'r')
					SWDOWN = wrf.variables['SWDOWN'][:]
					SWDDNI = wrf.variables['SWDDNI'][:]
					n_wrf = wrf.variables['tr_ci'][:]
					ZNW = wrf.variables['ZNW'][0]
					deta = -np.diff(ZNW)
					MU = wrf.variables['MUB'][:]+wrf.variables['MU'][:]
					SINALPHA = wrf.variables['SINALPHA'][0]
					COSALPHA = wrf.variables['COSALPHA'][0]
					U = wrf.variables['U'][:,:3]
					V = wrf.variables['V'][:,:3]
					Z = (wrf.variables['PH'][:,:4] + wrf.variables['PHB'][:,:4])/9.81
					U10 = wrf.variables['U10'][:]
					V10 = wrf.variables['V10'][:]
					wrf.close()
					tr_ci = np.zeros(SWDOWN.shape)
					for i in range(tr_ci.shape[1]):
						for j in range(tr_ci.shape[2]):
							tr_ci[:,i,j] = ((n_wrf[:,:,i,j].T*MU[:,i,j]).T*deta).sum(axis=1)
					U = (U[:,:,:,:-1]+U[:,:,:,1:])/2
					V = (V[:,:,:-1,:]+V[:,:,1:,:])/2
					U_rotada = U*COSALPHA - V*SINALPHA
					V_rotada = V*COSALPHA + U*SINALPHA
					for t in range(len(Z)):
						Z[t] = Z[t]-Z[t,0]
					Z = (Z[:,:-1]+Z[:,1:])/2
					rootgrp = nc.Dataset(output_path+'/sim'+datetime.strftime(fec+timedelta(minutes=11),'%Y%m%d_%H%M')+'.nc', 'w', format='NETCDF4')
					dim0 = rootgrp.createDimension('dim0',SWDOWN.shape[0])
					dim1 = rootgrp.createDimension('dim1',SWDOWN.shape[1])
					dim2 = rootgrp.createDimension('dim2',SWDOWN.shape[2])
					dim3 = rootgrp.createDimension('dim3',3)
					SWDOWNnc = rootgrp.createVariable('SWDOWN','f8',('dim0','dim1','dim2'))
					SWDOWNnc[:] = SWDOWN
					SWDDNInc = rootgrp.createVariable('SWDDNI','f8',('dim0','dim1','dim2'))
					SWDDNInc[:] = SWDDNI
					tr_cinc = rootgrp.createVariable('tr_ci','f8',('dim0','dim1','dim2'))
					tr_cinc[:] = tr_ci
					U_rotadanc = rootgrp.createVariable('U','f8',('dim0','dim3','dim1','dim2'))
					U_rotadanc[:] = U_rotada
					V_rotadanc = rootgrp.createVariable('V','f8',('dim0','dim3','dim1','dim2'))
					V_rotadanc[:] = V_rotada
					Znc = rootgrp.createVariable('Z','f8',('dim0','dim3','dim1','dim2'))
					Znc[:] = Z
					U10nc = rootgrp.createVariable('U10','f8',('dim0','dim1','dim2'))
					U10nc[:] = U10
					V10nc = rootgrp.createVariable('V10','f8',('dim0','dim1','dim2'))
					V10nc[:] = V10
					rootgrp.close()
					
					fec_anterior = fec
					
				except:
					print('Error en WRFrestart2')
					fec_anterior = fec


#Finally, CIADCast GHI is computed transforming cloud index (tr_ci) into clear sky index value (kc) and multiplying the clear sky index value by the clear-sky irradiance. The European Solar Radiation Atlas (ESRA) clear-sky solar irradiance model was used with the worldwide monthly climatology Linke turbidity parameter. CIADCast DNI estimates were obtained from GHI values using the DirIndex method with the ESRA clear-sky solar irradiance model. References in http://dx.doi.org/10.1016/j.solener.2017.07.045
