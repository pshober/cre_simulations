'-------------------------------'
'Author: Patrick Shober --------'
'Date: 24/02/2023 --------------'
'---- Near-Earth CRE Ages ------'
'This script randomly generates-'
'particles in near-Earth space--'
'and tracks the evolution until-'
'they impact the Earth. The-----'
'orbit just before impact, just-'
'after ejection, and the time---'
'inbetween is recorded.---------'
'-------------------------------'

import os
import sys
import argparse
from glob import glob
import time
import csv
from astropy import coordinates as coord
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time
from astropy.io import ascii
from astropy import units as u
from astropy import constants as const
from mpi4py import MPI
import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
import yaml
import datetime
import rebound
import reboundx
import shutil
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
from astropy.io import fits
import pymap3d as pm
from pathlib import Path
import logging
import re

def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def sim_setup():

    '''SET UP THE SOLAR SYSTEM'''
    sim = rebound.Simulation()
    sim.units = ('yr', 'AU', 'Msun')

    sim.add("Sun")
    sim.add("Mercury")
    sim.add("Venus")
    sim.add("399") # Earth
    sim.add("301") # Moon
    sim.add("Mars")
    sim.add("Jupiter")
    sim.add("Saturn")
    sim.add("Uranus")
    sim.add("Neptune")

    active_particles = sim.N
    sim.N_active = active_particles

    # Set planets' radii in AU
    sim.particles[0].r = 4.65e-3
    sim.particles[1].r = 1.63e-5
    sim.particles[2].r = 4.045e-5
    sim.particles[3].r = 4.26e-5
    sim.particles[4].r = 1.16e-5
    sim.particles[5].r = 2.27e-5
    sim.particles[6].r = 4.78e-4
    sim.particles[7].r = 4.03e-4
    sim.particles[8].r = 1.71e-4
    sim.particles[9].r = 1.646e-4

    return sim, active_particles

number_of_particles = 1000
n_clones = 10
input_near_earth_pop_granvik = "/home/patrick/Downloads/Granvik+_2018_Icarus.csv"

df = pd.read_csv(input_near_earth_pop_granvik)

semi_major_axis = np.array(data['semi_major_axis'])
eccentricity = np.array(data['eccentricity'])
inclination = np.array(data['inclination'])
long_ascending_node = np.array(data['long_ascending_node'])
argument_perihelion = np.array(data['argument_perihelion'])
mean_anomaly = np.array(data['mean_anomaly'])
absolute_magnitude = np.array(data['absolute_magnitude'])

# divide up the dataset into reasonable chunks that can be integrated
len_df = len(df)
steps = int(len_df / number_of_particles)
remainder = len_df % number_of_particles  # add to last sim

# create save folder
date_str = datetime.datetime.now().strftime('%Y%m%d')
save_folder = os.path.join(os.getcwd(),'nea_cre_sims_'+date_str)
if not os.path.isdir(save_folder): # Create the directory if it doesn't exist
    os.mkdir(save_folder)
else:
    # Removes all the subdirectories
    shutil.rmtree(save_folder)
    os.mkdir(save_folder)

sim, active_particles = sim_setup()
sim.save("solar_system.bin")
for i in range(steps):
    sim = rebound.Simulation("solar_system.bin")  # start rebound simulation

    # add particles to simulation
    for n in range(number_of_particles): # adds particles within errors of each body
        if i == steps - 1:
            a = semi_major_axis[i*number_of_particles:],
            e = eccentricity[i*number_of_particles:],
            inc = inclination[i*number_of_particles:],
            Omega = long_ascending_node[i*number_of_particles:],
            omega = argument_perihelion[i*number_of_particles:],
            M = mean_anomaly[i*number_of_particles:]
        else:
            a = semi_major_axis[i*number_of_particles:(i*number_of_particles)+n],
            e = eccentricity[i*number_of_particles:(i*number_of_particles)+n],
            inc = inclination[i*number_of_particles:(i*number_of_particles)+n],
            Omega = long_ascending_node[i*number_of_particles:(i*number_of_particles)+n],
            omega = argument_perihelion[i*number_of_particles:(i*number_of_particles)+n],
            M = mean_anomaly[i*number_of_particles:(i*number_of_particles)+n])

        # add the particle from the dataset
        sim.add(a=a, e=e, inc=inc, Omega=Omega, omega=omega, M=M)

        # store pos/vel to generate clones based on them
        x, y, z = sim.particles[0].x, sim.particles[0].y, sim.particles[0].z
        vx, vy, vz = sim.particles[0].vx, sim.particles[0].vy, sim.particles[0].vz

        # generate random position and velocity deviation (AU, AU/yr)
        pos_dev_magnitude = (1000.0 * u.m).to(u.au).value
        vel_dev_magnitude = (100.0 * (u.m / u.s)).to(u.au/u.yr).value

        pos_variations = sample_spherical(n_clones - 1) * pos_dev_magnitude
        vel_variations = sample_spherical(n_clones - 1) * vel_dev_magnitude

        x_clones, y_clones, z_clones = x+pos_variations[0], y+pos_variations[1], z+pos_variations[2]
        vx_clones, vy_clones, vz_clones = x+vel_variations[0], y+vel_variations[1], z+vel_variations[2]

        # add clones
        for c in range(n_clones-1):
            sim.add(x=x_clones[c],y=x_clones[c],z=z_clones[c],vx=vx_clones[c],vy=vy_clones[c],vz=vz_clones[c])

    # integration params
    n_outputs = 100
    tmax = 50.0
    times = np.linspace(0, tmax, n_outputs)
    results = np.zeros(shape=(ng_out+n_outputs, sim.N, 19)) * np.nan

    # integrate
    for i, step in enumerate(times):
        percent = round(((i+1)/n_outputs*100),2)
        print(f"{percent} Integrated\nTimestep: {sim.dt}", flush=True, end="\033[F")

        for j in range(sim.N):
            if rank == 0 or j >= active_particles + len(object_list):
                # Correctly add to arrays no matter what core
                if rank == 0:
                    k = j
                elif j >= active_particles + len(object_list):
                    k = j - active_particles - len(object_list)

                # if rank != 0:
                #     logging.debug(f"rank: {rank}; Recorded Index: {j + (rank*(sim.N-active_particles))}; j: {j}; k: {k}; step: {step}")

                results[i+ng_out,k,0] = step
                results[i+ng_out,k,1] = j + (rank*(sim.N-active_particles-len(object_list)))
                results[i+ng_out,k,8] = sim.particles[j].a
                results[i+ng_out,k,9] = sim.particles[j].e
                results[i+ng_out,k,10] = sim.particles[j].inc
                results[i+ng_out,k,11] = sim.particles[j].Omega
                results[i+ng_out,k,12] = sim.particles[j].omega

    # save fits file in folder
    results_fits = fits.PrimaryHDU()
    results_name = os.path.join(save_folder, f'results{i:05}.fits')
    results_fits.writeto(results_name)

    cols = [fits.Column(name='time', format='D', array=results_all[:,0]),
            fits.Column(name='index', format='D', array=results_all[:,1]),
            fits.Column(name='a', format='D', array=results_all[:,2]),
            fits.Column(name='e', format='D', array=results_all[:,3]),
            fits.Column(name='inc', format='D', array=results_all[:,4]),
            fits.Column(name='Omega', format='D', array=results_all[:,5]),
            fits.Column(name='omega', format='D', array=results_all[:,6])]

    results_fits = fits.BinTableHDU.from_columns(cols)

    results_fits_list = fits.open(results_name, mode='append')
    results_fits_list.append(results_fits)
    results_fits_list.writeto(results_name, overwrite=True)
    results_fits_list.close()
