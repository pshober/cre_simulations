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
    sim.integrator = 'mercurius'

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

    sim.move_to_com()

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

number_of_particles = 100
n_clones = 10
input_near_earth_pop_granvik = "/home/patrick/Downloads/Granvik+_2018_Icarus.csv"

df = pd.read_csv(input_near_earth_pop_granvik)

semi_major_axis = np.array(df['semi_major_axis'])
eccentricity = np.array(df['eccentricity'])
inclination = np.array(df['inclination'])
long_ascending_node = np.array(df['long_ascending_node'])
argument_perihelion = np.array(df['argument_perihelion'])
mean_anomaly = np.array(df['mean_anomaly'])
absolute_magnitude = np.array(df['absolute_magnitude'])

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

# set up base simulation and save to use as a restart (faster)
sim, active_particles = sim_setup()
sim.save("solar_system.bin")

for s in range(steps):

    sim = rebound.Simulation("solar_system.bin")  # start new rebound simulation

    if s == steps - 1:
        a = semi_major_axis[s*number_of_particles:]
        e = eccentricity[s*number_of_particles:]
        inc = inclination[s*number_of_particles:]
        Omega = long_ascending_node[s*number_of_particles:]
        omega = argument_perihelion[s*number_of_particles:]
        M = mean_anomaly[s*number_of_particles:]
    else:
        a = semi_major_axis[s*number_of_particles:(s*number_of_particles)+number_of_particles]
        e = eccentricity[s*number_of_particles:(s*number_of_particles)+number_of_particles]
        inc = inclination[s*number_of_particles:(s*number_of_particles)+number_of_particles]
        Omega = long_ascending_node[s*number_of_particles:(s*number_of_particles)+number_of_particles]
        omega = argument_perihelion[s*number_of_particles:(s*number_of_particles)+number_of_particles]
        M = mean_anomaly[s*number_of_particles:(s*number_of_particles)+number_of_particles]

    # add particles to simulation
    for n in range(len(a)): # adds particles within errors of each body

        # add the particle from the dataset
        sim.add(a=a[n], e=e[n], inc=inc[n], Omega=Omega[n], omega=omega[n], M=M[n], primary=sim.particles[0])

        # store pos/vel to generate clones based on them
        x, y, z = sim.particles[-1].xyz
        vx, vy, vz = sim.particles[-1].vxyz

        # generate random position and velocity deviation (AU, AU/yr)
        pos_dev_magnitude = (1000.0 * u.m).to(u.au).value
        vel_dev_magnitude = (100.0 * (u.m / u.s)).to(u.au/u.yr).value

        pos_variations = sample_spherical(n_clones - 1) * pos_dev_magnitude
        vel_variations = sample_spherical(n_clones - 1) * vel_dev_magnitude

        x_clones, y_clones, z_clones = x+pos_variations[0], y+pos_variations[1], z+pos_variations[2]
        vx_clones, vy_clones, vz_clones = vx+vel_variations[0], vy+vel_variations[1], vz+vel_variations[2]

        # add clones
        for c in range(n_clones-1):
            sim.add(x=x_clones[c],y=y_clones[c],z=z_clones[c],vx=vx_clones[c],vy=vy_clones[c],vz=vz_clones[c])

    # integration params
    n_outputs = 100
    tmax = 50.0
    times = np.linspace(0, tmax, n_outputs)
    results = np.zeros(shape=(n_outputs, sim.N, 7)) * np.nan

    # integrate
    for i, step in enumerate(times):
        percent = round(((i+1)/n_outputs*100),2)
        print(f"{percent} Integrated\nTimestep: {sim.dt}", flush=True, end="\033[F")

        for j in range(sim.N):
            results[i,j,0] = step
            results[i,j,1] = j

            if j == 0:  # if the sun...
                results[i,j,2] = np.nan
                results[i,j,3] = np.nan
                results[i,j,4] = np.nan
                results[i,j,5] = np.nan
                results[i,j,6] = np.nan
            else:
                sun_orbit = sim.particles[j].calculate_orbit(sim.particles[0])
                results[i,j,2] = sun_orbit.a
                results[i,j,3] = sun_orbit.e
                results[i,j,4] = sun_orbit.inc
                results[i,j,5] = sun_orbit.Omega
                results[i,j,6] = sun_orbit.omega

        sim.integrate(step)

    results = np.concatenate(results)

    # save fits file in folder
    results_fits = fits.PrimaryHDU()
    results_name = os.path.join(save_folder, f'results{s:05}.fits')
    results_fits.writeto(results_name)

    cols = [fits.Column(name='time', format='D', array=results[:,0]),
            fits.Column(name='index', format='D', array=results[:,1]),
            fits.Column(name='a', format='D', array=results[:,2]),
            fits.Column(name='e', format='D', array=results[:,3]),
            fits.Column(name='inc', format='D', array=results[:,4]),
            fits.Column(name='Omega', format='D', array=results[:,5]),
            fits.Column(name='omega', format='D', array=results[:,6])]

    results_fits = fits.BinTableHDU.from_columns(cols)

    results_fits_list = fits.open(results_name, mode='append')
    results_fits_list.append(results_fits)
    results_fits_list.writeto(results_name, overwrite=True)
    results_fits_list.close()
