#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Plot native eigenproblem/boundary/contour CSVs with Matplotlib 3.11.2."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def rows(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def values(data, key):
    return np.array([float(row[key]) for row in data])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','svg.hashsalt':'openpfc-spectra'})
    blue,red,green='#526a82','#a44d2d','#358075'
    def read(case,name):return rows(args.runs/case/'results'/name)
    def save(fig,path):
        target=args.output/path;target.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(target,metadata={'Date':None});plt.close(fig)

    data=read('oscillator','energies.csv')
    fig,axes=plt.subplots(1,2,figsize=(9.2,3.7),layout='constrained')
    for mode,color in zip(range(1,5),(blue,red,green,'#876b95')):
        subset=[r for r in data if int(r['mode'])==mode]
        axes[0].plot(values(subset,'n'),values(subset,'eigenvalue'),'o-',color=color,ms=3,label=f'Mode {mode}')
        axes[0].axhline(2*mode-1,color=color,ls='--',lw=.8,alpha=.6)
        axes[1].semilogy(values(subset,'n'),np.maximum(abs(values(subset,'eigenvalue')-values(subset,'exact')),1e-16),'o-',color=color,ms=3)
    axes[0].set(title='Periodic truncation of a harmonic trap',xlabel='N',ylabel='Energy')
    axes[1].set(title='Error relative to infinite-line energy',xlabel='N',ylabel='Absolute error')
    axes[0].legend(frameon=False,fontsize=8)
    for ax in axes:ax.grid(alpha=.15)
    save(fig,'fourier/harmonic-oscillator/figures/energies.svg')

    data=read('mathieu','mathieu.csv')
    fig,ax=plt.subplots(figsize=(7.2,4.5),layout='constrained')
    for mode in range(1,12):
        subset=[r for r in data if int(r['mode'])==mode]
        ax.plot(values(subset,'q'),values(subset,'eigenvalue'),'-' if mode%2 else '--',lw=1.2,color=blue if mode%2 else green)
    ax.set(title='Eleven lowest periodic Mathieu eigenvalues · N = 42',xlabel='Potential strength q',ylabel='Eigenvalue')
    ax.grid(alpha=.15)
    save(fig,'fourier/mathieu/figures/mathieu.svg')

    spectrum=read('eigenmodes','spectrum.csv');stiff=read('stiffness','spectrum.csv')
    modes=read('eigenmodes','modes.csv');high=read('stiffness','modes.csv')
    fig,axes=plt.subplots(2,2,figsize=(9.2,6.5),layout='constrained')
    axes[0,0].loglog(values(spectrum,'mode'),values(spectrum,'eigenvalue'),'.-',ms=3,color=blue,label='Degree 36')
    axes[0,0].loglog(values(stiff,'mode'),values(stiff,'eigenvalue'),'.-',ms=2,color=green,label='Degree 60')
    axes[0,0].loglog(values(stiff,'mode'),values(stiff,'continuum'),'--',color=red,label='Continuum')
    axes[0,0].set(title='Physical and grid-scale spectrum',xlabel='Mode index',ylabel='Eigenvalue of −D²')
    axes[0,0].legend(frameon=False,fontsize=8)
    for position,mode in enumerate((5,10,15,20,25,30)):
        line=[r for r in modes if int(r['mode'])==mode and r['kind']=='curve']
        axes[0,1].plot(values(line,'x'),position+.34*values(line,'u'),color=blue,lw=.9)
    axes[0,1].set(title='Degree 36 · selected normalized shapes',xlabel='x',yticks=range(6),yticklabels=['5','10','15','20','25','30'],ylabel='Mode index · curves vertically offset')
    line=[r for r in high if int(r['mode'])==15 and r['kind']=='curve']
    x=values(line,'x');u=values(line,'u');exact=np.sin(15*np.pi*(x+1)/2)
    exact*=np.dot(exact,u)/np.dot(exact,exact)
    axes[1,0].plot(x,u,color=blue,label='Degree 60')
    axes[1,0].plot(x,exact,'--',color=red,label='Continuum shape, scaled')
    axes[1,0].set(title='A resolved physical mode · j = 15',xlabel='x',ylabel='Normalized u')
    axes[1,0].legend(frameon=False,fontsize=8)
    line=[r for r in high if int(r['mode'])==59 and r['kind']=='node' and abs(float(r['x']))<1]
    axes[1,1].semilogy(values(line,'x'),abs(values(line,'u')),'o-',ms=3,color=green)
    axes[1,1].set(title=f'Largest mode · λmax = {float(stiff[-1]["eigenvalue"])/60**4:.4f} N⁴',xlabel='x',ylabel='Absolute nodal amplitude')
    for ax in axes.flat:ax.grid(alpha=.15)
    save(fig,'polynomials/eigenmodes/figures/spectrum.svg')

    data=read('airy','airy.csv');eigen=read('airy','eigenvalues.csv')
    fig,axes=plt.subplots(2,2,figsize=(9.2,6),layout='constrained')
    for ax,n in zip(axes.flat,(12,24,36,48)):
        line=[r for r in data if int(r['n'])==n and r['kind']=='curve']
        nodes=[r for r in data if int(r['n'])==n and r['kind']=='node']
        lam=next(float(r['eigenvalue']) for r in eigen if int(r['n'])==n)
        ax.plot(values(line,'x'),values(line,'u'),color=blue)
        ax.plot(values(nodes,'x'),values(nodes,'u'),'o',ms=3,color=red)
        ax.set(title=f'N = {n} · λ₅ = {lam:.9f}',xlabel='x',ylabel='u(x), fixed u(0)')
        ax.grid(alpha=.15)
    save(fig,'polynomials/airy/figures/airy.svg')

    data=read('membrane','modes.csv');eigen=read('membrane','spectrum.csv')
    fig,axes=plt.subplots(2,2,figsize=(8.8,6.6),layout='constrained')
    for ax,mode in zip(axes.flat,range(1,5)):
        subset=[r for r in data if int(r['mode'])==mode]
        x=values(subset[:17],'x');y=values(subset[::17],'y');u=values(subset,'u').reshape(17,17)
        image=ax.contourf(x,y,u,levels=np.linspace(-1,1,25),cmap='RdBu_r')
        lam=float(eigen[mode-1]['eigenvalue'])
        ax.set(title=f'Mode {mode} · λ = {lam:.8f}',xlabel='x',ylabel='y',aspect='equal')
    fig.colorbar(image,ax=list(axes.flat),label='Normalized eigenfunction',shrink=.8,ticks=[-1,-.5,0,.5,1])
    save(fig,'polynomials/membrane/figures/membrane.svg')

    for case in ('poisson','helmholtz','laplace'):
        data=read(case,'solution.csv');side=int(np.sqrt(len(data)))
        x=values(data[:side],'x');y=values(data[::side],'y');u=values(data,'u').reshape(side,side);xx,yy=np.meshgrid(x,y)
        fig,axes=plt.subplots(1,2,figsize=(9.2,4),layout='constrained')
        if case=='laplace':
            axes[0].plot(x,u[0,:],'o-',color=blue,ms=3,label='Upper edge · coordinate x')
            axes[0].plot(y,u[:,0],'o-',color=red,ms=3,label='Right edge · coordinate y')
            axes[0].set(title='Prescribed boundary values',xlabel='Edge coordinate',ylabel='u')
            axes[0].legend(frameon=False,fontsize=8);axes[0].grid(alpha=.15)
        else:
            forcing=10*np.sin(8*xx*(yy-1)) if case=='poisson' else np.exp(-10*((yy-1)**2+(xx-.5)**2))
            image=axes[0].contourf(x,y,forcing,levels=25,cmap='RdBu_r' if case=='poisson' else 'viridis')
            fig.colorbar(image,ax=axes[0],label='f(x,y)',shrink=.8)
            axes[0].set(title='Prescribed forcing',xlabel='x',ylabel='y',aspect='equal')
        image=axes[1].contourf(x,y,u,levels=25,cmap='RdBu_r')
        fig.colorbar(image,ax=axes[1],label='u(x,y)',shrink=.8)
        axes[1].set(title='Computed '+case.capitalize()+' field · N = 24',xlabel='x',ylabel='y',aspect='equal')
        save(fig,f'polynomials/{case}-square/figures/{case}.svg')

    data=read('gamma','gamma.csv');convergence=read('gamma','convergence.csv')
    x=values(data[:76],'real');y=values(data[::76],'imag');magnitude=values(data,'gamma_abs').reshape(51,76)
    fig,axes=plt.subplots(1,2,figsize=(9.2,4),layout='constrained')
    image=axes[0].pcolormesh(x,y,magnitude,shading='nearest',cmap='viridis',norm=LogNorm(vmin=np.nanmin(magnitude),vmax=6))
    axes[0].plot([-3,-2,-1,0],[0,0,0,0],'x',color=red,ms=5,label='Poles')
    axes[0].set(title='Gamma magnitude · 70 contour samples',xlabel='Re z',ylabel='Im z')
    axes[0].legend(frameon=False,fontsize=8)
    fig.colorbar(image,ax=axes[0],label='|Γ(z)| · logarithmic colour',extend='max',shrink=.8)
    for z,color in ((.5,blue),(1.5,green),(4.,red)):
        subset=[r for r in convergence if float(r['z'])==z]
        axes[1].semilogy(values(subset,'samples'),np.maximum(values(subset,'error'),1e-16),'o-',color=color,ms=3,label=f'z = {z:g}')
    axes[1].set(title='Reciprocal checks at positive real z',xlabel='Contour samples',ylabel='Absolute reciprocal error')
    axes[1].legend(frameon=False,fontsize=8);axes[1].grid(alpha=.15)
    save(fig,'quadrature/complex-contour/figures/gamma.svg')

    data=read('multigrid','field_33.csv');convergence=read('multigrid','convergence.csv')
    plane=[r for r in data if int(r['k'])==16];u=values(plane,'u').reshape(33,33);x=np.linspace(0,1,33)
    fig,axes=plt.subplots(1,2,figsize=(9.2,4),layout='constrained')
    image=axes[0].contourf(x,x,u,levels=25,cmap='viridis');fig.colorbar(image,ax=axes[0],label='u',shrink=.8)
    axes[0].set(title='Computed midplane · z = 0.5',xlabel='x',ylabel='y',aspect='equal')
    h=values(convergence,'h');error=values(convergence,'relative_error')
    axes[1].loglog(h,error,'o-',color=blue,label='Native multigrid-GMRES')
    axes[1].loglog(h,error[0]*(h/h[0])**2,'--',color=red,label='Second-order guide')
    axes[1].set(title='Continuous-field error',xlabel='Grid spacing h',ylabel='Relative L₂ error')
    axes[1].set_xticks([1/32,1/16,1/8],['1/32','1/16','1/8']);axes[1].tick_params(axis='x',which='minor',labelbottom=False)
    axes[1].legend(frameon=False,fontsize=8);axes[1].grid(alpha=.15)
    save(fig,'elliptic/multigrid/figures/multigrid.svg')


if __name__=='__main__':main()
