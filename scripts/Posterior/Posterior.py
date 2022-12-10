__all__ = ['Posterior','ChargedPosterior']

import pandas as pd
import arviz as az
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from .BoundedKDE import kdeplot_2d_clevels_return

# Load a copy of the data into `Posterior` and don't 
# hold the file hostage. Not doing this leads to permission errors
# when overwriting results
az.rcParams["data.load"] = 'eager' 

dont_include_in_dataframe = ['']

make_rtheta = lambda x: pd.Series([np.sqrt(x['Q_charge']**2 + x['chi']**2), 
                                    np.arctan(x['Q_charge']/x['chi'])], 
                                   index=['r','theta'])

make_chiq = lambda x: pd.Series([x['r']*np.cos(x['theta']), 
                                    x['r']*np.sin(x['theta'])], 
                                   index=['chi','Q_charge'])


class Posterior:
	"""Class that holds posteriors. These can be pulled from a file"""
	def __init__(self, eventname, filename=None, inference_data=None, good_chains=None):
		self.eventname = eventname
		self.filename = filename
		self._data = None

		# Import data
		if self.filename is not None:
			self._inference = az.from_netcdf(self.filename)
		else:
			self._inference = inference_data

		# Select good chains
		if good_chains is not None:
			self._good_chains = good_chains
		else:
			if self._inference is not None:
				self._good_chains = list(self._inference.posterior.coords['chain'].values) # list of good chains
			else:
				self._good_chains = None

	@property
	def good_chains(self):
		return self._good_chains

	@good_chains.setter
	def good_chains(self, value):
		self._good_chains = value
		self._data = None # Trigger recalculation of dataframe nexttime

	def remove_bad_chains(self, bad_chain_list):
		for chain in bad_chain_list:
			self._good_chains.remove(chain)
		self._data = None # Trigger recalculation of dataframe nexttime

	"""
	Load posteriors from a ringdown result
	"""
	def load_data(self, filepath):
		self._inference = az.from_netcdf(filepath)

	"""
	Pulls out the columns from the data
	"""
	def get_cols(self, cols):
		return self._inference.posterior[cols].values

	"""
	Returns inference data with the right chains selected
	"""
	@property
	def inference(self):
		return self._inference.sel(chain=self.good_chains)

	"""
	Creates the dataframe
	"""
	def create_dataframe(self):
		all_variables = self.inference.posterior.data_vars.variables
		all_dimensions = self.inference.posterior.dims
		flat_vars =  {k: v.values.reshape(np.prod(v.shape[0]*v.shape[1])) for k,v in all_variables.items() if len(v.shape) <= 2}

		all_vars = flat_vars.copy()
		for k,v in all_variables.items():
		    if ((k+"_dim_0" in all_dimensions) and ('h_det' not in k) and (k not in flat_vars)):
		        dim_name = k+"_dim_0"
		        all_vars.update({k+f"_{i}": v.values[:,:,i].reshape(np.prod(v.shape[0]*v.shape[1])) for i in range(all_dimensions[dim_name])})

		return pd.DataFrame(all_vars)

	"""
	Pulls the data as a dataframe
	"""
	@property
	def data(self):
		if self._data is not None:
			return self._data
		else:
			self._data = self.create_dataframe()
			return self._data


	"""
	Creates a trace plot
	"""
	def plot_trace(self, chain=None, **kwargs):
		inference = self.inference
		if chain:
			az.plot_trace(inference.sel(chain=chain), **kwargs)
		else:
			az.plot_trace(inference, **kwargs)
		plt.show()


	"""
	Combines IMR results and Ringdown Results
	"""
	def combined_results(self, IMR=None, ringdown_colnames=None, IMR_colnames=None, imr_n_samples=None, Model=['Ringdown', 'IMR']):
		cols = ringdown_colnames
		the_result = pd.DataFrame({col[0]: self.data[col[1]].values for col in cols})
		the_result['Model'] = Model[0]

		if IMR is not None:
			if imr_n_samples is not None:
				IMR = IMR.sample(imr_n_samples)
			else:
				IMR = IMR.sample(len(self.data.index)) if len(self.data.index) < len(IMR.index) else IMR
			cols = IMR_colnames
			the_result2 = pd.DataFrame({col[0]: IMR[col[1]].values for col in cols})
			the_result2['Model'] = Model[1]
			the_result = pd.concat([the_result, the_result2]).reset_index(drop=True)

		return the_result

	"""
	Creates an M_chi plot
	"""
	def M_chi_plot(self, IMR=None, title=None, M_IMR_alias='final_mass', chi_IMR_alias='final_spin', **kwargs):
		ringdown_colnames = [(r'$M$','M'), (r'$\chi$','chi')]
		IMR_colnames = [(r'$M$',M_IMR_alias), (r'$\chi$',chi_IMR_alias)]
		M_chi = self.combined_results(IMR, ringdown_colnames, IMR_colnames, **kwargs)

		g = sns.jointplot(data=M_chi, x=r'$M$', y=r'$\chi$', hue='Model', 
						kind='kde', fill=True, levels=[0.3,0.6,0.9], 
						common_norm=False, alpha=0.7)

		if title is not None:
			g.fig.suptitle(title)
		else:
			g.fig.suptitle(f"{self.eventname} M vs "+r"$\chi$")

		g.fig.tight_layout()
		g.fig.subplots_adjust(top=0.95)
		#plt.show()
		return g

	"""
	Plot waveform reconstruction
	"""
	def waveform_reconstruction_plot(self, draw):
		pass
    
	
		

class ChargedPosterior(Posterior):
	"""lass that holds charged posteriors. These can be pulled from a file"""
	def __init__(self, *args, **kwargs):
		super(ChargedPosterior, self).__init__(*args, **kwargs)
		
	@property
	def Q(self):
		col = 'Q_charge'
		return self.data[col]

	@property
	def chi(self):
		col = 'chi'
		return self.data[col]

	@property
	def chi_q(self):
		cols = ['chi', 'Q_charge']
		return self.data[cols]

	@property
	def r_theta(self):
		self.data
		if ('r' not in self._data.columns) or ('theta' not in self._data.columns):
			self._data.loc[:, ['r','theta']] = self._data.apply(make_rtheta, result_type='expand', axis=1)
		return self.data[['r','theta']]

	"""
	Creates an chi-q plot
	"""
	def chi_q_plot(self, title=None, **kwargs):
		cols = [(r'$\chi$','chi'), (r'$Q$','Q_charge')]
		chi_q = self.combined_results(ringdown_colnames=cols, Model=['Charged Ringdown'])

		g = sns.jointplot(data=chi_q, x=r'$\chi$', y=r'$Q$', hue='Model', 
						kind='kde', fill=True, levels=[0.3,0.6,0.9], 
						common_norm=False, alpha=0.7)

		x = np.linspace(0,1,100)
		g.ax_joint.plot(x, np.sqrt(1-x**2))
		g.ax_joint.set_xlim((0,1))
		g.ax_joint.set_ylim((0,1))

		if title is not None:
			g.fig.suptitle(title)
		else:
			g.fig.suptitle(f"{self.eventname} "+r"$\chi$ vs "+r"$Q$")

		g.fig.tight_layout()
		g.fig.subplots_adjust(top=0.95)
		plt.show()
		return g

	"""
	Creates an chi-q hexbin plot
	"""
	def chi_q_plot_hexbin(self, title=None, **kwargs):
		cols = [(r'$\chi$','chi'), (r'$Q$','Q_charge')]
		chi_q = self.combined_results(ringdown_colnames=cols, Model=['Charged Ringdown'])

		g = sns.jointplot(data=chi_q, x=r'$\chi$', y=r'$Q$', 
						kind='hex', **kwargs)

		x = np.linspace(0,1,100)
		g.ax_joint.plot(x, np.sqrt(1-x**2))
		g.ax_joint.set_xlim((0,1))
		g.ax_joint.set_ylim((0,1))

		if title is not None:
			g.fig.suptitle(title)
		else:
			g.fig.suptitle(f"{self.eventname} "+r"$\chi$ vs "+r"$Q$"+" hexbin")

		g.fig.tight_layout()
		g.fig.subplots_adjust(top=0.95)
		plt.show()
		return g

	"""
	Creates a bounded chi-q plot
	"""
	def chi_q_plot_bounded(self, title=None, levels=[0.3,0.6,0.9], ax=None, **kwargs):
		RS, ThetaS, ZS, l = kdeplot_2d_clevels_return(self.r_theta['r'], self.r_theta['theta'], levels=levels)

		XS = np.array([[RS[i,j]*np.cos(ThetaS[i,j]) for i in range(RS.shape[0])] for j in range(RS.shape[1])])
		YS = np.array([[RS[i,j]*np.sin(ThetaS[i,j]) for i in range(RS.shape[0])] for j in range(RS.shape[1])])
		ZS = np.array([[ZS[i,j] for i in range(RS.shape[0])] for j in range(RS.shape[1])])

		if ax is None:
			fig, ax = plt.subplots(1, figsize=(5,5))
		ax.contourf(XS, YS, ZS, levels=[l[-i-1] for i in range(len(l))], alpha=0.5)
		t = np.linspace(0,1,100)
		ax.plot(t, np.sqrt(1-t**2))
		ax.set_aspect(1)
		ax.set_xlim((0,1))
		ax.set_ylim((0,1))
		ax.set_xlabel(r"$\chi$")
		ax.set_ylabel(r"$Q$")

		ax.plot(self.chi.median(),self.Q.median(),'+',c='k')
		ax.set_title(title or f"Bounded KDE plot for {self.eventname}")
		return ax

