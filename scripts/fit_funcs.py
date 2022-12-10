import ringdown
import numpy as np
import pandas as pd
# Define functions
def get_target(event, N_samps=None, target_sample_rate=4096, f_low=20, f_ref=20, q=0.5):
    """
    Calculates the sky location and start time that will be chosen for the fit object. 
    This function will 
    1. randomly get a subset posterior samples (of size `N_samps`)
    2. calculate the geocenter peak strain for each of the posterior samples
    3. choose the `q`-th quantile (i.e. `q*100`-th percentile) sample and package its t0, ra, dec and psi 
    
    
    Parameters
    -----------
    event : event object
        The name of the event e.g. db.event("GW150914")
    N_samps : int
        This many samples will be chosen from the posterior for the actual analysis. Some IMR posteriors have too many samples.
        Defaults to using all samples
    target_sample_rate : float
        The waveform generator will use this sample rate when generating waveforms
    f_low : float (default : 20)
        f_low
    f_ref : float (default : 20)
        reference frequency
    q :  float (default : 0.5)
        The chosen sample for the target will be the one which lands on the `q`-th quantile when ordered by the geocent peak time. 
        
        
    Returns
    --------
    args : Dict with keys 't0', 'ra', 'dec' and 'psi' 
        This is a dictionary of the chosen target parameters.
    median_samp : pd.Series
        This is the chosen median sample. Returned if any other parameters like final_mass and final_spin are needed. 
    samps : pd.DataFrame
        A dataframe of all the samples that were chosen for calculation and their corresponding peak times at each ifo.
    
    """
    #event = db.event(eventname)
    strains = event.strain()
    posts = event.posteriors()

    N_samps = N_samps or len(posts)

    try:
        f_low = float(event.read_posterior_file_from_schema('f_low'))
        f_ref = float(event.read_posterior_file_from_schema('f_ref'))
        waveform_name = posts['waveform_name'].unique()[0]
        waveform_code = int(posts['waveform_code'].unique()[0])
    except:
        print("Falling back to f_ref = 20, f_low=20")
        waveform_name = posts['waveform_name'].unique()[0]
        waveform_code = int(posts['waveform_code'].unique()[0])
        f_low = f_low
        f_ref = f_ref


    print(f"Using {waveform_name}")
    wf_ls = waveform_code
    sample_rate = np.max([a.fsamp for a in strains.values()])


    samps = [x.to_dict() for i,x in posts.sample(N_samps).iterrows()]

    for i,x in enumerate(samps):
        t_peak, t_dict, hp, hc = ringdown.complex_strain_peak_time_td(x,
                                                                      wf=wf_ls, dt=1/target_sample_rate,
                                                                      f_ref=f_ref, f_low=f_low)
        samps[i].update({k+'_peak':v for k,v in t_dict.items()})

    samps = pd.DataFrame(samps)

    # Get median sample
    ref_ifo = 'H1'
    im = (samps[f'{ref_ifo}_peak'] - samps[f'{ref_ifo}_peak'].quantile(q)).abs().argmin()
    median_samp = samps.iloc[im]

    # Construct Arguments for set_target
    args = median_samp[['geocent_peak', 'ra','dec','psi']].rename({'geocent_peak':'t0'}).to_dict()
    print("The median time at H1 is: ", median_samp['H1_peak'], "s")

    # Get the mass time-scale
    Mass_Time = lambda M: 6.674e-11*M*1.989e+30/(3e8)**3
    t_M = Mass_Time(median_samp['final_mass'])
    print("The mass time-scale is: ", np.round(t_M*1000,3), "ms")

    times_above_below = (samps['H1_peak'].quantile(0.95) - samps['H1_peak'].quantile(0.05))/(2*t_M)
    print(f"The 90% CI of H1 peak time is +/- {np.round(times_above_below,1)} t_M")

    return args, median_samp, samps

def set_fit(event, target, mass_for_prior, duration=0.1, target_sample_rate=4096, model='mchiq_exact', modes=[(1, -2, 2, 2, 0), (1, -2, 2, 2, 1)], cond_kws = None, **model_kwargs):

    '''
    The convention for modes is (prograde, s, l, m, n)
    The options for models are mchi, mchiq, and mchiq_exact. 
    mchiq is for all beyond-GR EVP applications.
    You can change the beyon-GR model using the kwargs, which get passed to ringdown.Fit
    The coefficients must be created by fitting EVP data as a separate step.
    df_coeffs = [[a0,a1,a2,a3...],[a0,a1,a2,a3...],..]; n_modes x n_coeffs for the frequency shifts
    dg_coeffs = (similar structure as df) for the decay rate shifts
    '''
    
    #event = db.event(eventname)
    strains = event.strain()
    
    #Hacking in ability to change amplitude priors:
    #Aprior = model_kwargs.pop('flat_A',1)
    
    fit = ringdown.Fit(model=model, modes=modes, **model_kwargs)
    
    for ifo in strains.keys():
        fit.add_data(strains[ifo])
        
    sample_rate = np.max([f.fsamp for f in strains.values()])

    fit.set_target(**target, duration=duration)
    
    #Adding a cond kwargs to allow us to low pass or add other conditioning arguments
    default_kws = dict(digital_filter=True, trim=0.0)
    default_kws.update(cond_kws or {})
    
    #Revisit the trim = 0 setting. Why was this done? Naively this should be set to a nonzero value

    fit.condition_data(ds=int(sample_rate/target_sample_rate),**default_kws)
    fit.compute_acfs()

    #fit.update_prior(A_scale=5e-21, M_min=mass_for_prior*0.5,
    #                 M_max=mass_for_prior*2.0,
    #                 flat_A=1)
    fit.update_prior(A_scale=5e-21, M_min=mass_for_prior*0.5,
                     M_max=mass_for_prior*2.0,
                     flat_A=1)
    return fit