__all__ = ['kdeplot_2d_clevels_return']

import ringdown
import numpy as np
import matplotlib.pyplot as plt

def kdeplot_2d_clevels_return(xs, ys, levels=11, **kwargs):
    try:
        xs = xs.values.astype(float)
        ys = ys.values.astype(float)
    except AttributeError:
        pass
    
    try:
        len(levels)
        f = 1 - np.array(levels)
    except TypeError:
        f = linspace(0, 1, levels+2)[1:-1]
    if kwargs.get('auto_bound', False):
        kwargs['xlow'] = min(xs)
        kwargs['xhigh'] = max(xs)
        kwargs['ylow'] = min(ys)
        kwargs['yhigh'] = max(ys)
    kde_kws = {k: kwargs.pop(k, None) for k in ['xlow', 'xhigh', 'ylow', 'yhigh']}
    k = ringdown.Bounded_2d_kde(np.column_stack((xs, ys)), **kde_kws)
    size = max(10*(len(f)+2), 500)
    c = np.random.choice(len(xs), size=size)
    p = k(np.column_stack((xs[c], ys[c])))
    i = np.argsort(p)
    l = np.array([p[i[int(round(ff*len(i)))]] for ff in f])

    x = np.linspace(0, 1, 128)
    y = np.linspace(0, np.pi/2, 128)

    XS, YS = np.meshgrid(x, y, indexing='ij')
    ZS = k(np.column_stack((XS.flatten(), YS.flatten()))).reshape(XS.shape)
    
    return XS, YS, ZS, l