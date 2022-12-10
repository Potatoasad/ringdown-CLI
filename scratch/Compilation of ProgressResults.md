# Compilation of Progress/Results

### Simple Back to basics plot

So I wanted to investigate if there was any Bias in the ringdown code itself. 

What i mean by this is that if I plug in a sinosoid wave, where I know the $f$, $\gamma$ and Amplitude $A$, then do I faithfully get that value back?

###### Bias checking

To test this I do runs for the following parameters:

| Noise Amplitude | Plot |
| --------------- | ---- |
| 1               |      |
| 0.1             |      |
| 0.01            |      |

 The plot should get thinner and thinner around the true value. 

###### Noise Realisation test

Also I ran 40 different realisations of the input noise for the same signal and did 40 identical inference procedures. Of course since there is a different realisation of the noise each time one should get a different contour. 

However the true value of $f$, $\gamma$ should be within the 90% contour about 90% of the time, and similarly for the others. 

Here is an animation of all different runs:

>  insert animation

Here is the plot that shows a plot of $y$ vs $x$ where:
$$
y \% \textrm{ of points were inside the x\% contour for the variable }\gamma 
$$
Normally the scenario where our analysis is correct is where this plot falls on the line $y = x$. Our plot is similar which shows that there probably isn't that much bias

> insert plot

## $M$ vs $\chi$ plot for GW150914 for different target samples

We go through the 10th,20th ... 90th percentile samples for H1_peak time and geocent_peak time.





















