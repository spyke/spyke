import mdp
X = np.random.random((100000, 500))

node = mdp.nodes.PCANode()
node.execute(X)


# multithreading/processing seems slower, at least in this case:

flow = mdp.parallel.ParallelFlow([node])
scheduler = mdp.parallel.ProcessScheduler()
flow.train([[X]], scheduler=scheduler)




node.execute(X)
'''
- should really check to see if waveform data is gaussian before applying PCA to it. If it isn't, should use ICA, according to Odelia Schwartz. But, that's for modelling the data. All we're really interested in is clustering it.


- restore extract pane to spyke, one line per extraction method:
    - position, PCA, ICA, splines
        - need to get knots for splines - should this be on a per raw chan basis? maybe these pole positions should be decided upon once on a sample of spikes, and then hard coded.
    - auto column with check box to set which type of extraction to do auto during/after detection
    - extract button beside each method
    - extract all button below
    - populate clustering param box with only those params that are available
        - this also means auto generating the correct number of entries for methods like PCA and ICA
    - maybe stop auto plotting on detection, since default plot params might not match chosen extraction params
    - during clustering, when no existing clusters selected, and certain number of PCA/ICA components are selected (or a certain variance to explain is selected/entered somehow), do PCA/ICA on concatenation of each spike's chans, doing a separate run for all 54 possible maxchans. Save eigenvalues for each spike? Sure, in a temp array during clustering. Not much use after, since they're only of use relative to the other spikes with the same maxchans. May as well just recalc eigenvectors and values each time
        - when existing clusters/spikes selected and certain chans selected, do PCA/ICA only on concatenation of selected chans, then cluster
        - when existing clusters/spikes selected, but no chans selected, do PCA on concatenation of all overlapping chans
    - will need to generate temporary matrix of feature values for PCA/ICA - can't save this stuff in spikes array due to arbitrary number of components

    - can't think of how to visualize eigenvalues from concatenated PCA when dealing with different chan sets for different spikes. Nick does this somehow...

    - actually, ICA, even FastICA, seems way way slower than PCA. Unacceptably so.

- parameterize shape instead, using cubic splines. First, figure out ideal positions of knots, by taking maxchan waveforms of random selection of spikes detected. Do this fresh every time every time you do initial clustering on new detection?
    - during detection don't extract position
    - for each spike, get splines coeffs from maxchans. Then, 
    
    
- matching unsorted spikes to existing clusters:
    - when you hit U:
        - iterate over all unsorted spikes, for each spike get rms error vs each cluster, and record which cluster it fits best, and what the error is vs that cluster
        - decide which cluster i has the most best fits
        - deselect everything
        - create a new -1 cluster (deleting any previously existing one first, like during save) consisting only of those spikes that best fit cluster i, and only of those spikes whose rms error is less than an rms error scroll edit widget next to the U in the toolbar (defaults to 5mV). Select both clusters i and -1, so they're immediately compared, and so you can start hitting R to randomly sample them, or immed merge them. Whenever rms error scroll edit is changed, immed replace the spikes that belong to cluster -1. Every time you hit U, replace the existing set of errors, recalc everything
'''
