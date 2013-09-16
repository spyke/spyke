# code to run in IPython shell to test whether clustering info in spikes struct array and in
# the neurons dict is consistent:

for nid in sorted(self.sort.neurons):
    print(nid, (self.sort.neurons[nid].sids == np.where(self.sort.spikes['nid'] == nid)[0]).all())
