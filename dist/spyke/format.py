""" spyke.format

Class declarations of common objects used throughout spyke.
"""

__author__ = 'Reza Lotun'

import numpy

from spyke.stream import WaveForm

class Spike(WaveForm):
    """ A spike event """
    def __init__(self, waveform=None, channel=None, event_time=None):
        self.data = waveform.data
        self.ts = waveform.ts
        self.sampfreq = waveform.sampfreq
        self.channel = channel
        self.event_time = event_time
        self.name = str(self)

    def __str__(self):
        return 'Channel ' + str(self.channel) + ' time: ' + \
                str(self.event_time)

    def __hash__(self):
        return hash(str(self.channel) + str(self.event_time) + \
                                                        str(self.data))

    def __eq__(self, other):
        return hash(self) == hash(other)


class Template(set):
    """ A spike template is simply a collection of spikes. """
    def __init__(self, *args):
        set.__init__(self, *args)
        self.name = str(self)
        self.active_channels = None

    def mean(self):
        """ Returns the mean of all the contained spikes. """
        if len(self) == 0:
            return None

        sample = iter(self).next()
        dim = sample.data.shape
        _mean = Spike(sample)
        _mean.data = numpy.asarray([0.0] * dim[0] * dim[1]).reshape(dim)

        for num, spike in enumerate(self):
            _mean.data += spike.data

        _mean.data = _mean.data / (num + 1)

        return _mean

    def __hash__(self):
        # XXX hmmm how probable would collisions be using this...?
        return hash(str(self.mean()) + str(self))

    def __str__(self):
        return 'Template (' + str(len(self)) + ')'


class Collection(object):
    """ A container for Templates. Collections are associated with
    Surf Files. By default a Collection represents a single sorting session.
    Initially detected spikes will be added to a default set of spikes in a
    collection - these spikes will be differentiated through a combination
    of algorithmic and/or manual sorting.
    """
    def __init__(self, file=None):
        # XXX: populate this with pertinent info
        self.templates = []
        self.unsorted_spikes = []        # these represent unsorted spikes
        self.recycle_bin = []

    def __len__(self):
        return len(self.templates)

    def __str__(self):
        """ Pretty print the contents of the Collection."""
        s = []
        for t in self:
            s.extend([str(t), '\n'])
            for sp in t:
                s.extend(['\t', str(sp), '\n'])
        return ''.join(s)

    def __iter__(self):
        for template in self.templates:
            yield template

