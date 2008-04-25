"""Provides an interface to a collection, used by gui tools"""

__author__ = 'Reza Lotun'

import spyke
from spyke import SpykeError


class CollectionError(SpykeError):


class CollectionManager(object):
    """Provides an abstraction layer around a collection, suitable for manipulation
    by subsystems with difference conceptions of how a collection is represented"""
    def __init__(self, collection):
        self.collection = collection

    def display(self, item):
        self._plot(item, visible=True)

    def undisplay(self, item):
        self._plot(item, visible=False)

    def _plot(self, item, visible=True):
        """Plot an item. Item must be a PlottedItem"""
        pass

    def bin(self, item):
        """Demote item (that is, either move it to unsorted list or the recycle bin)"""
        self.collection.unsorted_spikes.remove(item)
        self.collection.recycle_bin.append(item)

    def deleteTemplate(self, template):
        """Remove a template from the collection"""
        # check if this template contains spikes
        if len(template) > 0:
            for spike in template:
                self.collection.recycle_bin.append(spike)
        # finally, remove template
        self.collection.templates.remove(template)
        # TODO: cleanse template from plot

    def addToTemplate(self, item, template):
        """Add a spike to the template"""
        template.add(item)

    def removeFromTemplate(self, item, template):
        """Remove a spike from a template"""
        template.remove(item)
        self.collection.recycle_bin.append(item)

    def setTemplateChannels(self, template, channels):
        """Set the channel mask for a template"""
        pass
