""" rip demo """

__author__ = 'Reza Lotun'

import cPickle

import numpy

import spyke.surf
import spyke.stream
from spyke.detect import Collection, Template



class Ripper(object):
    """ Exhaustively matches the templates across the surf file, calculating
    residual squared error for each point in time.
    """
    def __init__(self, collection, surf_file):
        self.collection = collection
        self.surf_file = surf_file
        self.surf = None
        self.surf_stream = None
        self.itime = None
        self.ripped_templates = []
        self.surf = spyke.surf.File(self.surf_file)
        self.surf.parse()

    def reset(self):
        """ Get stream at beginning of surf file """
        self.surf_stream = spyke.stream.Stream(self.surf.highpassrecords)
        self.itime = self.surf_stream.records[0].TimeStamp

    def cleanup(self):
        self.surf.close()
        # XXX
        temp_ripped = 'ripped'
        f = file(temp_ripped, 'wb')
        cPickle.dump(self.collection, f, -1)

    def rip(self):
        print 'Starting ripping process...'
        self.reset()
        for template in self.collection.templates:
            print 'Ripping ' + str(template)
            self.fitTemplate(template)
            self.reset()

        self.cleanup()

    def fitTemplate(self, template):
        window = 1e3     # ten seconds

        start = self.itime

        # width of our template
        template_array = template.mean().data
        width = template_array.shape[1]

        # when we reach this point, we have to allocate a new chunk
        last_point = start + window - width

        template.ripped_match = []
        try:
            while True:
                # fit to whole file
                chunk = self.surf_stream[start:start + window].data
                print chunk
                for i in xrange(start, last_point + 1):
                    print i
                    data = template.mean().data
                    error = sum(sum((chunk[i:i + width] - data) ** 2))
                    template.ripped_match.append(error)
                    print error
                start = last_point + 1
                last_point = start + window - width
        except:
            # XXX: end of file?
            return


if __name__ == '__main__':
    import sys
    from spyke.gui.plot import filenames as FILENAMES
    if len(sys.argv) > 2:
        surf_name = sys.argv[1]
        collection_name = sys.argv[2]
    else:
        collection_name = 'collection.pickle'
        for surf_name in FILENAMES:
            try:
                stat = os.stat(filename)
                break
            except:
                continue

    collection = cPickle.load(file(collection_name))
    ripper = Ripper(collection, surf_name)
    ripper.rip()

