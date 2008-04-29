"""rip demo"""

__author__ = 'Reza Lotun'

import cPickle
import struct

import numpy

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import spyke.surf
import spyke.stream
from spyke.detect import Collection, Template



class Ripper(object):
    """Exhaustively matches the templates across the surf file, calculating
    residual squared error for each point in time"""
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
        """Get stream at beginning of surf file"""
        self.surf_stream = spyke.stream.Stream(self.surf.highpassrecords)
        self.itime = self.surf_stream.records[0].TimeStamp

    def cleanup(self):
        self.surf.close()
        # XXX
        temp_ripped = 'ripped'
        f = file(temp_ripped, 'wb')
        cPickle.dump(self.collection, f, -1)

        self.plot()

    def plot(self):
        for i, template in enumerate(self.collection.templates):
            fig = Figure()
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            ax.plot(template.ripped_match)
            ax.set_title(str(template))
            canvas.print_figure(str(i)+str(template))

            fig = Figure()
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            ax.hist(template.ripped_match)
            canvas.print_figure(str(i)+'hist'+str(template))

    def rip(self):
        print 'Starting ripping process...'
        self.reset()
        for template in self.collection.templates:
            print 'Ripping ' + str(template)
            self.fitTemplate(template)
            self.reset()

        self.cleanup()

    def fitTemplate(self, template):
        window = int(1e6)     # ten seconds

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
                buf_size = chunk.shape[1]
                end_point = buf_size - width
                #print chunk
                for i in xrange(end_point):
                    #print i
                    data = template.mean().data
                    #print chunk[:,i: i + width].shape, data.shape
                    error = sum(sum((chunk[:,i:i + width] - data) ** 2))
                    template.ripped_match.append(error)
                    #print error
                start = last_point + 1
                last_point = start + window - width
        except struct.error:
            # XXX: end of file?
            print 'Done!'
            return
        except:
            raise


if __name__ == '__main__':
    import sys
    import os
    from spyke.gui.plot import filenames as FILENAMES
    if len(sys.argv) > 2:
        surf_name = sys.argv[1]
        collection_name = sys.argv[2]
    else:
        collection_name = 'collection.pickle'
        for surf_name in FILENAMES:
            try:
                stat = os.stat(surf_name)
                print surf_name
                break
            except:
                continue

    collection = cPickle.load(file(collection_name, 'rb'))
    print surf_name
    ripper = Ripper(collection, surf_name)
    ripper.rip()

