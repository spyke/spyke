>>> figure()
<matplotlib.figure.Figure object at 0x03D94450>
>>> t = np.arange(start=-12/2-0-0, stop=-12/2-0+13-0, step=1)
>>> plot(h(t)*hamming(t, 12))
[<matplotlib.lines.Line2D object at 0x040FBE10>]
>>> t = np.arange(start=-12/2-0.25-0, stop=-12/2-0.25+13-0, step=1)
>>> plot(h(t)*hamming(t, 12))
[<matplotlib.lines.Line2D object at 0x04103F30>]
>>> t = np.arange(start=-12/2-0.5-0, stop=-12/2-0.5+13-0, step=1)
>>> plot(h(t)*hamming(t, 12))
[<matplotlib.lines.Line2D object at 0x040DE8D0>]
>>> t = np.arange(start=-12/2-0.75-0, stop=-12/2-0.75+13-0, step=1)
>>> plot(h(t)*hamming(t, 12))
[<matplotlib.lines.Line2D object at 0x04110D30>]
