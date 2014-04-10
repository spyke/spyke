Here's a tutorial that goes over the basics:

1. Run the program from the command line: `python main.py`. Status messages tend to be printed
back to the command line, so it's a good idea to keep an eye on it, at least initially.

2. Open a data file, typically a `.srf` file, such as this [sample
file](http://swindale.ecc.ubc.ca/spyke?action=AttachFile&do=get&target=ptc15_tr7c_r87_spont_20sec.srf).

3. Press the `Detect` button in the Detect tab. This will run spike detection on the entire
file, given the current detection settings.

4. When it's done, three new windows will open up: sort, cluster and matplotlib. The sort
window shows detected waveforms and cluster means, the cluster window shows each spike as a
point in some 3D cluster space, and the matplotlib window plots various other things as
appropriate. You can manipulate the view in the cluster window using the mouse, rotating,
zooming, and panning. Try using the different mouse buttons and scrollwheel, in tandem with
`Ctrl` and `Shift`. You can select detected unsorted events in the lower list in the sort
window. `Ctrl` and `Shift` allow for multiple selections.

5. Deselect any selected spikes (`ESC` is good for this). Click the `/` button in the sort
window toolbar, or just press the `/` key. This will roughly divide the spikes up into
clusters based on their maximum channel. The neuron list in the sort window (upper left list)
is now populated, and the points in the cluster window are now coloured. You can select any
number of units that you like, and their mean waveforms will be plotted. You can also select
specific spikes of the selected neurons in the upper right list in the sort window. Pressing
the `R` toolbar button or keyboard key will randomly select a subset of spikes from the
selected neurons. The slider widget just below the sort window toolbar provides a sliding
selection time window of points. Most of the units here have very few spikes, since it's such
a short example file, so they might not look very impressive in the cluster window. You'll
probably have to select at least a few clusters to display a decent number of spikes.

6. Hit `Esc` again (possibly twice) to clear any selections. Now press the renumber button in
the sort window toolbar (looks like a pencil on my system), or press the `#` keyboard
character. This will renumber all clusters in vertical spatial order. Hit `Yes`.

7. Assuming you're working on the sample file linked to above, select clusters 11 and 12 in
the sort window (or any few units that have some channel overlap). Once selected, hit `Enter`,
and they're plotted in the cluster window. Back in the main window, in the Cluster tab,
there's a section labelled Plotting, where you can control the dimensions plotted in the
cluster window. It defaults to x0, y0 and Vpp for the 3 dimensions. x0 and y0 are the
estimated positions of each spike along the polytrode, and Vpp is the peak to peak voltage on
the maximum channel. You can change these dimensions to c0, c1, c2 via the dropdown lists
(labelled x, y and z, coloured red, green and blue respectively), or by pressing the `c0c1c2`
button. The c stands for "component", and since the `c =` dropdown is set to PCA, the cluster
window is now showing the first 3 principal components.

8. Notice that green horizontal lines have now appeared on a few of the channels in the sort
window. Those are the channels that were automatically selected for component analysis
(currently PCA). You can toggle a channel by clicking it. You can change the duration of
timepoints used for component analysis using the mouse scrollwheel over the sort window. When
you're done changing channel selection, hit `Enter` to see the result in the cluster window.

9. You should see that the orange cluster (cluster 12) of the two clusters selected above
looks like it needs splitting. Select only cluster 12, and hit `Enter`. Now press the
`Cluster` button in the Cluster tab, or hit `Space` in the cluster window. This will run what
we call gradient ascent clustering (GAC, see http://dx.doi.org/10.3389/fnsys.2014.00006) on
the points, and should split the cluster fairly well into two new clusters (grey points are
classified as outliers and left unsorted). You can always undo a clustering operation by
hitting `Ctrl+Z`, or redo it with `Ctrl+Y`. If it seems the clustering algorithm didn't split
the points up enough, you can decrease the value of sigma in the cluster tab. If it seems the
algorithm oversplit the points into too many clusters, you can increase the value of sigma.
Use `Ctrl` or `Shift` in tandem with the mouse scrollwheel or the mouse right button in the
cluster window to manipulate sigma more conveniently. You'll see that the red/green/blue axes
in the center of the cluster window scale in proportion to the value of sigma. Before running
GAC, you generally want those axes to be roughly the size of the minimum separation distance
between points that you want to split. To change the 3D focal point of the cluster window (and
hence the position of the central axes), hover over a data point and hit the `F` key. To
select and deselect points under the cursor, use the `S` and `D` keys, respectively.

10. You can compare a given selected cluster to its most similar neighbours by selecting it
and using the `>` and `<` keys on the keyboard, or the respective buttons in the sort window
toolbar. For a given selection change (whether clusters or channels or timepoints), press
`Enter` to update the display in the cluster window. The cluster window does not update
automatically because updating it can sometimes be an expensive operation, depending on the
number of spikes, channels, and timepoints selected and the component analysis involved.

11. Repeat the above until you're happy with all the clusters. There are of course many other
details, such as using PCA+ICA instead of just PCA, and exhaustively comparing all
neighbouring clusters with each other, but I'll leave that for another day.

12. Periodically save your results using `Ctrl+S`, or by pressing the save button on the main
window toolbar, or File->Save in the main menu. This will create a trio of files with the same
base name: a `.wave` file that stores the waveforms of all the spikes, a `.spikes` file that
stores other per-spike data, and a `.sort` file which stores more general information,
allowing you to return to exactly where you currently are in your spike sorting session.
Sorting information is split up into these three files purely for the sake of minimizing
saving and loading times of millions of spikes taking up many GBs of space. You should treat
them as an inseperable group. You can open, or reopen, a previous sort session using `Ctrl+O`,
or by pressing the open button on the main window toolbar, or File->Open in the main menu.
Generally, you need not have the original source data file (a `.srf` file in this case) on
hand, because all the relevant spike waveforms are already stored in the `.wave` file. This is
good, because in general the original source file will be at least an order of magnitude
greater in size, and therefore potentially inconvenient to keep on hand.
