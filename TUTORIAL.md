### Basic tutorial

1. Run the program from the command line by typing `spyke`. Status messages tend to be printed
back to the command line, so it's a good idea to keep an eye on it.

2. Open a data file (typically a `.dat`, `.ns6` or `.srf` file), such as this [sample
file](http://swindale.ecc.ubc.ca/spyke?action=AttachFile&do=get&target=ptc15_tr7c_r87_spont_20sec.srf).

3. By default, only the spike window opens up. This shows about 1 or 2 ms of high-pass voltage
data on all channels at a time, with waveforms arranged according to the channel layout on the
polytrode. For a broader view, press the chart window button in the toolbar, or enable it in
the View menu. The chart window displays 50 ms of high-pass voltage data at a time, in a
purely vertical layout. Now press the LFP window button in the toolbar, or enable it in the
View menu. The LFP window shows 1 s of low-pass voltage data at a time. Channel colours and
vertical positions correspond across all three voltage waveform windows. You can browse the
data in several ways: using the slider widget in the main window; entering a time point in
microseconds in the edit box near the middle top; clicking the start and end buttons on either
side of the edit box; or clicking on the timepoint you want to center on in any of the voltage
data windows. You can also scale voltage and time with `Ctrl+Scrollwheel` and
`Shift+Scrollwheel` respectively. Now, toggle all 3 voltage data windows in the toolbar to
hide them and tidy up the GUI before we go and detect some spikes. Note that even when the
data windows are hidden, the data file remains open.

4. Press the `Detect` button in the Detect tab. This will run spike detection on the entire
file, given the current detection settings. As an alternative to spyke's built-in detection,
you can use the output of KiloSort (see section below).

5. When detection is done, three new windows will open up: sort, cluster and matplotlib. The
sort window shows detected waveforms and cluster means, the cluster window shows each spike as
a point in some 3D cluster space, and the matplotlib window plots various other things as
appropriate. You can manipulate the view in the cluster window using the mouse, rotating,
zooming, and panning. Try using the different mouse buttons and scroll wheel, in tandem with
`Ctrl` and `Shift`. You can select detected unsorted events in the lower list in the sort
window. `Ctrl` and `Shift` allow for multiple selections.

6. Deselect any selected spikes (`Esc` and `E` are both good for this). Click the `/` button
in the sort window toolbar, or just press the `/` key. This will roughly divide the spikes up
into clusters based on their maximum channel. The neuron list in the sort window (upper list)
is now populated, and the points in the cluster window are now coloured. You can select any
number of units that you like, and their mean waveforms will be plotted. You can also select
specific spikes of the selected neurons in the middle list in the sort window. The bottom list
will show unsorted spikes (currently none). Pressing the `R` toolbar button or keyboard key
will randomly select a subset of spikes from the selected neurons. The slider widget just
below the sort window toolbar provides a sliding selection time window of points. Most of the
units here have very few spikes, since it's such a short example file, so they might not look
very impressive in the cluster window. You'll probably have to select at least a few clusters
to display a decent number of spikes.

7. Hit `Esc` or `E` again (possibly twice) to clear any selections. Now press the renumber
button `#` in the sort window toolbar or on the keyboard. This will renumber all clusters in
vertical spatial order. Hit `Yes`.

8. Assuming you're working on the sample file linked to above, select clusters 11 and 12 in
the sort window (or any few clusters that have some channel overlap). Once selected, hit
`Enter` or `A`, and they're plotted in the cluster window. Back in the main window, in the
Cluster tab, there's a section labelled Plotting, where you can control the dimensions plotted
in the cluster window. It defaults to x0, y0 and Vpp for the 3 dimensions. x0 and y0 are the
estimated positions of each spike along the polytrode, and Vpp is the peak to peak voltage on
the maximum channel. You can change these dimensions to c0, c1, c2 via the dropdown lists
(labelled x, y and z, coloured red, green and blue respectively), or by pressing the `c0c1c2`
button. The c stands for "component", and since the `c =` dropdown is set to PCA, the cluster
window is now showing the first 3 principal components.

9. Notice that green horizontal lines have now appeared on a few of the channels in the sort
window. Those are the channels that were automatically selected for component analysis
(currently PCA). You can toggle a channel by clicking it. You can change the duration of
timepoints used for component analysis using the mouse scroll wheel over the sort window. When
you're done changing channel and timepoint selection, hit `Enter` or `A` to see the result in
the cluster window. Channel and timepoint selection can play a major role in the
clusterability of the dimension-reduced data, so it's good to quickly experiment a bit for
each cluster and/or combination of clusters, and only include channels and timepoint ranges
that encompass meaningful spike data (see section  3 of this
[thesis](http://mspacek.github.io/mspacek_thesis.pdf)).

10. You should see that the orange cluster (cluster 12) of the two clusters selected above
looks like it needs splitting. Select only cluster 12, and hit `Enter` or `A`. Now press the
`Cluster` button in the Cluster tab, or hit `Space` in the cluster window. This will run what
we call gradient ascent clustering (GAC, see [Spike sorting for polytrodes: a divide and
conquer approach](http://dx.doi.org/10.3389/fnsys.2014.00006)) on the points, and should split
the cluster fairly well into two new clusters (grey points are classified as outliers and left
unsorted). You can always undo a clustering operation by hitting `Ctrl+Z`, or redo it with
`Ctrl+Y`. If it seems the clustering algorithm didn't split the points up enough, you can
decrease the value of sigma in the cluster tab. If it seems the algorithm oversplit the points
into too many clusters, you can increase the value of sigma. Use `Ctrl` or `Shift` in tandem
with the mouse scroll wheel or the mouse right button in the cluster window to manipulate
sigma more conveniently. You'll see that the red/green/blue axes in the center of the cluster
window scale in proportion to the value of sigma. Before running GAC, you generally want those
axes to be roughly the size of the minimum separation distance between points that you want to
split. To change the 3D focal point of the cluster window (and hence the position of the
central axes), hover over a data point and hit the `F` key. To select and deselect points
under the cursor, use the `S` and `D` keys, respectively. You can do the above while moving
the mouse, allowing you to "paint" a selection.

11. You can compare a given selected cluster to its most similar neighbours by selecting it
and using the `>` and `<` keys on the keyboard, or the respective buttons in the sort window
toolbar. For a given selection change (whether clusters or channels or timepoints), press
`Enter` or `A` to update the display in the cluster window. The cluster window does not update
automatically because updating it can sometimes be an expensive operation, depending on the
number of spikes, channels, and timepoints selected and the component analysis involved.

12. Repeat the above until you're happy with all the clusters. There are of course many other
details, such as using ICA instead of just PCA, and exhaustively comparing all neighbouring
clusters with each other. See chapter 3 of this
[thesis](http://mspacek.github.io/mspacek_thesis.pdf) for all of the details.

13. Periodically save your results using `Ctrl+S`, or by pressing the save button on the main
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
good, because in general the original source file will be much bigger, and therefore
potentially inconvenient to keep on hand.

### Sorting full tracks/series with [KiloSort](https://github.com/cortex-lab/KiloSort)

spyke has the concept of a "track": a set of recordings in multiple files that were all
recorded in the same position in the brain, for which spike sorting should be done all
together. This is completely synonymous with the alternative term "series". spyke can work
fairly seamlessly with KiloSort to provide a better initial set of clusters, which can then be
refined as usual within spyke. To sort a full track with spyke and KiloSort:

1. File->New Track. Navigate to your folder with a track/series worth of source files (`.ns6`
or `.dat` or `.srf`), type in the desired track/series name and hit Save to create a `.track`
file. If you're using `.ns6` source files that don't already have at least one `.json`
metadata file describing their probe and adapter type, and if the probe and adapter type are
anything other than `A1x32` and `null`, then you need to create a `.json` metadata file that
names the probe and adapter used in that recording. Otherwise, spyke will (hopefully) raise an
error. For example `.json` files, see `templates/json` in the spyke code directory.

2. The track file is automatically opened. Open up the chart and LFP windows (View menu or
toolbar) for a better look at the data. You can scroll through the track by clicking, pressing
`LEFT`/`RIGHT` or `PGUP`/`PGDOWN`, or by scrolling with the mouse wheel. You might notice some
blank periods, representing the time in between recordings in the track. You can play around
with Filtering, CAR (common average reference) and Sampling settings in the menu (although
these are all ignored during raw data export). Notice how CAR can make quite a difference in
the noise level. If you use CAR->Median (the default), the noise is reduced without really
changing any spike amplitudes. Disable any channels you think are faulty by right-clicking on
them. If you right-click on a channel again, it's re-enabled.

3. File->Save Track Channels to save any channel selections you may have made. These
selections are saved to the `.track` file, which you can inspect with a plain text editor.

4. File->Export->Raw Data->.dat & KiloSort files. Choose the desired destination folder
(probably a local folder, on which you will locally run KiloSort in MATLAB) and hit Open. This
exports a concatenated `.dat` file from all your source files in the current `.track` file, as
well as the required MATLAB files to run KiloSort. Note that the Filtering, CAR and Sampling
settings you are currently using in spyke are ignored during export. Only raw unprocessed data
is exported. KiloSort will then do its own preprocessing on the raw data.

5. Open the `_ks_run.m` file that was created in the above step. Check the first two lines to
make sure that the path to wherever you installed KiloSort and
[npy-matlab](https://github.com/kwikteam/npy-matlab) is correct. If it isn't correct, you can
fix it there, or better yet, fix it permanently for yourself in the
`templates/kilosort/ks_run.m` template file in the spyke code directory.

6. Start MATLAB, `cd` to the folder where you exported the raw data to, and run the
`_ks_run.m` file. KiloSort automatically does its own filtering and CAR, but it doesn't
resample.

7. When KiloSort is done, with the `.track` still open in spyke, go File->Convert KiloSort
.npy to events.zip. Select the `ks_results` folder generated by KiloSort. This should be a
subfolder of where you exported the raw data to.

8. File->Open the `.events.zip` file you just created. This will import the results from
KiloSort into spyke. This will probably take quite a while as it extracts the waveforms from
the raw data and performs spatial localization on each spike.

9. When it's finally done, go File->Save to save it to a `.sort` file, spyke's native format.
Now you can proceed as usual (starting at step 7 in the basic tutorial above), inspecting and
refining the set of clusters generated by KiloSort.
