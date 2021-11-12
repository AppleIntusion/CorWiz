Summary = \
'         This script generates heatmap and '        + \
'         feature-space correlation plots for '      + \
'         selected regoins of a structure. All '     + \
'         arguments below are required. '            + \
'         Documentation is super '                   + \
'         thourough so if you are curious about '    + \
'         how anything works just call the '         + \
'         `help()` function on the imported '        + \
'         module or read the source code.  '     

Description = \
    "Dependencies:\n\n"                           + \
    "\tThere are quite a few dependencies. You \n"+ \
    "\twill need `pytraj matplotlib pandas \n"    + \
    "\tnumpy seaborn os sys sklearn`. I\n"        + \
    "\treccomend using a conda environment. \n"   + \
    "\tDon't have a .yml file but if you have \n" + \
    "\tconda installed and issue the following \n"+ \
    "\tcommands you'll be just fine.\n"           + \
    "\t    $ conda create --name analysis \n"     + \
    "\t      python=3.6 matplotlib pandas \n"     + \
    "\t      numpy seaborn os sys sklearn\n"      + \
    "\t    $ conda activate analysis\n"           + \
    "\t    $ conda install -c ambermd pytraj\n"   + \
    "\tsubsequently you will only need to issue\n"+ \
    "\t`conda activate analysis` before\n"        + \
    "\trunning the script.\n\n"                   + \
    "NOTES: If you are using this, as a \n"       + \
    "       module in your own script or for \n"  + \
    "       its intended functionality, you \n"   + \
    "       are likely on the Li lab's GPU \n"    + \
    "       machine, Gullveig and you are \n"     + \
    "       likely using a conda environment.\n"  + \
    "       In that case, be sure to run \n"      + \
    "       `unset PYTHONPATH`. Otherwise \n"     + \
    "       pytraj will not import and you \n"    + \
    "       will be very sad.\n" 

traj_help = 'Trajectory file name. Ex) traj.nc'
top_help = 'Topology file name. Ex) top.top'
region_help = 'Region selections. Ex) ":1-3@CA :4-7@CA :9-10@CA"'
reorder_help = 'Used to reorder correlation matrix. Just keep in original order if you do not want anything reordered. Ex) "0 1 2"'
names_help = 'Group names. Names of each of the above groups. Should contain two parts seperated by a period. The first half is used for ' +\
             'assigning color in the feature-space plot and the second half will appear next to each point on the plot. '  +\
             'Ex) "DI.RI DI.RII DII.RI" in this example the first two selections will be the same color.'
cut_help =   "Correlation line cutoff. Lines will be drawn between points that have an absolute correlation above this value. Ex) 0.6."
saveh_help = "Name of the heatmap output file. Ex) heatmap.png"
savef_help = "Name of the feature-space output file. Ex) feature.png"
start_help = "The frame to begin processing at. Ex) 5000"
end_help =   "The frame to end processing at. Ex) 500000"
stride_help = "The number of frames to skip while reading. Ex) 10"
