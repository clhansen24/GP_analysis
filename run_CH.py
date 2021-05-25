'''
Runs the pipeline 
'''

import sys, os

#All the input csv files for analysis should be in the file named Input 
InputPath = 'Input'

#For each input file, a results folder with the same name as the input file can be found in Output 
os.mkdir("Output")
os.mkdir("StraightenedAggregates_projections")

#Passing all the files in Input through the pipeline 
for filename in os.listdir(InputPath):
    if (filename == '.DS_Store'):
        continue

    #Printing out the name of the file 
    os.system('echo && echo && echo'); os.system('echo Input/%s' % filename); os.system('echo')

    #Creating an Output file with the same name as the input file so results have a destination 
    os.chdir('Output')
    outputFile = filename[:-4]
    os.mkdir('%s' % outputFile)
    os.chdir('..')
    
    #Splitting channels and visualizing the aggregate 
    os.system('/bin/python3.6 split_channels_CH.py Input/%s' % filename)
    os.system('/bin/python3.6 scatter_plot.py')

    
    #Clustering
    #os.system('/bin/python3.6 clustering.py C1.csv ClusteredC1.csv')
    #os.system('/bin/python3.6 clustering.py C2.csv ClusteredC2.csv')
    #os.system('/bin/python3.6 clustered_plots.py')
    

    #CortexRemoval
    #os.system('/bin/python3.6 cortex_removal.py')
    #os.system('/bin/python3.6 scatter_plot_cortex_removed.py')

    #Orienting aggregate
    #os.system('/bin/python3.6 Orientation.py')
    
    #Straightening
    os.system('Rscript principal_curve_CH.R')
    os.system('/bin/python3.6 straightening_CH.py')

    #PCA and 2D plots for each RNP type
    os.system('/bin/python3.6 2D_projections.py StraightenedC1.csv ComponentsC1.csv r Output/%s/PrincipalComponentsC1.png' % outputFile)
    os.system('/bin/python3.6 2D_projections.py StraightenedC2.csv ComponentsC2.csv g Output/%s/PrincipalComponentsC2.png' % outputFile)

    #Removing outliers
    os.system('/bin/python3.6 outliers.py')

    #Curve Fitting
    os.system('/bin/python3.6 curve_fitting_components.py')

#Formatting the data collected
os.system('/bin/python3.6 data_formatting.py')









    
