# HSW-V Toolkit

A graphical interface for computing time series of zonal-average variables of latitude bands in the HSW-V limited variability datasets, and visualizing the results

![demo_screenshot](demo.png)

# Getting Started

1. Clone the repository
   
`clone https://github.com/jhollowed/hswpp_toolkit.git`
   
2. Launch the application

```
cd hswpp_toolkit
python ./interface.py
```

3. Select options

   Set the desired configuration in the various options above the Results table. This includes:

   - the Data Release (correspondoing to different HSW-V ensembles)
   - the Dataset (corresponding to different ensemble members, or the ensemble mean, of the chose Data Release)
   - the SO2 Magnitude (corresponding to ensembles in this Data Release of varying eruption magnitudes)
   - the Latitude Bands (defines the northern and southern boundaries of four latitude bands)
   - the Anomaly Base (variable anomalies will be defined with respect to this dataset)
   - the Anomaly Definition (variable anomalies will be defined by this criteria
   - the Pressure Level for Tracers (the three-dimensional tracer fields `SO2` and `SULFATE` will be reduced to a two-dimensional horizontal field at the vertical level nearest this pressure)
  
     If any of these options are disabled (faint text, unclickable), then they are not currently available in this release of the Toolkit, or the datasets for those selections do not exist.
  
4. Press *refresh results table*

5. repeat steps 3 and 4 as needed

If the chosen Data Release has not yet been run with the current installation of HSW-V Toolkit when *refresh results table* is pressed, it will be downloaded from FigShare. When this occurs, the *refresh results table* button will be altered to read *fetching data*. The download will take several minutes. Progress will be represented in the progress bar to the right of the button, as well as more detailed progress printed to the terminal.
