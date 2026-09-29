# loads DEM tiles (and merges into chunks) from directory of small DEM scenes.
# feeds into model deployer and handles result internally. Saves output of each subchunk (size depending on system RAM etc) as it goes.
# at the end, combines all into one final .shp file. 
# need to set up kwargs here for sure.

class SCDModel():

    def __init__(self, model, veto, dem_dir, device, resolutions, **kwargs):
        ...

        
