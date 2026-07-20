# post processing

# want a neural network based off spectral bands (just RGB)
# this will only serve to reject/deny predictions: idea being it will see anthropogenic structures, etc and know they arent SCDs
# therefore increasing precision!
# this will not affect recall, the model won't be able to see new candidates, just already predicted ones, with the aim of screening 
# bad predictions out

# also: shape based screening -> angular predictions with one straight side (edge artefact) might be removable?
