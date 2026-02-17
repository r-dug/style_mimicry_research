# This script will apply countermeasures to attempt removing signal from perturbation based defenses
# It will apply transformations to each sample of protected artwork in each of the subdirectories of: ../../data/style_mimicry/protected_art 
#   corresponding to the < protection method used / category of artist {historical or contemporary} / artist > 
# It will save the transformed images in a corresponding subdirectory of ../../data/style_mimicry/robust_samples
#   to the countermeasure applied
# It will also save an error log of the transformations not successfully applied to in a corresponding subdirectory of ../../data/style_mimicry/robust_samples/logs
#   to the countermeasure applied

