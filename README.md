# DTS_Processing
Scripts for handling DTS data - generally and for specific purposes
90% of this has aggressive hardcording for my personal computer.

* BLM_ProcessingClean.py
 -> This contains the processing for the BLM paper, taking mapped DTS dictionaries and determining integral and wavenumber-dependent convective velocities
 
* DTSHelperFunctions.py
 -> Collection of functions to help with raw data file conversion/management
 -> Depended on in other scripts
 
* DTSProcessing.py
 -> Mostly initial processing pipeline: reformating the raw data files, mapping them, calibrating temperatures, etc.
 
* fileProofer.py
 -> Checks for errors in the processed files from BLM_ProcessingClean
 
* autoProg.py
 -> Periodically runs progressCheck so you can passively monitor
 
* progressCheck.py
 -> Checks a given set of folders for processing progress
