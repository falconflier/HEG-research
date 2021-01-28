#!/bin/bash

File=$1
Label=$2
Jets=$3
Granularity=$4

#Start=3
#End=4
#Label=0
#Jets=1

CurrentDir='/home/npervan/e2e/CMSSW_9_3_0/src/MLAnalyzer'
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=slc7_amd64_gcc630
cd /home/npervan/e2e/CMSSW_9_3_0/src/MLAnalyzer
eval `scramv1 runtime -sh`
cd $CurrentDir

python ${CurrentDir}/hdf5_conversion_loop.py -n $File -l $Label -j $Jets -g $Granularity

