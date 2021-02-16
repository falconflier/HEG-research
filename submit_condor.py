import sys, os
import glob
import time
import subprocess

label = 1
jets = 1
# Changed on Jan 26, 2021
#jets = 2
gran = 1

if label == 0:
    if jets == 1:
        dirs = ['/home/npervan/e2e/qcd_aod_ntuples/']
    elif jets == 2:
        dirs = []
elif label == 1:
    if jets == 1:
#        dirs = ['/home/npervan/e2e/jmar_aod_ntuples/']
        dirs = ['/home/npervan/e2e/TTbar_AF']
    elif jets == 2:
        dirs = []
else:
    print 'did not put right inputs'
    quit()

def submit(start, end):
    print 'Submitting new set of condor jobs'
    os.system('python loopCondor.py -l %d -g %d -j %d --start %d --end %d' % (label, gran, jets, start, end))

def check_progress():
    progress = subprocess.check_output(['condor_q'])
    progress = progress.splitlines()
    return len(progress) == 9

def main():
    if not check_progress():
         choice = raw_input('Warning, currently have condor jobs running. Do you wish to continue? (Y/N)\n')
         if choice.lower() == 'y':
           print 'Will run script anyways'
         else:
            print 'Exiting script'
            return 0
    nFiles = 0
    for directory in dirs:
        nFiles += len(glob.glob('%s/*' % directory))
    idxs = list(range(0,nFiles, 150))
    print 'Parameters are:'
    print 'Use x%d granularity' % gran
    print 'Process on label', label
    print 'Process jet number', jets
    print 'will submit following number of jobs', nFiles

    time.sleep(10)

    for idx in idxs:
        while not check_progress():
            print 'Condor jobs still running'
            time.sleep(120)
        submit(idx, idx+150)
        time.sleep(10)
            

if __name__ == '__main__':
    main()

