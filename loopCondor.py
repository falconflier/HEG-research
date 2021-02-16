# Added Feb 16, 2021 to check that this was the file being called
#print("Hello there, you called me?")
#!/usr/bin/env python
import os, sys
import glob

import argparse
parser = argparse.ArgumentParser(description='Tell what label you are using')
parser.add_argument('-l', '--label', required=True, type=int, help='0 is QCD, 1 is TTbar')
parser.add_argument('-j', '--nJets', type=int, default=1, help='Number of jets in the root file')
parser.add_argument('-g', '--gran', type=int, default=1, help='Increased track and pixel granularity')
parser.add_argument('--start', type=int, default=0, help='Which file to start on')
parser.add_argument('--end', type=int, default=99999, help='Which file to end on')
args=parser.parse_args()

#ttbar_files = len(glob.glob('/home/npervan/e2e/jmar_aod_ntuples/*'))
ttbar_files = len(glob.glob('/home/npervan/e2e/TTbar_AF/*'))
qcd_files = len(glob.glob('/home/npervan/e2e/qcd_aod_ntuples/*'))

mc = ''
njets = args.nJets


assert args.label == 1 or args.label == 0
if args.label == 0:
        mc = 'QCD'
        files = qcd_files
elif args.label == 1:
        mc = 'TTbar'
        files = ttbar_files
def condor(start, stop):
        basedir = os.getcwd()
        if not os.path.exists(basedir+'/condor/%s'%mc):
                os.makedirs(basedir+'/condor/%s'%mc)
                os.makedirs(basedir+'/condor/%s/log_out'%mc)
        for i in range(start, stop):
                dict={'file':i,'dir':basedir,'mc':mc,'label':args.label, 'jets':njets, 'gran':args.gran}
                filename = '%(dir)s/condor/%(mc)s/condor_%(jets)sjets_%(file)s_x%(gran)s.job' % dict
                jdf = open(filename, 'w')
                jdf.write("""
universe = vanilla
executable = %(dir)s/loopdoConvert.sh
Should_Transfer_Files = YES
WhenToTransferOutput = ON_EXIT
Notification = Error

Arguments = %(file)s %(label)s %(jets)s %(gran)s

Output = %(dir)s/condor/%(mc)s/log_out/log_%(file)s_x%(gran)s.stdout
Error = %(dir)s/condor/%(mc)s/log_out/log_%(file)s_x%(gran)s.stderr
Log = %(dir)s/condor/%(mc)s/log_out/log_%(file)s_x%(gran)s.condorlog

Queue 1
                """%dict)

                jdf.close()
                os.system('condor_submit %s' % filename)

pass
condor(args.start, args.end)

