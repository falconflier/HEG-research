import h5py
import numpy as np
import glob
import os
import random
import argparse
random.seed(1337)

# directory that merge.py is in
script_dir = os.getcwd() + '/'
#print(script_dir)

spacer = "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Argument parsing
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Argument parsing.  TODO: add a field to make the user able to input TTbar and QCD directory file paths
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--number', type=int, default=0, help='which file to run')
parser.add_argument('-t', '--ttbar_dir', type = str, default = 'test_input/ttbar', help = 'which directory to take ttbar input from')
parser.add_argument('-q', '--qcd_dir', type = str, default = 'test_input/qcd', help = 'which directory to take qcd input from')
# Getting the output path.  Could also use '>' or possibly '>>' in the shell, but I added this for consistency
parser.add_argument('-o', '--output_dir', type = str, default = 'test_output', help = "which directory to put output in.  Defaults to the directory that merge.py is in")
args = parser.parse_args()

# Getting information that the user passed in
nFile = args.number
# Getting the full path to the directories that contain ttbar and qcd files
TTbar_dir = script_dir + args.ttbar_dir
QCD_dir = script_dir + args.qcd_dir
# Setting the output directory based on what the user input
outDir = script_dir + args.output_dir
#outDir = '/home/afriberg/work/CMSSW_10_6_4/src/MLAnalyzer/test_output/merged_files_x1_d1'

# TODO: figure this out
test = False

batch_sz = 32
chunk_sz = batch_sz*500
#if os.path.isfile(output_name):
#    os.remove(output_name)

# First few characters of the output file name
output_name = 'jmar_x1'
gran = 1
ch = 5


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Getting all the files that we want to use, then
# iterating through the lists of files to count the number of events
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Gets a list of all files contained in TTbar_dir
TTbar_files = glob.glob('%s/*'%TTbar_dir)
# Gets a list of all files contained in QCD_dir
QCD_files   = glob.glob('%s/*'%QCD_dir)

# Randomizing the order of the input
random.shuffle(TTbar_files)
random.shuffle(QCD_files)

print(spacer + "\n TTbar files:\n\n")
print(TTbar_files)
print(spacer + "\n QCD files:\n\n")
print(QCD_files)

TTbar_jets = 0
QCD_jets = 0
nevts = 0
TTbar_split = [0]
QCD_split = [0]
for i, jet in enumerate(TTbar_files):
    # The structure of root files is <blablablabla> + '_n' + <number of events (I think)>.root
    # The next line takes the integer after the last '_'
    num_file = int(jet.split('_')[-1][1:-5])
    # Add this value to TTbar_jets and nevts
    TTbar_jets += num_file
    nevts += num_file

    # If the total number of events exceeds the chunk size, start a new counter and record where
    # the split took place
    if nevts >= chunk_sz:
        TTbar_split.append(i+1)
        nevts = 0
# Record the index of the last file and reset the event counter
TTbar_split.append(len(TTbar_files)-1)
nevts = 0

# Do the same thing for QCD files
for i, jet in enumerate(QCD_files):
    QCD_jets += int(jet.split('_')[-1][1:-5])
    nevts += int(jet.split('_')[-1][1:-5])
    if nevts >= chunk_sz:
        QCD_split.append(i+1)
        nevts = 0
QCD_split.append(len(QCD_files)-1)

print(spacer)
print('Number of TTbar Jets:', TTbar_jets)
print('Number of QCD Jets:', QCD_jets)
print('TTbar splits are:', TTbar_split)
print('QCD_splits are:', QCD_split)
print('\n')

# TODO: how do you divide a list by an integer???? (doesn't work on my computer)
file_sz = int(2 * min(TTbar_jets, QCD_jets) / chunk_sz) * chunk_sz
#if test == True:
#    file_sz = 2*chunk_sz

def WriteToFile(f, i, TTbar_start, TTbar_stop, QCD_start, QCD_stop, TTbar_files, QCD_files):
    TTbar_start = TTbar_split[i]
    TTbar_stop = TTbar_split[i+1]
    QCD_start = QCD_split[i]
    QCD_stop = QCD_split[i+1]

    print("File size is: " + str(file_sz) + "\n")
    #quit()

    #nevts = 0
    #for file in TTbar_files[TTbar_start:TTbar_stop]:
    #    jets = int(file.split('_')[-1][1:-5])
    #    nevts += jets
    #    print('number of jets in file', jets)
    #print 'Number of top jets', nevts
    #nevts = 0
    #for file in QCD_files[QCD_start:QCD_stop]:
    #    jets = int(file.split('_')[-1][1:-5])
    #    nevts += jets
    #    print('number of jets in file', jets)
    #print('Number of qcd jets', nevts)

    files = TTbar_files[TTbar_start:TTbar_stop] + QCD_files[QCD_start:QCD_stop]
    print(spacer + "\nFiles are:\n\n" + str(files) + "\n")
    random.shuffle(files)
    print(spacer + "\nShuffled files are:\n\n" + str(files) + "\n")


    dsets = [h5py.File(file, 'r') for file in files]

    idxs = list(range(sum([len(dset['y']) for dset in dsets])))
    random.shuffle(idxs)

    for i, idx in enumerate(idxs):

        # so you only use up the 32000 jets in the file
        if i >= file_sz:
            break

        # need to figure out which file you are taking the event from
        file_num = 0
        tot_jets = 0
        for ii, dset in enumerate(dsets):
            tot_jets += len(dset['y'])
            if idx < tot_jets:
                file_num = ii
                ijet = idx - (tot_jets - len(dset['y']))
                break
        #print('Jet number', idx, 'of 32000')
        #print('In file number', file_num, 'this is jet number', ijet)
        X = np.array(dsets[file_num][input_keys[0]][ijet])

        # For ZS make sure that a track is only suppressed based on pT
        # and that if a track is ZSed, it is removed from all track channels
        X[...,0][X[...,0] < 1.e-3] = 0 # pt zero-suppresion
        X[...,1][X[...,0] < 1.e-3] = 0 # d0 zero-suppresion (based on pt)
        X[...,2][X[...,0] < 1.e-3] = 0 # dz zero-suppresion (based on pt)
        X[...,1][X[...,1] < -10] = 0 # d0 PU removal
        X[...,1][X[...,1] > 10] = 0 # d0 PU removal
        X[...,2][X[...,2] < -20] = 0 # dz PU removal
        X[...,2][X[...,2] > 20] = 0 # dz PU removal
        X[...,3][X[...,3] < 1.e-3] = 0 # ecal zero-suppresion
        X[...,4][X[...,4] < 1.e-3] = 0 # hcal zero-suppresion
        #X[X < 1.e-3] = 0
        #X[-1,...] = 25.*X[-1,...]
        #X[1,...] = X[1,...]*2.
        #X = X/100.
        X[...,0] = X[...,0]/100. #pt
        X[...,1] = X[...,1]/10. #d0
        X[...,2] = X[...,2]/20. #dz
        X[...,3] = (gran**2)*X[...,3]/40. #ECAL - accounting for up sample
        X[...,4] = (gran**2)*X[...,4]/55. #HCAL - accounting for up sample

        pt = dsets[file_num][input_keys[1]][ijet]
        m0 = dsets[file_num][input_keys[2]][ijet]
        y = dsets[file_num][input_keys[3]][ijet]

        if not i % 1000:
            print('Writing jet %d to output file' % i)
            print('label is %d, pT is %d, m0 is %d' % (y, pt, m0))
        f[output_keys[0]][i] = np.copy(X)
        f[output_keys[1]][i] = pt
        f[output_keys[2]][i] = m0
        f[output_keys[3]][i] = y


    for dset in dsets:
        dset.close()
    # End of WriteToFile() function definition


max_per_file = 32000 * 1
total_jets = 0

f = h5py.File('%s/%s_file-%d.hdf5'%(outDir,output_name,nFile), 'w')


input_keys = ['X_jets', 'jetPt', 'jetM', 'y']
output_keys = ['X_jets', 'pt', 'm0', 'y']

file_sz = min(file_sz, max_per_file)

files = TTbar_files[TTbar_split[nFile]:TTbar_split[nFile+1]] + QCD_files[QCD_split[nFile]:QCD_split[nFile+1]]
print(spacer + "\nfiles: \n\n" + str(files))
file_sz = 0
for file in files:
    file_sz += int(file.split('_')[-1][1:-5])
f.create_dataset(output_keys[0], (file_sz, 125*gran, 125*gran, ch), chunks = (batch_sz, 125*gran, 125*gran, ch), compression='lzf')
for key in output_keys[1:]:
    f.create_dataset(key, (file_sz, ), chunks = (batch_sz, ), compression='lzf')


for i in range( min(len(TTbar_split), len(QCD_split)) - 1 ):
    if nFile != i:
        print('skipping', i)
        continue
    else:
        print('Proessing file', i)
    WriteToFile(f, i, TTbar_split[i], TTbar_split[i+1], QCD_split[i], QCD_split[i+1], TTbar_files, QCD_files)
    print('Succesfully wrote chunk %d to file' % (i))

f.close()
