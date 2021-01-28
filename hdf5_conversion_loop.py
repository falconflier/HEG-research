import glob, os
import ROOT
import numpy as np
import h5py
from skimage.measure import block_reduce
from numpy.lib.stride_tricks import as_strided
import itertools


# TODO make it so we can read a file where each event has either 1 OR 2 jets, and process all jets

import argparse
parser = argparse.ArgumentParser(add_help=True, description='Process some integers.')
parser.add_argument('-l', '--label', required=True, type=int, help='Decay label.')
parser.add_argument('-n', '--fileNum', type=int, required=True, help='Choose which file you are processing')
parser.add_argument('-j', '--doJet', type=int, required=True, help='How many are in your root file. If 0, will do all jets. If 1 or 2, will only convert 1st or 2nd jet')
parser.add_argument('-g', '--granularity', default=1, type=int, help='Increased Image Pixel Granularity')
parser.add_argument('-i', '--input', type = str, default = '/home/npervan/e2e/TTbar_AF', help = 'Choose which directory to take input from')
parser.add_argument('-o', '--output', type = str, default = 'conversion_loop_output', help = 'Choose which directory to take output from')
args = parser.parse_args()

# Specifying the input and output directories using the argparse arguments
inputDir = args.input
outDir = args.output

def upsample_array(x, b0, b1):

    r, c = x.shape                                    # number of rows/columns
    rs, cs = x.strides                                # row/column strides
    x = as_strided(x, (r, b0, c, b1), (rs, 0, cs, 0)) # view as a larger 4D array

    return x.reshape(r*b0, c*b1)/(b0*b1)              # create new 2D array with same total occupancy

def resample_EE(imgECAL, factor=2):

    # EE-
    imgEEm = imgECAL[:140-85] # EE- in the first 55 rows
    imgEEm = np.pad(imgEEm, ((1,0),(0,0)), 'constant', constant_values=0) # for even downsampling, zero pad 55 -> 56
    imgEEm_dn = block_reduce(imgEEm, block_size=(factor, factor), func=np.sum) # downsample by summing over [factor, factor] window
    imgEEm_dn_up = upsample_array(imgEEm_dn, factor, factor)/(factor*factor) # upsample will use same values so need to correct scale by factor**2
    imgECAL[:140-85] = imgEEm_dn_up[1:] ## replace the old EE- rows

    # EE+
    imgEEp = imgECAL[140+85:] # EE+ in the last 55 rows
    imgEEp = np.pad(imgEEp, ((0,1),(0,0)), 'constant', constant_values=0) # for even downsampling, zero pad 55 -> 56
    imgEEp_dn = block_reduce(imgEEp, block_size=(factor, factor), func=np.sum) # downsample by summing over [factor, factor] window
    imgEEp_dn_up = upsample_array(imgEEp_dn, factor, factor)/(factor*factor) # upsample will use same values so need to correct scale by factor*factor
    imgECAL[140+85:] = imgEEp_dn_up[:-1] # replace the old EE+ rows

    return imgECAL

def crop_jet(imgECAL, iphi, ieta, gran, jet_shape=125):

    # NOTE: jet_shape here should correspond to the one used in RHAnalyzer
    off = jet_shape//2
    iphi = int(iphi*5 + 2)#*gran + gran//2 # 5 EB xtals per HB tower
    ieta = int(ieta*5 + 2)#*gran + gran//2 # 5 EB xtals per HB tower

    # Wrap-around on left side
    if iphi < off:
        phi_diff = off-iphi
        #print('phi low')
        #if eta above or below
        if ieta < off:
            #print('eta low: ', ieta)
            eta_diff = off-ieta
            img_crop = np.concatenate((imgECAL[0:ieta+off+1,-phi_diff:],
                                   imgECAL[0:ieta+off+1,:iphi+off+1]), axis=-1) #Set lower eta edge to 0 then pad
            img_crop = np.pad(img_crop,((eta_diff,0),(0,0)), 'constant')

        elif 280-ieta < off:
            #print('eta high: ', ieta)
            eta_diff = off - (280-ieta)
            img_crop = np.concatenate((imgECAL[ieta-off:280,-phi_diff:],
                                   imgECAL[ieta-off:280,:iphi+off+1]), axis=-1) #Set upper eta edge to 280 then pad
            img_crop = np.pad(img_crop,((0,eta_diff+1),(0,0)), 'constant')

        else:
            #print('eta okay')
            img_crop = np.concatenate((imgECAL[ieta-off:ieta+off+1,-phi_diff:],
                                   imgECAL[ieta-off:ieta+off+1,:iphi+off+1]), axis=-1)

    # Wrap-around on right side
    elif 360*gran-iphi < off:
        #print('phi high')
        phi_diff = off - (360-iphi)
        #if eta above or below
        if ieta < off:
            #print('eta low: ', ieta)
            eta_diff = off-ieta
            img_crop = np.concatenate((imgECAL[0:ieta+off+1,iphi-off:],
                                       imgECAL[0:ieta+off+1,:phi_diff+1]), axis=-1) #Set lower eta edge to 0 then pad
            img_crop = np.pad(img_crop,((eta_diff,0),(0,0)), 'constant')

        elif 280-ieta < off:
            #print('eta high: ', ieta)
            eta_diff = off - (280-ieta)
            img_crop = np.concatenate((imgECAL[ieta-off:280,iphi-off:],
                                       imgECAL[ieta-off:280,:phi_diff+1]), axis=-1) #Set upper eta edge to 280 then pad
            img_crop = np.pad(img_crop,((0,eta_diff+1),(0,0)), 'constant')

        else:
            #print('eta okay')
            #print(ieta-off, ieta+off+1, iphi-off, phi_diff+1)
            #print(type(imgECAL))
            #print(imgECAL[ieta-off:ieta+off+1,iphi-off:])
            #print(imgECAL[ieta-off:ieta+off+1,:phi_diff+1])
            img_crop = np.concatenate((imgECAL[ieta-off:ieta+off+1,iphi-off:],
                                       imgECAL[ieta-off:ieta+off+1,:phi_diff+1]), axis=-1)


    # Nominal case
    else:
        #print('phi okay')
        #if eta above or below
        if ieta < off:
            #print('eta low: ', ieta)
            eta_diff = off-ieta
            img_crop = imgECAL[0:ieta+off+1,iphi-off:iphi+off+1]
            img_crop = np.pad(img_crop,((eta_diff,0),(0,0)), 'constant')

        elif 280-ieta < off:
            #print('eta high: ', ieta)
            eta_diff = off - (280-ieta)
            img_crop = imgECAL[ieta-off:280,iphi-off:iphi+off+1]
            img_crop = np.pad(img_crop,((0,eta_diff+1),(0,0)), 'constant')

        else:
            #print('eta okay')
            img_crop = imgECAL[ieta-off:ieta+off+1,iphi-off:iphi+off+1]

    return img_crop

xrootd='root://cmsxrootd.fnal.gov' # FNAL

decays = ['QCD', 'TTbar']

scale = [1., 1.]
jet_shape = 125 * args.granularity
doJet = args.doJet
width = 280*args.granularity
height = 360*args.granularity


if doJet == 1:
    #ttbar_files = glob.glob('%s/jmar_aod_ntuples/*' % inputDir)
    #qcd_files = glob.glob('%s/qcd_aod_ntuples/*' % inputDir)
    ttbar_files = glob.glob('%s/*' % inputDir)
    #ttbar_files = glob.glob('%s/jmar_aod_ntuples/*' % inputDir)
    qcd_files = glob.glob('%s/qcd_aod_ntuples/*' % inputDir)
elif doJet == 2:
    ttbar_files = glob.glob('%s/ttbar_2jets/*' % inputDir)
    qcd_300to600 = glob.glob('%s/qcd_300to600_2jets/*' % inputDir)
    qcd_400to600 = glob.glob('%s/qcd_400to600_2jets/*' % inputDir)
    qcd_600to3000 = glob.glob('%s/qcd_600to3000_2jets/*' % inputDir)
elif doJet == 0:
    ttbar_files = glob.glob('%s/*' % inputDir)

#qcd_files = qcd_300to600 + qcd_400to600 + qcd_600to3000

TEST = True
test = 'test_output/af_multijet_test_x%d.hdf5' % args.granularity

# Loop over decays
for d, decay in enumerate(decays):

    if d != args.label:
        continue

    if d == 0:
        filelist = qcd_files
        tfile_idxs = range(1, len(qcd_files)+1)
    elif d == 1:
        filelist = ttbar_files
        tfile_idxs = range(1, len(ttbar_files)+1)
    else:
        print 'decay must be equal to 0 or 1'
        break

    print '>> Doing decay[%d]: %s'%(d, decay)

    # Get root tree
    tfile_str = str(filelist[args.fileNum])
    print " >> For input file:", tfile_str
    #tfile = ROOT.TXNetFile(tfile_str)
    tfile = ROOT.TFile(tfile_str)
    tree = tfile.Get('fevt/RHTree')
    nevts = tree.GetEntries()
    tree.SetBranchStatus("*",0)

    if TEST:
        nevts = 100

    njets = 0
    tree.SetBranchStatus("jetPt",1)
    for iEvt in range(0, nevts):
        tree.GetEntry(iEvt)
        njets += len(tree.jetPt)

    print " >> Total events:", nevts
    print " >> Total jets:", njets
    tree.SetBranchStatus("*",1)

    outPath = '%s/%s_IMGjet_%d-Granularity'%(outDir, decay, args.granularity)
    # note, if mutliple jobs are submitted to condor and multiple try to create the directory simulataneously, a lot of those jobs will fail
    if not os.path.isdir(outPath):
        os.makedirs(outPath)
    fout_str = '%s/%s_f%d_j%d_n%d.hdf5' % (outPath, decay, args.fileNum, doJet, nevts)
    if TEST:
        fout_str = test

    # create output file and define it's datasets
    fout = h5py.File(fout_str, 'w') #Changing all datasets from nevts to njets
    fout.create_dataset('X_jets', (njets, jet_shape, jet_shape, 5), compression='lzf') # note, number of image channels was hardcoded here
    fout.create_dataset('jetPt', (njets, ), compression='lzf')
    fout.create_dataset('jetM', (njets, ), compression='lzf')
    fout.create_dataset('y', (njets, ), compression='lzf')

    # define branch names based on whether or not you are using high granularity images
    if args.granularity == 1:
        br_pt = 'ECAL_tracksPt_atECALfixIP'
        br_d0 = 'ECAL_tracksD0_atECALfixIP' #Removed Sig
        br_dz = 'ECAL_tracksDz_atECALfixIP' #Removed Sig
        br_ecal = 'ECAL_energy'
        br_hcal = 'HBHE_energy'
        br_pix1 = 'BPIX_layer1_ECAL_atPV'
        br_pix2 = 'BPIX_layer2_ECAL_atPV'
        br_pix3 = 'BPIX_layer3_ECAL_atPV'
    else:
        br_pt = 'ECALadj_tracksPt_%dx%d'%(args.granularity, args.granularity)
        br_d0 = 'ECALadj_tracksD0_%dx%d'%(args.granularity,args.granularity) #Removed Sig
        br_dz = 'ECALadj_tracksDz_%dx%d'%(args.granularity,args.granularity) #Removed Sig
        br_ecal = 'ECAL_energy'
        br_hcal = 'HBHE_energy'
        br_pix1 = 'BPIX_layer1_ECALadj_%dx%d'%(args.granularity,args.granularity)
        br_pix2 = 'BPIX_layer2_ECALadj_%dx%d'%(args.granularity,args.granularity)
        br_pix3 = 'BPIX_layer3_ECALadj_%dx%d'%(args.granularity,args.granularity)

    # convert to hdf5 file using an event loop
    jet_count = 0
    for iEvt in range(0, nevts):

        if not iEvt % 100:
            print 'Processing event', iEvt
        #if TEST:
        #    print iEvt

        # initialize tree
        tree.GetEntry(iEvt)

        # pyroot makes you normally do TreeName.BranchName
        # however, BranchName is a variable in our case, so we use
        # the getattr method
        TracksPt = np.array(getattr(tree, br_pt)).reshape(width, height)
        TracksD0 = np.array(getattr(tree, br_d0)).reshape(width, height)
        TracksDz = np.array(getattr(tree, br_dz)).reshape(width, height)
        Ecal = np.array(getattr(tree, br_ecal)).reshape(280,360)
        Ecal = resample_EE(Ecal)
        if args.granularity != 1:
            Ecal = upsample_array(Ecal, args.granularity, args.granularity)
        Hcal = np.array(getattr(tree, br_hcal)).reshape(56,72)
        Hcal = upsample_array(Hcal, 5*args.granularity, 5*args.granularity)
        #print(np.shape(TracksPt))
        #pix1 = np.array(getattr(tree, br_pix1)).reshape(width, height) #Removed bc no rec hits
        #pix2 = np.array(getattr(tree, br_pix2)).reshape(width, height) #Removed bc no rec hits
        #pix3 = np.array(getattr(tree, br_pix3)).reshape(width, height) #Removed bc no rec hits

        #jet_stack = np.stack((TracksPt, TracksD0, TracksDz, Ecal, Hcal, pix1, pix2, pix3), axis=-1)

        #del TracksPt
        #del TracksD0
        #del TracksDz
        #del Ecal
        #del Hcal
        #del pix1
        #del pix2
        #del pix3

        #TracksPt = None
        #TracksD0 = None
        #TracksDz = None
        #Ecal = None
        #Hcal = None
        #pix1 = None
        #pix2 = None
        #pix3 = None

        pts = tree.jetPt
        m0s = tree.jetM
        iphis = tree.jetSeed_iphi
        ietas = tree.jetSeed_ieta

        nJets = len(pts)
        #print(nJets, np.shape(pts))
        #print(type(pts))

        for ijet in range(nJets):
            if doJet > 0 and ijet+1 != doJet:
                continue
            else:
                pass

            y = d
            pt = pts[ijet]
            m0 = m0s[ijet]
            iphi = iphis[ijet]
            ieta = ietas[ijet]

            # crop images individually so it is less cpu intensive
            TracksPt_cj = crop_jet( TracksPt, iphi, ieta, args.granularity, jet_shape )
            TracksD0_cj = crop_jet( TracksD0, iphi, ieta, args.granularity, jet_shape )
            TracksDz_cj = crop_jet( TracksDz, iphi, ieta, args.granularity, jet_shape )
            Ecal_cj = crop_jet( Ecal, iphi, ieta, args.granularity, jet_shape )
            Hcal_cj = crop_jet( Hcal, iphi, ieta, args.granularity, jet_shape )
            #pix1 = crop_jet( pix1, iphi, ieta, args.granularity, jet_shape ) #Removed bc no rec hits
            #pix2 = crop_jet( pix2, iphi, ieta, args.granularity, jet_shape ) #Removed bc no rec hits
            #pix3 = crop_jet( pix3, iphi, ieta, args.granularity, jet_shape ) #Removed bc no rec hits

            X_jet = np.stack((TracksPt_cj, TracksD0_cj, TracksDz_cj, Ecal_cj, Hcal_cj), axis=-1) #Removed pix1, pix2, pix3

            TracksPt_cj = None
            TracksD0_cj = None
            TracksDz_cj = None
            Ecal_cj = None
            Hcal_cj = None
            #pix1 = None
            #pix2 = None
            #pix3 = None

            # stacking images when we crop so we don't have two copies of all of the image channels saved as different variables
            #X_jet = crop_jet( np.concatenate([TracksPt, TracksD0, TracksDz, Ecal, Hcal, pix1, pix2, pix3], axis=-1), iphi, ieta, args.granularity )
            #X_jet = crop_jet( jet_stack, iphi, ieta, args.granularity )

            #del jet_stack
            #jet_stack = None

            fout['X_jets'][jet_count] = X_jet  #changed index from iEvt to jet_count
            fout['jetPt'][jet_count] = pt
            fout['jetM'][jet_count] = m0
            fout['y'][jet_count] = y

            X_jets = None
            jet_count += 1

        #TracksPt = None
        #TracksD0 = None
        #TracksDz = None
        #Ecal = None
        #Hcal = None
        #pix1 = None
        #pix2 = None
        #pix3 = None

    fout.close()

print "  >> Done.\n"
