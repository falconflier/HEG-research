from CRABClient.UserUtilities import config, getUsernameFromSiteDB
import random
config = config()


mc='TTbar'
pT=200
m_num=2000
filelist_ = ''
files_ = []
the_name=''
splitting = 1 
nJets_=1
isTTbar_=0

#dataset_ = '/RSGluonToTT_M-4000_TuneCUETP8M1_14TeV-pythia8/PhaseIITDRFall17DR-PU200_93X_upgrade2023_realistic_v2-v1/GEN-SIM-RECO'
assert mc == 'TTbar' or mc == 'QCD'
if mc=='TTbar':
    filelist_ = 'aod_m-%d_filelist.txt' % (m_num)
    the_name = 'TTbar_pT%d_%dTjet_aod_m-%d' % (pT,nJets_,m_num)
    splitting = 20 #current ~50 events per file
    files_ = open(filelist_).readlines()
    isTTbar_=1
elif mc=='QCD':
    filelist_ = '/uscms/home/bburkle/nobackup/working_area/CMSSW_5_3_32/src/opendatadnn/step2_QCDPt_15to3000'
    the_name = 'QCD_pT%d_njets%d_opendata' % (pT,nJets_)
    splitting = 3 #current 300 events per file
    files_ = open(filelist_).readlines()[:150]
    random.shuffle(files_)
    files_ = files_[:150]
    #nJets_=1
    isTTbar_=0

outputFile_ = '%s.root' % the_name
maxEvents_ = -1
skipEvents_ = 0


config.General.requestName = the_name
config.General.workArea = '/home/npervan/e2e/CMSSW_8_0_21/src/MLAnalyzer'
config.General.transferOutputs = True
config.General.transferLogs = True

config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'RecHitAnalyzer/python/ConfFile_cfg.py'
config.JobType.pyCfgParams = ['outputFile=%s'%outputFile_,'isTTbar=%d'%isTTbar_]
config.JobType.maxMemoryMB = 4000
config.JobType.allowUndistributedCMSSW = True
#config.JobType.pyCfgParams = ['maxEvents=%s'%maxEvents_,'outputFile=%s'%outputFile_,'skipEvents=%s'%skipEvents_]
#config.JobType.pluginName = 'RecHitAnalyzer/python/ConfFile_cfg.py outputFile=%s maxEvents=%s skipEvents=%s' % (outputFile_, maxEvents_, skipEvents_)

#config.Data.userInputFiles = files_
config.Data.inputDataset = files_[0]
#config.Data.inputDBS = 'global'
config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = splitting #if splitting by files
config.Data.publication = False
config.Data.outLFNDirBase = '/store/user/%s/e2e/test/' % (getUsernameFromSiteDB())
config.Data.outputDatasetTag = the_name

config.Data.ignoreLocality = True
config.Site.whitelist = ['T3_US_Brown']
config.Site.ignoreGlobalBlacklist = True 

config.Site.storageSite = 'T3_US_FNALLPC'

