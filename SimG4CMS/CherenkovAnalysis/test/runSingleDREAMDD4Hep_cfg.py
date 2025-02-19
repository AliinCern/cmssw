import FWCore.ParameterSet.Config as cms
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

process = cms.Process("CaloTest", dd4hep)
process.load("SimGeneral.HepPDTESSource.pdt_cfi")
process.load("Configuration.EventContent.EventContent_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("SimG4CMS.CherenkovAnalysis.gun_cff")
process.load("Configuration.Geometry.GeometryDD4hep_cff")
process.load("Geometry.HcalCommonData.caloSimulationParameters_cff")
process.load('GeneratorInterface.Core.generatorSmeared_cfi')
process.load("SimG4Core.Application.g4SimHits_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('Cherenkov-e10-a0-dd4hep.root')
)


process.MessageLogger.cerr.FwkReport.reportEvery = 5
if hasattr(process,'MessageLogger'):
    process.MessageLogger.EcalSim=dict()
    process.MessageLogger.HCalGeom=dict()
    process.MessageLogger.CherenkovAnalysis=dict()
    process.MessageLogger.SimG4CoreGeometry=dict()

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789

process.load("SimG4CMS.CherenkovAnalysis.cherenkovAnalysis_cfi")

process.p1 = cms.Path(process.generator*process.VtxSmeared*process.generatorSmeared*process.g4SimHits*process.cherenkovAnalysis)

process.DDDetectorESProducer.confGeomXMLFiles = cms.FileInPath("SimG4CMS/CherenkovAnalysis/data/SingleDREAMDD4Hep.xml")
process.generator.PGunParameters.MinE = 10.0
process.generator.PGunParameters.MaxE = 10.0
process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.type = 'SimG4Core/Physics/QGSP_FTFP_BERT_EML'
process.g4SimHits.OnlySDs = ['CaloTrkProcessing', 'DreamSensitiveDetector']
process.g4SimHits.ECalSD = cms.PSet(
    TestBeam = cms.untracked.bool(False),
    ReadBothSide = cms.untracked.bool(True),
    BirkL3Parametrization = cms.bool(False),
    doCherenkov = cms.bool(True),
    BirkCut = cms.double(0.1),
    BirkC1 = cms.double(0.013),
    BirkC3 = cms.double(0.0),
    BirkC2 = cms.double(9.6e-06),
    SlopeLightYield = cms.double(0.0),
    UseBirkLaw = cms.bool(False),
    BirkSlope = cms.double(0.253694)
)

