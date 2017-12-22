#!/usr/bin/env sh
expID=AI/AI_hg8   # snapshots and log file will save in checkpoints/$expID
dataset=AI        # mpii | mpii-lsp | lsp |
gpuID=0,1,2,3             # GPUs visible to program
nGPU=4                # how many GPUs will be used to train the model
batchSize=20
#LR=4e-4
LR=6.7e-4
schedule='11'
netType=hg-prm        # network architecture
nStack=8
scaleFactor=0.3
dataAug=1
focus=1
nResidual=1
nThreads=10            # how many threads will be used to load data
#nThreads=4            # how many threads will be used to load data
minusMean=true
nClasses=14
nEpochs=14
snapshot=1           # save models for every $snapshot
nFeats=256
baseWidth=9
cardinality=4

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$gpuID th main.lua \
   -dataset $dataset \
   -expID $expID \
   -batchSize $batchSize \
   -nGPU $nGPU \
   -LR $LR \
   -schedule '11' \
   -momentum 0.0 \
   -weightDecay 0.0 \
   -netType $netType \
   -nStack $nStack \
   -nResidual $nResidual \
   -scaleFactor $scaleFactor \
   -nThreads $nThreads \
   -minusMean $minusMean \
   -nClasses $nClasses \
   -nEpochs $nEpochs \
   -snapshot $snapshot \
	 -nFeats $nFeats \
	 -baseWidth $baseWidth \
	 -cardinality $cardinality \
	 -resume checkpoints/$expID \
   #-testOnly true
   #-testRelease true
   # -resume checkpoints/$expID  \  # uncomment this line to resume training
   #batchSize=16          
