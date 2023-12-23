#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 08:00:51 2020

@author: daltonm
"""
#%matplotlib notebook
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc
import statsmodels.api as sm
from scipy.integrate import cumtrapz
from scipy.integrate import simps
from mpl_toolkits import mplot3d 

class path:
    storage = r'Z:/marmosets/processed_datasets/2019_11_26/'
    
class params:
    spkSampWin = 0.01
    trajShift = 0.02 #sample every 25ms
    lead = [0.1] #[0.2, 0.15, 0.1, 0.05, 0] #[0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0] #[0.05] # 100ms lead time
    lag = [0.3] #[0.2, 0.25, 0.3, 0.35, 0.4] #[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4] #[0.35] # 300ms lag time
    spkRatioCheck = 'off'
    normalize = 'off'
    keepMarkers = np.array([2])
    numThresh = 1000
    trainRatio = 0.9
    numIters = 100
    minSpikes = 4
    nDims = 3
    nShuffles = 1
    starting_tNum = 0 ##### this is to pull only session 3 samples. Should be changed for other purposes (26)

#%% load data

with open(path.storage + '2019_11_26_foraging_and_homeCage_spikeData.p', 'rb') as f:
    spikeData = pickle.load(f)

with open(path.storage + '2019_11_26_foraging_and_homeCage_camSignals.p', 'rb') as f:
    camSignals = pickle.load(f)
    
with open(path.storage + '2019_11_26_foraging_trajectories_session_1_2_3_shuffle1_330000.p', 'rb') as f:
    trajectories = pickle.load(f)
    
class allData:
    spikeData = spikeData
    camSignals = camSignals
    trajectories = trajectories
  
#%% fix camSignal data for sharing (TMP)

bounds = camSignals['eventBoundaries']
session = camSignals['session']
eventTimes = camSignals['eventTimes']
expTimes = camSignals['expStartSamp']
expSamples = camSignals['expStartTimes'] 
newOrder = [0, 2, 4, 1, 3, 5]
for idx in range(len(bounds)-1, 0, -2):
    del session[idx], bounds[idx], eventTimes[idx], expTimes[idx], expSamples[idx]        
bounds = [bounds[idx] for idx in newOrder]
session = [session[idx] for idx in newOrder]
eventTimes = [eventTimes[idx] for idx in newOrder]
expTimes = [expTimes[idx] for idx in newOrder]
expSamples = [expSamples[idx] for idx in newOrder]

#%% fix spikeData for sharing (TMP)



#%%

# set up dict variables for saving to pickle and mat files
combData = {'cam_session': session, 
            'cam_eventBound_samples': bounds, 
             'cam_eventBound_times':  eventTimes, 
             'cam_exposureSamples':   expSamples,
             'cam_exposureTimes':     expTimes,
             'spikes_session':        spikeData['session'],
             'spikes_channel':        spikeData['channel'],
             'spikes_unit':           spikeData['unit'],
             'spikes_timestamp':      spikeData['spikes'],
             'trajectories_position': trajectories['position'],
             'trajectories_velocity': trajectories['velocity']}
#%%
# session4mat = np.empty((len(session),), dtype=np.object)
# eventBoundaries4mat = np.empty_like(session4mat)
# segmentTimes4mat = np.empty_like(session4mat)
# expStartSamples4mat = np.empty_like(session4mat)
# expStartTimes4mat = np.empty_like(session4mat)
# for i in range(len(session)):
#     session4mat[i]          = session[i]
#     eventBoundaries4mat[i]  = eventBoundaries[i]
#     segmentTimes4mat[i]     = segmentTimes[i]
#     expStartSamples4mat[i]  = expStartSamples[i]
#     expStartTimes4mat[i]    = expStartTimes[i]
# camSignals4mat = {'session': session4mat, 'eventBoundaries': eventBoundaries4mat, 
#              'eventTimes': segmentTimes4mat, 'expStartSamp': expStartSamples4mat,
#              'expStartTimes': expStartTimes4mat}


# with open(os.path.join(path.storage, params.camSignal_filename)  + '.p', 'wb') as fp:
#     pickle.dump(camSignals, fp, protocol = pickle.HIGHEST_PROTOCOL)

# savemat(os.path.join(path.storage, params.camSignal_filename) + '.mat', mdict = camSignals4mat)
    
#%% extract data

def extract_trajectories_and_define_model_features(lead, lag):
    camPeriod = np.mean(np.diff(allData.camSignals['expStartTimes'][0][np.logical_and(allData.camSignals['expStartTimes'][0] >= allData.camSignals['eventTimes'][0][0, 1], 
                                                                                      allData.camSignals['expStartTimes'][0] < allData.camSignals['eventTimes'][0][1, 1])]))  
    camPeriod = np.round(camPeriod * 1e5) / 1e5
    
    trajSampShift = int(np.round(params.trajShift / camPeriod))
    leadSamps = int(np.round(lead / camPeriod))
    lagSamps = int(np.round(lag / camPeriod))
    
    shortSamps100 = int(np.round(.1 / camPeriod))
    shortSamps150 = int(np.round(.15 / camPeriod))
    
    nCams = 4
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    tNum = params.starting_tNum
    for sessIdx in range(0, len(allData.camSignals['eventBoundaries']), nCams):#[8]:  # to pull only session 3 data. change the this for future: #range(0, len(allData.camSignals['eventBoundaries']), nCams):
        sess = int(sessIdx / nCams)
        sessSpikes = [allData.spikeData['spikes'][idx] for idx in range(len(allData.spikeData['spikes'])) if allData.spikeData['session'][idx] == sess]
        
        if sess == 0:
            del sessSpikes[10], sessSpikes[1]
        elif sess == 1:
            del sessSpikes[12], sessSpikes[9], sessSpikes[7], sessSpikes[1] 
        else:
            del sessSpikes[10], sessSpikes[6]
        
        boundaries = allData.camSignals['eventBoundaries'][sessIdx]
        eventTimes = allData.camSignals['eventTimes'][sessIdx]
        exposures = allData.camSignals['expStartSamp'][sessIdx]
        for evt in range(np.shape(boundaries)[-1]):
            
            if tNum >= len(allData.trajectories['position']):
                break
            
            eventExp = exposures[np.logical_and(exposures >= boundaries[0, evt], exposures < boundaries[1, evt])]
            eventExpTimes = allData.camSignals['expStartTimes'][sessIdx][np.logical_and(exposures >= boundaries[0, evt], exposures < boundaries[1, evt])]
            fullTraj = allData.trajectories['velocity'][tNum]
            fullPos = allData.trajectories['position'][tNum]
            trajLength = np.shape(fullPos)[-1]
            
            if len(eventExp) == trajLength:
                print('found trajNum ' + str(tNum) + ' match at event ' + str(evt) + ' with length ' + str(len(eventExp)))
#                if tNum >= 0: #tNum == 2 or tNum == 5 or tNum == 8:
#                    ax.plot3D(fullPos[2, 0, :], fullPos[2, 1, :], fullPos[2, 2, :], c = 'black')
#                    pt = np.min(np.where(~np.isnan(fullPos[2, 0, :]))[0])
#                    ax.scatter3D(fullPos[2, 0, pt], fullPos[2, 1, pt], fullPos[2, 2, pt], c = 'red')
                numTraj = int(np.ceil((trajLength - lagSamps - leadSamps) / trajSampShift))            
                traj = np.empty((np.shape(fullTraj)[0], numTraj, 3, leadSamps + lagSamps - 1))
                shortTraj = np.empty((np.shape(fullTraj)[0], numTraj, 3, shortSamps150 - shortSamps100))
                avgSpeed = np.empty((np.shape(fullTraj)[0], numTraj))
                avgPos = np.empty((np.shape(fullTraj)[0], numTraj, 3))
                spikes = np.empty((len(sessSpikes), numTraj))
                for marker in range(np.shape(fullTraj)[0]):
                    for t, idx in enumerate(range(leadSamps, trajLength - lagSamps, trajSampShift)):                    
                        traj[marker, t, ...] = fullTraj[marker, :, idx - leadSamps : idx + lagSamps - 1]
                        shortTraj[marker, t, ...] = fullTraj[marker, :, idx + shortSamps100 : idx + shortSamps150]
                        
                        avgSpeed[marker, t] = np.mean(np.linalg.norm(traj[marker, t, ...], axis = -2))
                        avgPos[marker, t, :] = np.mean(fullPos[marker, :, idx - leadSamps : idx + lagSamps - 1], axis = -1)
    
                        if params.normalize == 'on':
                            traj[marker, t, ...] = traj[marker, t, ...] / np.linalg.norm(traj[marker, t, ...], axis = -2)
                            shortTraj[marker, t, ...] = shortTraj[marker, t, ...] / np.linalg.norm(shortTraj[marker, t, ...], axis = -2)
                    
                        # get spike/no-spike in 10ms window centered around idx
                        if marker == 0:
                            for s, unitSpks in enumerate(sessSpikes):
                                spikes[s, t] = np.round(np.sum(np.logical_and(unitSpks > eventExpTimes[idx] - params.spkSampWin/2, 
                                                                              unitSpks < eventExpTimes[idx] + params.spkSampWin/2)))
    
                if tNum == params.starting_tNum: #tNum == 0:
                    stackedTraj = traj
                    stackedShortTraj = shortTraj
                    sampledSpikes = spikes
                    stackedSpeed = avgSpeed
                    stackedPos = avgPos
                else: 
                    stackedTraj = np.hstack((stackedTraj, traj))
                    stackedShortTraj = np.hstack((stackedShortTraj, shortTraj))
                    sampledSpikes = np.hstack((sampledSpikes, spikes))
                    stackedSpeed = np.hstack((stackedSpeed, avgSpeed))
                    stackedPos = np.hstack((stackedPos, avgPos))
                
                if params.spkRatioCheck == 'on':
                    bins = np.arange(np.floor(eventTimes[0, evt] * 1e2), eventTimes[1, evt] * 1e2 + 1, 1) / 1e2
                    tmpSpikes = np.empty((len(sessSpikes), len(bins) - 1), dtype=np.int8)
                    for sNum, spks in enumerate(sessSpikes):
                        binnedSpikes = pd.DataFrame(data = spks[np.logical_and(spks >= eventTimes[0, evt], spks <= eventTimes[1, evt])], columns = ['spikeTimes'])
                        binnedSpikes['bins'] = pd.cut(binnedSpikes['spikeTimes'], bins = bins)
                        tmpSpikes[sNum, :] = np.array(binnedSpikes['bins'].value_counts(sort=False), dtype = np.int8)
                        
                    if tNum == 0:
                        reachSpikes = tmpSpikes
                    else:
                        reachSpikes = np.hstack((reachSpikes, tmpSpikes))
                           
    #            reachSpikes.append(tmpSpikes)
                tNum += 1
                
    # rearrange traj array into a list of arrays, with each element being the array of trajectories for a single marker
    trajectoryList = [stackedTraj[marker, ...] for marker in range(np.shape(stackedTraj)[0])]
    shortTrajectoryList = [stackedShortTraj[marker, ...] for marker in range(np.shape(stackedShortTraj)[0])]
    avgPos = [stackedPos[marker, ...] for marker in range(np.shape(stackedTraj)[0])]
    avgSpeed = [stackedSpeed[marker, ...] for marker in range(np.shape(stackedTraj)[0])]
    
    sampledSpikes = np.delete(sampledSpikes, np.where(np.isnan(avgSpeed[2]))[0], axis = 1)
    
    for marker in range(len(trajectoryList)):
        trajectoryList[marker] = np.delete(trajectoryList[marker], np.where(np.isnan(avgSpeed[marker]))[0], axis = 0)      
        shortTrajectoryList[marker] = np.delete(shortTrajectoryList[marker], np.where(np.isnan(avgSpeed[marker]))[0], axis = 0)      
        avgPos[marker] = np.delete(avgPos[marker], np.where(np.isnan(avgSpeed[marker]))[0], axis = 0)      
        avgSpeed[marker] = np.delete(avgSpeed[marker], np.where(np.isnan(avgSpeed[marker]))[0], axis = 0)      
    
    del stackedTraj, stackedShortTraj, stackedPos, stackedSpeed
    
    if params.spkRatioCheck == 'on':
        fullRatio = np.sum(reachSpikes == 1) / np.sum(reachSpikes == 0)
        sampledRatio = np.sum(sampledSpikes == 1) / np.sum(sampledSpikes == 0)
        print('ratio of spike/no-spike for full dataset is ' + str(fullRatio) + ', ratio for sampled set is ' + str(sampledRatio)) 
    
    # Take PCA of trajectories
    
#    if 'allTraj' in locals():
#        del allTraj
    for mk, (traj, shortTraj) in enumerate(zip(trajectoryList, shortTrajectoryList)):
        traj = np.reshape(traj, (np.shape(traj)[0], np.shape(traj)[1] * np.shape(traj)[2]))
        shortTraj = np.reshape(shortTraj, (np.shape(shortTraj)[0], np.shape(shortTraj)[1] * np.shape(shortTraj)[2]))        
        if np.any(params.keepMarkers == mk):
            if 'allTraj' not in locals():
                allTraj = traj
                allShortTraj = shortTraj
            else:
                allTraj = np.hstack(allTraj, traj)
                allShortTraj = np.hstack(allShortTraj, shortTraj)

##### PCA on full trajectory ###### 
    
#    allTraj = StandardScaler().fit_transform(allTraj) 
    allTraj_tmp = allTraj.copy()
    
    # find out how many PCs to use
    pca = PCA()
    pca.fit(allTraj_tmp)      
    
    cumVar = np.cumsum(pca.explained_variance_ratio_)
#    plt.plot(cumVar, '-o')
#    plt.plot(0.9*np.ones(np.shape(cumVar)))
#    plt.show()
    
    cutComp = np.where(cumVar >= 0.9)[0][0]
#    cutComp = np.where(cumVar >= 0.8)[0][0] 
#    print(cutComp)
    
    # redo the pca and get projections with the desired number of comps
    pca = PCA(n_components = cutComp+1)
    projectedTraj = pca.fit_transform(allTraj)
    features = np.hstack((projectedTraj, np.expand_dims(avgSpeed[2], axis=1), avgPos[2]))
    # features = np.hstack((projectedTraj, np.expand_dims(avgSpeed[2], axis=1)))
#    features = np.hstack((projectedTraj, avgPos[2]))
#    features = projectedTraj
    compsOut = pca.components_

##### PCA on short trajectory ######

#    allTraj = StandardScaler().fit_transform(allShortTraj) 
    allShortTraj_tmp = allShortTraj.copy() 
    
    pca = PCA()
    pca.fit(allShortTraj_tmp)      
    
    cumVar = np.cumsum(pca.explained_variance_ratio_)
#    plt.plot(cumVar, '-o')
#    plt.plot(0.9*np.ones(np.shape(cumVar)))
#    plt.show()
    
#    cutComp = np.where(cumVar >= 0.95)[0][0]
#    print(cutComp)
    
    # redo the pca and get projections with the desired number of comps
    pca = PCA(n_components = cutComp+1)
    projectedShortTraj = pca.fit_transform(allShortTraj)        
    shortFeatures = np.hstack((projectedShortTraj, np.expand_dims(avgSpeed[2], axis=1), avgPos[2]))
    # shortFeatures = np.hstack((projectedShortTraj, np.expand_dims(avgSpeed[2], axis=1) ))
#    shortFeatures = np.hstack((projectedShortTraj, avgPos[2]))
#    shortFeatures = projectedShortTraj
    
    plt.show()
    
    return features, shortFeatures, sampledSpikes, compsOut 

#%%

def train_and_test_glm(features, sampledSpikes, mode):   
    aucComb = []
    done = np.zeros((np.shape(sampledSpikes)[0],))
    bestAUC = np.ones_like(done) * 0.75
    for n in range(params.numIters):

        # Create train/test datasets for cross-validation
        print('iteration = ' +str(n))
        testSpikes = []
        trainSpikes = []
        trainFeatures = []
        testFeatures = []
        for unit, spikes in enumerate(sampledSpikes):
            spikeIdxs = np.where(spikes == 1)[0]
            noSpikeIdxs = np.where(spikes == 0)[0]
                
            idxs = np.union1d(spikeIdxs, noSpikeIdxs)
            trainIdx = np.hstack((np.random.choice(spikeIdxs, size = round(params.trainRatio*len(spikeIdxs)), replace = False), 
                                 np.random.choice(noSpikeIdxs, size = round(params.trainRatio*len(noSpikeIdxs)), replace = False)))
            testIdx = np.setdiff1d(idxs, trainIdx)
                
            if np.sum(spikes[testIdx] == 1) >= params.minSpikes:
                trainSpikes.append(spikes[trainIdx])
                testSpikes.append(spikes[testIdx])
                trainFeatures.append(features[trainIdx, :])
                testFeatures.append(features[testIdx, :])                 
                
                if n == 0:
                    print('unit ' + str(unit) + ' is prepared for GLM with ' + str(int(params.trainRatio*100)) + '/' + str(int(100-params.trainRatio*100)) + 
                          ' split, with train/test spikes = ' + str((np.sum(spikes[trainIdx] == 1), np.sum(spikes[testIdx] == 1))))        
            else:
                if n == 0:
                    print('unit ' + str(unit) + ' had only ' + str(np.sum(spikes[testIdx] == 1)) + ' spikes in the sampled time windows and is removed from analysis')
            
        # Train GLM
        
        models = []
        predictions = []
        trainPredictions = []
        coef = np.empty((np.shape(trainFeatures[0])[1] + 1, len(trainSpikes)))
        pVals = np.empty_like(coef)
        aic = np.empty((len(trainSpikes), ))
        for unit, trainSpks in enumerate(trainSpikes):
            if mode == 'shuffle':
                trainSpks = np.random.permutation(trainSpks)
            glm = sm.GLM(trainSpks, sm.add_constant(trainFeatures[unit]), family=sm.families.Poisson(link=sm.families.links.log))
            encodingModel = glm.fit()
            coef[:, unit] = encodingModel.params            
            pVals[:, unit] = np.round(encodingModel.pvalues, decimals = 3)            
            aic[unit] = round(encodingModel.aic, 1)            
            models.append(encodingModel)
            predictions.append(encodingModel.predict(sm.add_constant(testFeatures[unit])))
            trainPredictions.append(encodingModel.predict(sm.add_constant(trainFeatures[unit]))) 
                
        if n == 0:
            allModelsCoefs = coef
            coef_pVals = pVals
            modelAICs = aic
        else:
            allModelsCoefs = np.dstack((allModelsCoefs, coef))
            coef_pVals = np.dstack((coef_pVals, pVals))
            modelAICs = np.vstack((modelAICs, aic))
            
        # Test GLM --> area under ROC
        
        allHitProbs = []
        allFalsePosProbs = []
        areaUnderROC = []
        for unit, preds in enumerate(predictions):
            thresholds = np.linspace(preds.min(), preds.max(), params.numThresh)            
            hitProb = np.empty((len(thresholds),))
            falsePosProb = np.empty((len(thresholds),))
            for t, thresh in enumerate(thresholds):    
                posIdx = np.where(preds > thresh)
                hitProb[t] = np.sum(testSpikes[unit][posIdx] == 1) / np.sum(testSpikes[unit] == 1)
                falsePosProb[t] = np.sum(testSpikes[unit][posIdx] == 0) / np.sum(testSpikes[unit] == 0)
            
            areaUnderROC.append(auc(falsePosProb, hitProb))
            
            allHitProbs.append(hitProb)
            allFalsePosProbs.append(falsePosProb)
    
#            if mode == 'real' and areaUnderROC[-1] > bestAUC[unit] and done[unit] == 0:
#                fig = plt.figure()
#                ax = plt.axes(projection=None)
#                ax.plot(preds)
#                ax.set_title('Unit' + str(unit))
#                tmp = np.array(testSpikes[unit])
#                tmp[tmp == 0] = np.nan
#                tmp[~np.isnan(tmp)] = preds[~np.isnan(tmp)]
#                ax.plot(tmp, 'o', c = 'orange')
#                plt.show()
#                done[unit] = 1
#                bestAUC[unit] = areaUnderROC[-1]
#                print((len(preds), len(testSpikes[unit])))
            
    #        plt.plot(falsePosProb, hitProb)
        
    #    plt.plot(np.linspace(0,1,params.numThresh), np.linspace(0,1,params.numThresh), '-k')
    #    plt.show()
        
        # ROC on train data
        areaUnderROC_train = []
        for unit, preds in enumerate(trainPredictions):
            thresholds = np.linspace(preds.min(), preds.max(), params.numThresh)
            hitProb = np.empty((len(thresholds),))
            falsePosProb = np.empty((len(thresholds),))
            for t, thresh in enumerate(thresholds):    
                posIdx = np.where(preds > thresh)
                hitProb[t] = np.sum(trainSpikes[unit][posIdx] == 1) / np.sum(trainSpikes[unit] == 1)
                falsePosProb[t] = np.sum(trainSpikes[unit][posIdx] == 0) / np.sum(trainSpikes[unit] == 0)
            
            areaUnderROC_train.append(auc(falsePosProb, hitProb))
            
    #        plt.plot(falsePosProb, hitProb)
        
    #    plt.plot(np.linspace(0,1,params.numThresh), np.linspace(0,1,params.numThresh), '-k')
    #    plt.show()
            
        aucComb.append(np.vstack((np.array(areaUnderROC), np.array(areaUnderROC_train))).transpose())
        
    return aucComb, allModelsCoefs, coef_pVals, modelAICs


#%%
                
def test_model_significance(trueAUC_means, shuffleAUC_means):
    
    p_val = np.empty((np.shape(trueAUC_means)[0], np.shape(trueAUC_means)[1]))
    for unit, (trueMean, shuffleMeans) in enumerate(zip(trueAUC_means, shuffleAUC_means)):
        p_val[unit, 0] = np.sum(shuffleMeans[0, :] > trueMean[0]) / np.shape(shuffleMeans)[-1]     
        p_val[unit, 1] = np.sum(shuffleMeans[1, :] > trueMean[1]) / np.shape(shuffleMeans)[-1]     
    
    return p_val

#    aucRes = np.empty((np.shape(aucComb[0])[0], np.shape(aucComb[0])[1], len(aucComb)))    
#    shuffledAuC = np.empty_like(aucRes)        
#    for a, (areas, shuffledAreas) in enumerate(zip(aucComb, shuffledAucComb)):
#        aucRes[..., a] = areas * 100
#        shuffledAuC[..., a] = shuffledAreas * 100
#    
#    aucError = np.std(aucRes, axis = -1) / np.sqrt(np.shape(aucRes)[-1])
#    aucMean = np.mean(aucRes, axis = -1)
#    
#    # find mean and std of shuffled distribution, see how far mean auc is above shuffled auc
#    shuffledMean = np.mean(shuffledAuC, axis = -1)
#    shuffledSTD = np.std(shuffledAuC, axis = -1)
#    stDevsAboveShuffles = np.divide(aucMean - shuffledMean, shuffledSTD)
#    
#    shuffleTest = np.empty((np.shape(aucComb[0])[0], np.shape(aucComb[0])[1], params.nShuffles))
#    meanShuffleTest = np.empty_like(shuffleTest)
#    for shuffle in range(params.nShuffles):
#        trueSampIdx = np.random.choice(np.arange(len(aucComb)), size = np.shape(aucRes)[1], replace = True)
#        shuffledSampIdx = np.random.choice(np.arange(len(aucComb)), size = np.shape(aucRes)[1], replace = True)
#        shuffleTest[:, 0, shuffle] = np.greater(aucRes[:, 0, trueSampIdx[0]], shuffledAuC[:, 0, shuffledSampIdx[0]])
#        shuffleTest[:, 1, shuffle] = np.greater(aucRes[:, 1, trueSampIdx[1]], shuffledAuC[:, 1, shuffledSampIdx[1]])
#        
#        meanShuffleTest[:, 0, shuffle] = np.greater(aucMean[:, 0], shuffledAuC[:, 0, shuffledSampIdx[0]])
#        meanShuffleTest[:, 1, shuffle] = np.greater(aucMean[:, 1], shuffledAuC[:, 1, shuffledSampIdx[1]])    
#
#    shuffleSig = np.sum(shuffleTest, axis = -1) / params.nShuffles
#    meanShuffleSig = np.sum(meanShuffleTest, axis = -1) / params.nShuffles    
#    t, p = ttest_ind(aucRes, shuffledAuC, axis = -1, equal_var = False)
#    
##    d = {'testAuC': aucMean[:, 0], 'testError': aucError[:, 0], 'testStDevs': stDevsAboveShuffles[:, 0], 
##         'testShuffledMean': shuffledMean[:, 0],  'testP': p[:, 0], 'testShuffleSig': shuffleSig[:, 0],  
##         'trainAuC': aucMean[:, 1], 'trainError': aucError[:, 1], 'trainStDevs': stDevsAboveShuffles[:, 1], 
##         'trainShuffledMean': shuffledMean[:, 1], 'trainP': p[:, 1], 'trainShuffleSig': shuffleSig[:, 1]}
#    d = {'testAuC': aucMean[:, 0], 'testError': aucError[:, 0], 'testShuffledMean': shuffledMean[:, 0], 
#         'testShuffleSig_dist': shuffleSig[:, 0], 'testShuffleSig_mean': meanShuffleSig[:, 0], 
#         'trainAuC': aucMean[:, 1], 'trainError': aucError[:, 1], 'trainShuffledMean': shuffledMean[:, 1], 
#         'trainShuffleSig_dist': shuffleSig[:, 1], 'trainShuffleSig_mean': meanShuffleSig[:, 1],}
#    aucResults = pd.DataFrame(data=d)
    
   
#%% run model over range of time leads and lags

results = []
meanAUC = []
shortMeanAUC = []
shuffleMeanAUC = []
allModelsCoefs = []
PCAcomps = []
modelAICs = []
param_pVals = []
for lead, lag in zip(params.lead, params.lag):
    features, shortFeatures, sampledSpikes, pComps = extract_trajectories_and_define_model_features(lead, lag)
    trueAUC, coefs, coef_pVals, AICs = train_and_test_glm(features, sampledSpikes, 'real')
    trueAUC_means = np.mean(np.moveaxis(np.array(trueAUC), 0, -1), axis = -1)
    
    shortAUC, shortCoefs, shortCoef_pVals, shortAICs = train_and_test_glm(shortFeatures, sampledSpikes, 'real')
    shortAUC_means = np.mean(np.moveaxis(np.array(shortAUC), 0, -1), axis = -1)
    
    shuffleAUC_means = np.empty((np.shape(trueAUC_means)[0], np.shape(trueAUC_means)[1], params.nShuffles))
    for s in range(params.nShuffles):
        print('')
        print('shuffle = ' + str(s))
        print('')
        shuffleAUC_tmp = train_and_test_glm(features, sampledSpikes, 'shuffle')[0] 
        shuffleAUC_means[..., s] = np.mean(np.moveaxis(np.array(shuffleAUC_tmp), 0, -1), axis = -1)            
    
    p_val = test_model_significance(trueAUC_means, shuffleAUC_means)
    
    results.append(trueAUC)
    meanAUC.append(trueAUC_means)
    shortMeanAUC.append(shortAUC_means)
    shuffleMeanAUC.append(shuffleAUC_means)
    allModelsCoefs.append(coefs)
    PCAcomps.append(pComps)
    modelAICs.append(AICs)
    param_pVals.append(coef_pVals)

    d = {'testAuC':          np.round(trueAUC_means[:, 0], 3), 
         'shortAUC':         np.round(shortAUC_means[:, 0], 3),
         'testShuffleAuC':   np.round(np.mean(shuffleAUC_means[:, 0], -1), 3),
         'p_val':            np.round(p_val[:, 0], 3)}
#    d = {'testAuC':          np.round(trueAUC_means[:, 0], 3), 
#         'shortAUC':         np.round(shortAUC_means[:, 0], 3)}
    testResults = pd.DataFrame(data=d)
    
#    d = {'trainAuC':         np.round(trueAUC_means[:, 1], 3),
#         'trainShuffleAuC':  np.round(np.mean(shuffleAUC_means[:, 1], -1), 3), 
#         'p_val':            np.round(p_val[:, 1], 3)}
    d = {'trainAuC':         np.round(trueAUC_means[:, 1], 3),
         'shortAUC':         np.round(shortAUC_means[:, 1], 3)}
#    trainResults = pd.DataFrame(data=d)
    
    print(testResults)
#    print(trainResults)
#%%
    fig = plt.figure()
    ax = plt.axes(projection=None)
    plt.plot(shortAUC_means[:, 0], trueAUC_means[:, 0], 'or')
    plt.plot(np.arange(0.48, 1.05, 0.1), np.arange(0.48, 1.05, 0.1), '--k')
    ax.set_xlim(0.48, 0.62)
    ax.set_ylim(0.48, 0.62)
    for axis in ['bottom','left']:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color('black')
    for axis in ['top','right']:
        ax.spines[axis].set_linewidth(0)
    ax.tick_params(width=2, length = 4, labelsize = 12)
    ax.set_xlabel('ROC area (+100 to +150 ms)', fontsize = 18, fontweight = 'bold')
    ax.set_ylabel('ROC area (-100 to +300 ms)', fontsize = 18, fontweight = 'bold')
    ax.grid(False)
    plt.show()
    
#%%
#k = 7
#superMeanAUC = [np.mean(-bottleneck.partition(-aucs[:, 0], k)[:k], axis = 0) for aucs in meanAUC]
#print(superMeanAUC)
#
#fig = plt.figure()
#ax = plt.axes()
#plt.plot(superMeanAUC)
#ax.set_ylim(min(superMeanAUC) - 0.1, max(superMeanAUC) + 0.1)
#plt.show()
#### NOTE
    
    # find mean and SE of coefs, determine if I should use the average coefs for computing pathlet
    # multiply pComps * coefs to get k
    # integrate k to get pathlet
    
    # use average k to get pathlet
    # calculate some pathlets from random trials and see how different they are

##### I should iterate over all trials and calculate cumulative divergence (just distance between match timepoints, summed)

#%%

pathlets = []
distance = []
for n, (coefs, comps) in enumerate(zip(allModelsCoefs, PCAcomps)):
    comps = comps.transpose()
#    beta = coefs[1:np.shape(comps)[-1]+1, :, 10]    
    beta = np.mean(coefs, axis = -1)[1:np.shape(comps)[-1]+1, :]
    velTraj = comps @ beta
    velTraj = np.swapaxes(velTraj.reshape((params.nDims, int(np.shape(velTraj)[0] / params.nDims), np.shape(velTraj)[-1])), 0, 1)
    
#    posTraj = np.empty(np.shape(velTraj))
#    for unit in range(np.shape(velTraj)[-1]):
#        posTraj[..., unit] = cumtrapz(velTraj[..., unit], dx = (params.lag[0] + params.lead[0]) / np.shape(velTraj)[0], axis = 0, initial = 0)

    posTraj = cumtrapz(velTraj, dx = (params.lag[0] + params.lead[0]) / np.shape(velTraj)[0], axis = 0, initial = 0)
    dist = simps(np.linalg.norm(velTraj, axis = 1), dx = (params.lag[0] + params.lead[0]) / np.shape(velTraj)[0], axis = 0)
        
    pathlets.append(posTraj)
    distance.append(dist)
    
    pathDivergence = np.empty(np.shape(coefs[0, ...].transpose()))
    for samp in range(np.shape(coefs)[-1]):
        beta_samp = coefs[1:np.shape(comps)[-1] +1, :, samp]
        velTraj_samp = comps @ beta_samp
        velTraj_samp = np.swapaxes(velTraj_samp.reshape((params.nDims, int(np.shape(velTraj_samp)[0] / params.nDims), np.shape(velTraj_samp)[-1])), 0, 1)
        posTraj_samp = cumtrapz(velTraj_samp, dx = (params.lag[0] + params.lead[0]) / np.shape(velTraj_samp)[0], axis = 0, initial = 0)
        pathDivergence[samp, :] = np.sum(np.linalg.norm(posTraj - posTraj_samp, axis = 1), axis = 0)
        
        divShuffle = np.empty((np.shape(pathDivergence)[0], np.shape(pathDivergence)[1], 100))
        for shuffle in range(100):
            idx = np.random.choice(np.arange(np.shape(pathDivergence)[1]), size = np.shape(pathDivergence)[1], replace = 0)
            while np.sum(idx == np.arange(np.shape(pathDivergence)[1])) > 0:
                idx = np.random.choice(np.arange(np.shape(pathDivergence)[1]), size = np.shape(pathDivergence)[1], replace = 0)
    
            divShuffle[samp, :, shuffle] = np.sum(np.linalg.norm(posTraj[..., idx] - posTraj_samp, axis = 1), axis = 0)
    
    pathDivergence_mean = np.mean(pathDivergence, axis = 0)
    shuffledPathDivergence_mean = np.mean(np.mean(divShuffle, axis = -1), axis = 0)
#%%
#
unit = 7
title = 'Unit ' + str(unit)

leadSamp = round(params.lead[0] / (params.lead[0] + params.lag[0]) * len(pathlets[0][:, 0, 0]))
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(pathlets[0][:leadSamp + 1, 0, unit], pathlets[0][:leadSamp + 1, 1, unit], pathlets[0][:leadSamp + 1, 2, unit], 'blue')
ax.plot3D(pathlets[0][leadSamp:, 0, unit], pathlets[0][leadSamp:, 1, unit], pathlets[0][leadSamp:, 2, unit], 'red')
ax.set_title(title, fontsize = 16, fontweight = 'bold')
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.set_xlabel('x', fontsize = 14)
ax.set_ylabel('y', fontsize = 14)
ax.set_zlabel('z', fontsize = 14)
ax.legend(['lead', 'lag'], loc='upper right', bbox_to_anchor=(0.9, 0.5), fontsize = 14, shadow=False)
ax.w_xaxis.line.set_color('black')
ax.w_yaxis.line.set_color('black')
ax.w_zaxis.line.set_color('black')
#ax.w_xaxis.('black')
plt.show()

#
#simpleResults = testRes.iloc[:, [0, 3, 4]]
#simpleResults.columns = ['AUC', 'p-val', 'p-val_withMean']
#simpleResults.loc[:, 'AUC'] = np.round(simpleResults.loc[:, 'AUC'] * 1e-2, decimals = 2)
#simpleResults.loc[:, 'p-val'] = np.round(1 - simpleResults.loc[:, 'p-val'], decimals = 2)
#simpleResults.loc[:, 'p-val_withMean'] = np.round(1 - simpleResults.loc[:, 'p-val_withMean'], decimals = 2)
#print(simpleResults)

#%% 

#avgAUC = []
#nNeurons = []
#cut = 57    
#for res in results:
#    nNeurons.append(np.sum(res['testAuC'] > cut))
#    avgAUC.append(np.mean(res['testAuC'].loc[res['testAuC'] > cut]))
#
#d = {'avgAUC': avgAUC, 'neurons': nNeurons}
#print(pd.DataFrame(data = d))