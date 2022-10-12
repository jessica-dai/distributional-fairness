import numpy as np

def f_err(pred_y, fair_y):
    return np.mean((pred_y - fair_y)**2)

def f_fair(Y, S):
    """
    Y: predictions vector (float)
    S: sensitive attribute vector (float)
    """
    tt = np.linspace(min(Y), max(Y), 1000) # tt represents discretized possible values of Y
    unique, counts = np.unique(S, return_counts=True) # should be [0, 1], [count_S==0, count_S==1] 
    fai = 0 
    for t in tt: #
        # percent of Y's < t from group 0 minus percent of Y's < t from group 1
        calc = len(Y[(S == 0) & (Y <= t)]) / counts[0] - len(Y[(S == 1) & (Y <= t)]) / counts[1]
        fai = max(fai, abs(calc))
    return fai 

def f_postprocess_vectorized(YL, YT, SL, ST):
    """
    YL: train probabilities
    YT: probabilities to postprocess
    SL: train sens
    ST: sens for postprocess
    """
    
    ST = ST.astype(int)
    
    # NEW **** 09.01.21
    YL = YL + np.random.uniform(size=len(YL))*0.05
    YT = YT + np.random.uniform(size=len(YT))*0.05  
    # train proper (we can save tmp1 artifact)
    sens_counts = np.array([len(SL) - sum(SL), sum(SL)]) # num 0, num 1
    ps = sens_counts/len(SL)
    grid = np.linspace(min(YL), max(YL), 100)
    tmp1_0 = np.sum(YL[np.where(SL == 1)] < grid.reshape([-1, 1]), axis=1)/sens_counts[1] # length 100 (for grid) 
    tmp1_1 = np.sum(YL[np.where(SL == 0)] < grid.reshape([-1, 1]), axis=1)/sens_counts[0]
    
    tmp1 = np.concatenate((tmp1_0.reshape([-1, 1]), tmp1_1.reshape([-1, 1])), axis=1) # 100 rows (1 per threshold) and 2 columns (1 per group)
    tmp1 = tmp1.transpose() # index by [group, threshold]
    
    # can't figure out how to combine these lines, surely there must be a way
    tmp2 = np.zeros(len(YT)) # 1 entry in tmp2 per prediction to correct
    tmp2[np.where(ST == 0)] = np.sum(YL[np.where(SL == 0)] < YT[np.where(ST == 0)].reshape([-1, 1]), axis=1)/sens_counts[0]
    tmp2[np.where(ST == 1)] = np.sum(YL[np.where(SL == 1)] < YT[np.where(ST == 1)].reshape([-1, 1]), axis=1)/sens_counts[1] 
    
    # calculate best t index for each prediction to correct
    ts_best = grid[np.argmin(np.abs(tmp1[ST] - tmp2.reshape([-1, 1])), axis=1)]
    
    return ps[ST]*YT + ps[1 - ST]*ts_best

def f_postprocess_vectorized_adjustments(YL, YT, SL, ST):
    """
    RETURNS ADJUSTMENTS TO YT ONLY, DOES NOT FIX YT IN-PLACE
    ret = fixed - YT
    fixed = YT + ret

    YL: train probabilities
    YT: probabilities to postprocess
    SL: train sens
    ST: sens for postprocess
    """
    
    ST = ST.astype(int)
    
    # NEW **** 09.05.21
    YL = YL + np.random.uniform(size=len(YL))*0.05
    YT = YT + np.random.uniform(size=len(YT))*0.05  

    # train proper (we can save tmp1 artifact)
    sens_counts = np.array([len(SL) - sum(SL), sum(SL)]) # num 0, num 1
    ps = sens_counts/len(SL)
    grid = np.linspace(min(YL), max(YL), 100)
    tmp1_0 = np.sum(YL[np.where(SL == 1)] < grid.reshape([-1, 1]), axis=1)/sens_counts[1] # length 100 (for grid) 
    tmp1_1 = np.sum(YL[np.where(SL == 0)] < grid.reshape([-1, 1]), axis=1)/sens_counts[0]
    
    tmp1 = np.concatenate((tmp1_0.reshape([-1, 1]), tmp1_1.reshape([-1, 1])), axis=1) # 100 rows (1 per threshold) and 2 columns (1 per group)
    tmp1 = tmp1.transpose() # index by [group, threshold]
    
    # can't figure out how to combine these lines, surely there must be a way
    tmp2 = np.zeros(len(YT)) # 1 entry in tmp2 per prediction to correct
    tmp2[np.where(ST == 0)] = np.sum(YL[np.where(SL == 0)] < YT[np.where(ST == 0)].reshape([-1, 1]), axis=1)/sens_counts[0]
    tmp2[np.where(ST == 1)] = np.sum(YL[np.where(SL == 1)] < YT[np.where(ST == 1)].reshape([-1, 1]), axis=1)/sens_counts[1] 
    
    # calculate best t index for each prediction to correct
    ts_best = grid[np.argmin(np.abs(tmp1[ST] - tmp2.reshape([-1, 1])), axis=1)]

    fixed = ps[ST]*YT + ps[1 - ST]*ts_best
    
    return fixed - YT