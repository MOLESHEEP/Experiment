from itertools import combinations
from statistics import mean
import numpy as np
import pandas as pd
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.functions import StepFunction

def time_dependent_concordance_index(surv_funs,y_time,y_event,points = 5):
    if not (len(y_time) == len(surv_funs) == len(y_event)):
        print(f"Ctd_index Error : time size ({len(y_time)},), event size ({len(y_event)},) and surv_funs size ({len(surv_funs)},) not match")
        return
    times = y_time.values
    times = [times[i] for i in range(0,len(times),len(times)//points)]
    time_vary_c_index = {}
    for time in times:
        #获取各个时间点分别对应的值
        c_index = concordance_index(y_time, np.array([1 - surv_funs[i](time) for i in range(len(surv_funs))]), y_event)
        time_vary_c_index[str(time)] = c_index
    mean_ctd_index = mean(time_vary_c_index.values())
    return time_vary_c_index,mean_ctd_index
        


def concordance_index(y_time, y_pred, y_event):
    if not (len(y_time) == len(y_pred) == len(y_event)):
        print(f"C_index Error : time size ({len(y_time)},), event size ({len(y_event)},) and y_pred size ({len(y_pred)},) not match")
        return
    """
    Compute concordance index.
    :param y_time: Actual Survival Times.
    :param y_pred: Predicted cumulative hazard functions or predicted survival functions.
    :param y_event: Actual Survival Events.
    :return: c-index.
    """
    predicted_outcome = [x.sum() for x in y_pred]
    possible_pairs = list(combinations(range(len(y_pred)), 2))
    concordance = 0
    permissible = 0
    for pair in possible_pairs:
        t1 = y_time.iat[pair[0]]
        t2 = y_time.iat[pair[1]]
        e1 = y_event.iat[pair[0]]
        e2 = y_event.iat[pair[1]]
        predicted_outcome_1 = predicted_outcome[pair[0]]
        predicted_outcome_2 = predicted_outcome[pair[1]]

        shorter_survival_time_censored = (t1 < t2 and e1 == 0) or (t2 < t1 and e2 == 0)
        t1_equals_t2_and_no_death = (t1 == t2 and (e1 == 0 and e2 == 0))

        if shorter_survival_time_censored or t1_equals_t2_and_no_death:
            continue
        else:
            permissible = permissible + 1
            if t1 != t2:
                if t1 < t2:
                    if predicted_outcome_1 > predicted_outcome_2:
                        concordance = concordance + 1
                        continue
                    elif predicted_outcome_1 == predicted_outcome_2:
                        concordance = concordance + 0.5
                        continue
                elif t2 < t1:
                    if predicted_outcome_2 > predicted_outcome_1:
                        concordance = concordance + 1
                        continue
                    elif predicted_outcome_2 == predicted_outcome_1:
                        concordance = concordance + 0.5
                        continue
            elif t1 == t2:
                if e1 == 1 and e2 == 1:
                    if predicted_outcome_1 == predicted_outcome_2:
                        concordance = concordance + 1
                        continue
                    else:
                        concordance = concordance + 0.5
                        continue
                elif not (e1 == 1 and e2 == 1):
                    if e1 == 1 and predicted_outcome_1 > predicted_outcome_2:
                        concordance = concordance + 1
                        continue
                    elif e2 == 1 and predicted_outcome_2 > predicted_outcome_1:
                        concordance = concordance + 1
                        continue
                    else:
                        concordance = concordance + 0.5
                        continue
    print("///concordance: %f /// permissible: %f ///"%(concordance,permissible))
    c = concordance / permissible
    return c

def time_depend_weight_brier_score(surv_funs, y_time, y_event, points = 5):
    """
    Calculate time-dependent Brier score.
    the BS was extended in Graf et al(1999).
    :param surv_funs: survival functions.
    :param observed_times: Observed event/censoring times.
    :param event_indicators: Event indicators (1 if event occurred, 0 if censored).
    :param max_time: Maximum time point to consider (optional).
    :param points: uniformly sample from y_time points
    :return: Time-dependent Brier score.
    """
    if not (len(y_time) == len(surv_funs) == len(y_event)):
        print(f"Brier Score Error : time size ({len(y_time)},), event size ({len(y_event)},) and surv_funs size ({len(surv_funs)},) not match")
        return
    y_event_for_Cox = y_event.replace({0: False, 1: True})
    times = y_time.values
    time_points = [times[i] for i in range(0,len(times),len(times)//points)]
    brier_scores = {}
    time,KM_estimate = kaplan_meier_estimator(y_event_for_Cox,y_time)
    KM_surv_funs = StepFunction(time,KM_estimate)
    for t in time_points:
        #y_true at time t
        event = ((y_time<=t) & (y_event == 1)).astype(int)
        brier = 0
        for i,fn in enumerate(surv_funs):
            #G(t)
            weight = (1 - y_event.iat[i])/KM_surv_funs(t) if y_time.iat[i] <= t else 1/KM_surv_funs(t)
            #y_hat_i(t)
            survival_prob = 1-fn(t)
            brier +=  weight * (survival_prob - event.iat[i]) ** 2
        brier_scores[str(t)] = brier / len(surv_funs)
    mean_brier_score = mean(brier_scores.values())
    print(f"mean_brier_score : {mean_brier_score} ")
    return brier_scores,mean_brier_score

def MSE(standard_surv_funs,eva_surv_funs):
    n,m = len(standard_surv_funs),len(eva_surv_funs)
    if n != m :
        print(f" Standard functions size : {n}  and evaluation functions size {m} not match")
        return
    if len(standard_surv_funs[0].y) != len(eva_surv_funs[0].y):
        print(f" Standard function length : {len(standard_surv_funs[0].y)}  and evaluation function length {len(eva_surv_funs[0].y)} not match")
        return
    mse_list = []
    for i in range(n):
        instance_mse = 0
        for y_sta,y_pre in zip(standard_surv_funs[i].y,eva_surv_funs[i].y):
            instance_mse += (y_pre - y_sta) ** 2
        mse_list.append(instance_mse/len(standard_surv_funs[i].y))
    mse = mean(mse_list)
    print(f"MSE STA AND EVA : {mse}")
    return mse,mse_list

def MAE(standard_surv_funs,eva_surv_funs):
    n,m = len(standard_surv_funs),len(eva_surv_funs)
    if n != m :
        print(f" Standard functions size : {n}  and evaluation functions size {m} not match")
        return
    # if len(standard_surv_funs[0].y) != len(eva_surv_funs[0].y):
    #     print(f" Standard function length : {len(standard_surv_funs[0].y)}  and evaluation function length {len(eva_surv_funs[0].y)} not match")
    #     return
    mae_list = []
    for i in range(n):
        instance_mae = 0
        for t in eva_surv_funs[i].x:
            instance_mae += abs(standard_surv_funs[i](t) - eva_surv_funs[i](t))
        mae_list.append(instance_mae/len(eva_surv_funs[i].x))
    mae = mean(mae_list)
    print(f"MAE STA AND EVA : {mae}")
    return mae,mae_list