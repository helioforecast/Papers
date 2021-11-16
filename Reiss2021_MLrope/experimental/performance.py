import pandas as pds
import datetime
import numpy as np
import event as evt


def evaluate(predicted_list, test_list, thres=0.51, durationCreepies=2.5):
    '''
    for each cloud of validation_list, gives the list of clouds in the
    predicted_list that overlap the cloud among the threshold
    '''
    TP = []
    FN = []
    FP = []
    detected = []
    for event in test_list:
        corresponding = evt.find(event, predicted_list, thres, 'best')
        if corresponding is None:
            FN.append(event)
        else:
            TP.append(corresponding)
            detected.append(event)
    FP = [x for x in predicted_list if max(evt.overlapWithList(x, test_list, percent=True)) == 0]
    seum = [x for x in FP if x.duration < datetime.timedelta(hours=durationCreepies)]
    for event in seum:
        FP.remove(event)
        predicted_list.remove(event)

    return TP, FN, FP, detected



