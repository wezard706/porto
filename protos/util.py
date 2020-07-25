import numpy as np
from logging import StreamHandler, INFO, DEBUG, Formatter, FileHandler, getLogger

logger = getLogger(__name__)

def calc_usample_len(pos_len, neg_len, desired_apriori=0.1):
    assert pos_len < neg_len
    print('before: (pos_len, neg_len)=({}, {})'.format(pos_len, neg_len))
    
    # Calculate the undersampling rate and resulting number of records with target=0
    usample_rate = ((1-desired_apriori) * pos_len)/(neg_len * desired_apriori)
    usample_len = int(round(usample_rate * neg_len, 1))
    print('after: (pos_len, neg_len)=({}, {})'.format(pos_len, usample_len))
    return usample_len
    

if __name__=='__main__':
    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel(INFO)
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    d_pos = np.ones(100)
    d_neg = np.zeros(100000)
    
    sample_len = calc_usample_len(len(d_pos), len(d_neg))
