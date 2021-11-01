from scipy.stats.stats import pearsonr

from icecream import ic

ic(pearsonr([1,2,3,0], [2,3,4,0])[0])
    
