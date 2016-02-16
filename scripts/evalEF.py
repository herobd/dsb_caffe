from IPython.utils import io
import numpy as np


# calculate some error metrics to evaluate actual vs. predicted EF values obtained from FCN model
data = np.transpose(np.loadtxt('accuracy.csv', delimiter=',')).astype('float')
ids, actual_edv, actual_esv, predicted_edv, predicted_esv = data
actual_ef = (actual_edv - actual_esv) / actual_edv
actual_ef_std = np.std(actual_ef)
actual_ef_median = np.median(actual_ef)
predicted_ef = (predicted_edv - predicted_esv) / predicted_edv # potential of dividing by zero, where there is no predicted EDV value
nan_idx = np.isnan(predicted_ef)
actual_ef = actual_ef[~nan_idx]
predicted_ef = predicted_ef[~nan_idx]
MAE = np.mean(np.abs(actual_ef - predicted_ef))
RMSE = np.sqrt(np.mean((actual_ef - predicted_ef)**2))
print 'Mean absolute error (MAE) for predicted EF: {:0.4f}'.format(MAE)
print 'Root mean square error (RMSE) for predicted EF: {:0.4f}'.format(RMSE)
print 'Standard deviation of actual EF: {:0.4f}'.format(actual_ef_std)
print 'Median value of actual EF: {:0.4f}'.format(actual_ef_median)
