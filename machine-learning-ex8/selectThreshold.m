function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Create a nx1 vector of all values predicted to be outliers
    predictions = (pval < epsilon);
    % True positives = both agree is anomaly
    tp = sum((yval == 1) & (predictions == 1));
    % False positives = not an anomaly but the algo thinks it is
    fp = sum((yval == 0) & (predictions == 1));
    % False negatives = is an anomaly but algo thinks it is not
    fn = sum((yval == 1) & (predictions == 0));
    % Precision = Fraction of predicted true results that are correct
    prec = tp / (tp + fp);
    % Recall = Fraction of true results predicted correctly 
    rec  = tp / (tp + fn);
    % Harmonic mean of precision and recall
    F1 = (2*prec*rec)/(prec+rec);
    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
