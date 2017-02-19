function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% For each X value
for i = 1:size(X)
    min_dist = Inf;
    x = X(i, :);
    % Iterate through each of the centroids
    for j = 1:K
        % Calculate the squared euclidean distance from the norm
        centroid = centroids(j, :);
        dist = dot (x - centroid, x - centroid);
        % Check which centroid is best
        if dist < min_dist
            idx(i) = j;
            min_dist = dist;
        end
    end
end

end

