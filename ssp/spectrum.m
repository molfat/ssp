function [vmin, sigma] = phd(x, p)
  %
  R = toeplitz(x);
  [v, d] = eig(R);
  sigma = min(diag(d);
  index = find(diag(d) == sigma);
  vmin = v(:, index);
  end;
