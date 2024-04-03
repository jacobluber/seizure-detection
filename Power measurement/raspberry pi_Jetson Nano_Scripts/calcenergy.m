function integral = calcenergy(t, i, t1, t2)

% Find indices for t1 and t2
idx1 = find(t >= t1, 1); 
idx2 = find(t >= t2, 1);

% Trapezoidal integration
integral = 0;
for k = idx1:idx2-1
    integral = integral + (t(k+1) - t(k)) * (i(k) + i(k+1)) / 2;
end

end