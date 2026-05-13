function d43 = d43FromMoments(moments)
%D43FROMMOMENTS Calculate d43 = M4 / M3 from diameter moments.

if numel(moments) < 5 || abs(moments(4)) < eps
    d43 = NaN;
else
    d43 = moments(5) / moments(4);
end

end
