function F = IFP_ReturnFn(aprime,a,z,r,w,gamma)

c = w*z + (1+r)*a - aprime;

F = -inf;
if c>0
    F = c^(1-gamma)/(1-gamma);
end

end