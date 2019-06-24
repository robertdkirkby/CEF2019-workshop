function F=Aiyagari1994_ReturnFn(aprime_val, a_val, z_val,alpha,delta,mu,r)

% For any combinations of variables and parameters that would lead to a
% violation of some condition or constraint, can just return F=-Inf
F=-Inf;
w=(1-alpha)*((r+delta)/alpha)^(alpha/(alpha-1));
c=w*z_val+(1+r)*a_val-aprime_val; 
if c>0
    if mu==1
        F=log(c);
    else
        F=(c^(1-mu) -1)/(1-mu);
    end
end

end