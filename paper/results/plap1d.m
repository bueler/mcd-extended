% generate convergence rates from results in plap1d.txt
% errors are from default V(1,1) runs

h = 2.^(-(0:9));
err = [2.263e-01 3.255e-02 9.108e-03 3.248e-03 5.501e-04 ...
       1.684e-04 4.656e-05 9.203e-06 3.430e-06 4.139e-07];
loglog(h,err,'*')
grid on,  xlabel h,  ylabel('|error|_{inf}')
legend('p=1.5')
p = polyfit(log(h),log(err),1)
%   p = 2.0047  -1.7579
