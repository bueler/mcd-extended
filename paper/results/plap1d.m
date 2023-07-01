% generate convergence rates from results in plap1d.txt

h = 2.^(-(0:9));
fasterr = [2.263e-01 3.255e-02 9.108e-03 3.248e-03 5.505e-04 ...
           1.690e-04 4.717e-05 9.522e-06 3.591e-06 4.948e-07];
degerr = [8.700e-02 3.812e-02 1.589e-02 6.318e-03 2.330e-03 ...
          6.992e-04 2.047e-04 1.078e-04 3.997e-05 1.506e-05];
loglog(h,fasterr,'*',h,degerr,'o')
grid on,  xlabel h,  ylabel('|error|_{inf}')
legend('p=1.5','p=4.0')
fastp = polyfit(log(h),log(fasterr),1)
degp = polyfit(log(h),log(degerr),1)

%   fastp = 2.0047  -1.7579
%   degp  = 1.4203  -2.2802
