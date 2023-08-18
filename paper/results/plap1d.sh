# generate results for Example 8.2 in paper
# run as
#   $ bash PATHTOHERE/plap1d.sh &> plap1d.txt
# from git@bitbucket.org:pefarrell/fascd.gitexamples/
# for convergence rates, run plap1d.m in octave

LLIST="1 2 3 4 5 6 7 8 9 10"

# do: V(1,0), V(0,1), V(1,1)
for VOPT in "-fascd_skip_upsmooth" "-fascd_skip_downsmooth" ""; do
    echo "******** p = 1.5 V-cycle VOPT=${VOPT} *******"
    for LEVS in $LLIST; do
        python3 plap1d.py $VOPT -p 1.5 -fascd_stol 0.0 -levs $LEVS
    done
    echo ""
done

for EPSREG in 0.0 1.0e-8; do
    echo "******** p = 1.5 F-cycle EPSREG=${EPSREG} *******"
    for LEVS in $LLIST; do
        python3 plap1d.py -fascd_cycle_type full -epsreg $EPSREG -p 1.5 -fascd_stol 0.0 -levs $LEVS
    done
    echo ""
done
