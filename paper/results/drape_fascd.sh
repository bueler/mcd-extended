# generate FASCD results for Example 8.1 in paper
# warning: the level 10,11 runs assume decent memory on you machine
# note: these runs are serial so as to fix the smoother

for PROB in ball spiral; do
    echo ""
    echo "******** PROB=${PROB}, V-cycles, tight tolerance (over-solving) *******"
    for LEVS in 1 2 3 4 5 6 7 8 9 10 11; do
        python3 drape.py -prob $PROB -levs $LEVS -fascd_cycle_type multiplicative -fascd_monitor -fascd_atol 1.0e-12 -fascd_rtol 1.0e-12 -fascd_stol 1.0e-12
    done
done

for PROB in ball spiral; do
    echo ""
    echo "******** PROB=${PROB}, F-cycles, to discretization error, with defaults rtol=stol=1.0e-8 and atol=1.0e-50 *******"
    for LEVS in 1 2 3 4 5 6 7 8 9 10 11; do
        python3 drape.py -prob $PROB -levs $LEVS -fascd_cycle_type full -fascd_monitor
    done
done
