# generate FASCD results for Example 8.1 in paper

for PROB in ball spiral; do
    echo ""
    echo "******** PROB=${PROB}, V-cycles, tight tolerance (over-solving) *******"
    for LEVS in 1 2 3 4 5 6 7 8 9 10; do
        python3 drape.py -prob $PROB -levs $LEVS -fascd_cycle_type multiplicative -fascd_rtol 1.0e-12
    done
done

for PROB in ball spiral; do
    echo ""
    echo "******** PROB=${PROB}, F-cycles, to discretization error *******"
    for LEVS in 1 2 3 4 5 6 7 8 9 10; do
        python3 drape.py -prob $PROB -levs $LEVS -fascd_cycle_type full
    done
done
