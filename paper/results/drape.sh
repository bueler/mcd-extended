# generate results for Example 8.1 in paper

for PROB in ball spiral; do
    echo ""
    echo "******** -nofas PROB=${PROB} *******"
    for LEVS in 1 2 3 4 5 6 7 8; do
        python3 drape.py -prob $PROB -levs $LEVS -nofas -snes_rtol 1.0e-12 -snes_atol 1.0e-12
    done
done

for CYCLE in multiplicative full; do
    for PROB in ball spiral; do
        echo ""
        echo "******** PROB=${PROB} CYCLE=${CYCLE} *******"
        for LEVS in 1 2 3 4 5 6 7 8 9 10 11; do
            python3 drape.py -prob $PROB -levs $LEVS -fascd_cycle_type $CYCLE -fascd_rtol 1.0e-12 -fascd_atol 1.0e-12
        done
    done
done
