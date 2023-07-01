# generate results for Example 9.1 in paper

for PROB in ball spiral; do
    echo ""
    echo "******** -nofas PROB=${PROB} *******"
    for LEVS in 1 2 3 4 5 6 7 8; do
        python3 drape.py -prob $PROB -levs $LEVS -nofas
    done
done

for CYCLE in multiplicative full; do
    for PROB in ball spiral; do
        echo ""
        echo "******** PROB=${PROB} CYCLE=${CYCLE} *******"
        for LEVS in 1 2 3 4 5 6 7 8 9 10; do
            python3 drape.py -prob $PROB -levs $LEVS -fascd_cycle_type $CYCLE
        done
    done
done
