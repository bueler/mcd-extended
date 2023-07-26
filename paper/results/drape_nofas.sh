# generate -nofas results for Example 8.1 in paper

for PROB in ball spiral; do
    echo ""
    echo "******** -nofas PROB=${PROB} *******"
    for LEVS in 1 2 3 4 5 6 7 8; do
        python3 drape.py -prob $PROB -levs $LEVS -nofas -snes_rtol 1.0e-12 -snes_atol 1.0e-12
    done
done
