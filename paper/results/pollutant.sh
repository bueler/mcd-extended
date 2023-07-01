# generate results for Example 9.3 in paper
# run as
#   $ bash PATHTOHERE/pollutant.sh &> pollutant.txt
# from git@bitbucket.org:pefarrell/fascd.git in examples/

LIST2D="1 2 3 4 5 6 7 8 9"
LIST3D="1 2 3 4 5"  # level 5 almost fills 128GB memory machine

echo "******** -dim 2 *******"
for LEVS in $LIST2D; do
    python3 pollutant.py -dim 2 -levs $LEVS
done
echo ""

# which mpiexec
echo "******** -dim 3; P = 16 parallel *******"
for LEVS in $LIST3D; do
    mpiexec -n 16 --map-by core --bind-to hwthread python3 pollutant.py -dim 3 -levs $LEVS
done
echo ""

