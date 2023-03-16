
cd ~/hw2/tests

make clean
make 2> make-stderr.out
FILE=./test



echo "==================================="
echo "=            Run lab2             ="
echo "==================================="

if [ -f "$FILE" ]; then
    # srun -N 3 -n 4 -c 3 ./lab2 500000000 100 2> run-stderr.out 
    srun -N 1 -n 4 -c 1 ${FILE} 2> run-stderr.out 
else
    echo "" > run-stderr.out
fi



echo "==================================="
echo "=      Print make-stderr.out      ="
echo "==================================="

cat make-stderr.out

echo "==================================="
echo "=      Print run-stderr.out       ="
echo "==================================="

cat run-stderr.out


echo ""
echo ""

