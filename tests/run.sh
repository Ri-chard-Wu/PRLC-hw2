
cd ~/hw2/tests

make clean
make 2> make-stderr.out
FILE=./test



echo "==================================="
echo "=            Run lab2             ="
echo "==================================="

if [ -f "$FILE" ]; then
    srun -N 1 -n 1 -c 1 ${FILE} 2> run-stderr.out 
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

