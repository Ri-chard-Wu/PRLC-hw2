
cd ~/hw2/sample

make clean; make 2> make-stderr.out
RunFile=./hw2


n_proc=${1}
n_core=${2}
  
num_threads=${n_core}
x1=-0.522
y1=2.874
z1=1.340
x2=0.0
y2=0.0
z2=0.0
width=128
height=128
filename=out.png

if [ -f "$RunFile" ]; then

    echo "==================================="
    echo "=            Run hw2             ="
    echo "==================================="

    srun -n ${n_proc} -c ${n_core} $RunFile $num_threads $x1 $y1 $z1 $x2 $y2 $z2 $width $height $filename 2> run-stderr.out 
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

