# - To run:
#    sh run.sh 2 3 4



cd ~/hw2/

make clean; make 2> make-stderr.out
RunFile=./hw2

source ./testcases/06.txt


# N=2
# n=4
# c=3
# x1=-0.522
# y1=2.874
# z1=1.340
# x2=0.0
# y2=0.0
# z2=0.0
# width=64
# height=64

num_threads=${c}
outFile=out.png

if [ -f "$RunFile" ]; then

    echo "==================================="
    echo "=            Run hw2             ="
    echo "==================================="

    srun -N ${N} -n ${n} -c ${c} $RunFile $num_threads $x1 $y1 $z1 $x2 $y2 $z2 $width $height $outFile 2> run-stderr.out 


    echo "==================================="
    echo "=      Print run-stderr.out       ="
    echo "==================================="

    cat run-stderr.out


    echo "==================================="
    echo "=            Validate             ="
    echo "==================================="

    ./hw2-diff $outFile $valid

else

    echo "==================================="
    echo "=      Print make-stderr.out      ="
    echo "==================================="

    cat make-stderr.out    

fi


echo ""
echo ""

