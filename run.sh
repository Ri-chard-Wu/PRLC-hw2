# - To run:
#    sh run.sh 2 3 4



cd ~/hw2/

make clean; make 2> make-stderr.out
RunFile=./hw2

source ./testcases/08.txt



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
    # rm $outFile

else

    echo "==================================="
    echo "=      Print make-stderr.out      ="
    echo "==================================="

    cat make-stderr.out    

fi


echo ""
echo ""

