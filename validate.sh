# - To run:
#    sh run.sh 2 3 4



cd ~/hw2/

make clean; make 2> make-stderr.out
RunFile=./hw2


echo "==================================="
echo "=          Validate hw2           ="
echo "==================================="


if [ -f "$RunFile" ]; then

    for i in {1..8} #{1..6}
    do
        echo -n "validating 0$i..."

        source ./testcases/0${i}.txt
        num_threads=${c}
        outFile=out.png

        srun -N ${N} -n ${n} -c ${c} $RunFile $num_threads $x1 $y1 $z1 $x2 $y2 $z2 $width $height $outFile 2> run-stderr.out
        ./hw2-diff $outFile $valid


    done
        

else

    echo "==================================="
    echo "=      Print make-stderr.out      ="
    echo "==================================="

    cat make-stderr.out    

fi




