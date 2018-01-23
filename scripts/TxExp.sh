#!/bin/bash
# Script for recording transmission times of various image sizes on varying bandwidths.

FILES=../input/ExpImg_100/*

echo -e "Filename\tImgHeight\tImgWidth\tDatasize\tTx" >> ../output/Tx_output.txt
startTot=$(date +%s.%N)
for f in $FILES
do
  # TODO: Add code for adding dimensions of images
  
  echo "processing: $f"
  W=`identify $f | cut -f 3 -d " " | sed s/x.*//` #width
  H=`identify $f | cut -f 3 -d " " | sed s/.*x//` #height
  DS=$(wc -c < $f)
  start=$(date +%s.%N)
  exe=$(scp $f h1:/usr/local/ExpImg_100)
  # exe2=$(alexiajp)
  end=$(date +%s.%N)
  DIFF=$(echo "$end - $start" | bc)
  # echo $DIFF >> $output
  echo -e "$f\t$H\t$W\t$DS\t$DIFF" >> ../output/Tx_output.txt
done
echo "Transferring complete!"
endTot=$(date +%s.%N)
DIFF1=$(echo "$endTot - $startTot" | bc)
echo "Total time: $DIFF1"

# sudo echo "export PYTHONPATH=$PYTHONPATH:/local/models/research:/local/models/research/slim" >> /etc/ssh/sshrc