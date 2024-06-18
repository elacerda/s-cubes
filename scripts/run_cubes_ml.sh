#!/bin/bash
workdir="scubes_S20"
S=20
mlfile="masterlist.csv"

echo "Enter S-PLUS Cloud credentials"
read -s -p "Username:" username
echo ""
read -s -p "Password:" password

for gal in `tail +2 $mlfile | cut -f1 -d','`
do 
        tag="${workdir}/$gal"
        of="${tag}_cube.fits"
        if [[ ! -e $of ]]
        then
				echo "Creating cube for $gal"
				scubesml -U $username -P $password -w $workdir -S $S -R $gal $mlfile
				mv ${tag}/${gal}_cube.fits ${workdir}/.
				rm -rf $tag
        fi
done