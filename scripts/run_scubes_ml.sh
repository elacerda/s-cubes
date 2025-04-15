#!/bin/bash
echo "Enter S-PLUS Cloud credentials"
read -p "Username: " username
read -s -p "Password: " password
echo ""
if [ -z "$username" ] || [ -z $password ]
then
    echo "Username and password are mandatory!"
    exit 1
fi

read -p "Data Release: " data_release

read -p "Size Multiplicator (20): " size_multiplicator
if [ -z $size_multiplicator ]
then
    size_multiplicator=20
fi

read -p "Masterlist file (masterlist.csv): " mlfile
if [ -z "$mlfile" ]
then
    mlfile="masterlist.csv"
fi

workdir="scubes_S${size_multiplicator}"
read -p "workdir ($workdir): " workdir_usr
if [ ! -z $workdir_usr ]
then
    workdir=$workdir_usr
fi

imgsdir=${workdir}/images
mkdir -p ${imgsdir}

mskdir=${workdir}/autom_masked_cubes
mkdir -p ${mskdir}

imgsmskdir=${workdir}/autom_masked_cubes/images
mkdir -p ${imgsmskdir}

read -p "Automatic masks arguments file (autom_masks.args): " mskargs
if [ -z "$mskargs" ]
then
    mskargs="autom_masks.args"
fi

for gal in `tail +2 $mlfile | cut -f1 -d','`
do 
        tag="${workdir}/$gal"
        of="${tag}_cube.fits"
        if [[ ! -e $of ]]
        then
            echo "Creating cube for $gal"
	        scubesml -d $data_release -U $username -P $password -w $workdir -S $size_multiplicator -R $gal $mlfile
            mv ${tag}/${gal}_cube.fits ${workdir}/.
            rm -rf $tag
            splots ${of}
            mv ${gal}*.png ${imgsdir}/.
            scube_mask @${mskargs} --no_interact -- ${tag}_cube.fits
            mv ${gal}.fits ${mskdir}/.
            mv ${gal}*.png ${imgsmskdir}/.            
        fi
done
