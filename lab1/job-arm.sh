#!/bin/sh
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:05:00
#SBATCH --partition=normal-arm
#SBATCH --account=f202500010hpcvlabuminhoa


module load GCC/11.3.0

echo "Compiling the code"
echo ""

make clean
make

echo ""
echo "Running the code"
echo ""

./build/social_media 120 1

echo ""
echo "Finished"
