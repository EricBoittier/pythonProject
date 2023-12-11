%nproc=4
%mem=5760MB
%chk=testjax.chk
#P PBE1PBE/Def2TZVP Symmetry=None scf(maxcycle=200) 

Gaussian input

0 1
O -0.00570308 0.38515872 -0.0
H -0.79607797 -0.194675 -0.0
H 0.801781 -0.1904837 0.0

