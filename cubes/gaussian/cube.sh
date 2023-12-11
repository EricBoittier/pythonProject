source /opt/cluster/programs/g09_d.01/g09/bsd/g09.login.bash


x=testjax.chk
formchk $x $x.fchk;
cubegen 1 Density $x.fchk $x.d.cube -2;
cubegen 1 Potential $x.fchk $x.p.cube -2;
