+ cd /lustre/haven/proj/UTK0150/atown/assignment4
+ echo 'changed into working directory'
++ head -n 1 /var/spool/torque/aux//4747421.apollo-acf
+ export MASTER_ADDR=acf-sc100
+ MASTER_ADDR=acf-sc100
+ export MASTER_PORT=12345
+ MASTER_PORT=12345
++ which python
+ mpirun -n 8 /lustre/haven/proj/UTK0150/envs/pytorch181/bin/python assignment4.py
=>> PBS: job killed: vmem 17223929856 exceeded limit 17179869184
