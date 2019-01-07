Instructions for configuring experimental fog computing analytics testbed in GENI.

=========================================================
Authors: Jon Patman, Dr. Prasad Calyam
Affiliation: University of Missouri-Columbia, USA
=========================================================

=========================================================
I. Transmission Delay Experiment
Workflow: configure SDN controller -> configure openflow switches -> reserve resources in GENI -> configure fog nodes for image processing service

I.a) Testbed Setup:

* - Note: be sure to use the latest distribution located at: https://github.com/junostar/FogAnalyticsTB/archive/master.tar.gz
    Note: Due to storage and download times, the input images have been reduced from the orignal 100 to 10 images for demonstration purposes.

I.a.1) Configure SDN controller:
    a) download the FogATB_ctrl_rspec.xml file from the project repo*
    b) using any instageni Aggregation Manager (AM) and any tool (e.g., omni, Jack) of your choice allocate the SDN controller in GENI
    c) save login information
    d) obtain and save IP address of the controller (e.g., using ifconfig)

I.a.2) Update controller IP address in FogATB rpsec and reserve Emulab-Xen resources in GENI
    a) download Fog_trans_rspec.xml file from the project repo*
    b) open it in any text editor of your choice, and substitute IP address of controller for the main OVS switch (i.e., s1, s2, etc.);
       - to do that find the following command:
         "<execute command="sudo ovs-vsctl set-controller br0 tcp:<ctrl IP addr>:6653" shell="/bin/sh"/>"
       - substitute your ctrl IP address with "<ctrl IP addr>", e.g.,:
         "<execute command="sudo ovs-vsctl set-controller br0 tcp:8.8.8.8:6653" shell="/bin/sh"/>"
    d) save changes    
    e) Using the Omni or Jack tool, import the ATB_trans.spec
    f) Note that the link bandwidths can be permanently throttled by specifying a bandwidth in the link settings.
    f) save login information
    
I.a.3) Run RYU controller and authorize hosts for ssh/scp access without password
    a) login to VCC controller and type the following commands to run POX controller: 
        $ /tmp/ryu/bin/ryu-manager --verbose --port=6653 /tmp/ryu/ryu/app/simple_switch.py
    b) to authorize SCP between each host login to the client node to generate an ssh key:
        $ ssh-keygen -t rsa 
        (do not enter a passphrase)
        $ ssh <user_name>@host1 mkdir -p .ssh
        (NOTE: command will ask your password for host1; to provide it, you always may login to the remote machine and use "sudo passwd <user_name>" command to assign new password)
        $ cat .ssh/id_rsa.pub | ssh <user_name>@host1 'cat >> .ssh/authorized_keys'
        (NOTE: command will ask your password)
    c) similarly, use commands in b) to authorize the client node for h2, h3, and h4 remote accesses 
       (for details refer to http://www.linuxproblem.org/art_9.html)
    d) keep running RYU controller

I.b) Measure Transmission Time
       
I.b.1) Execute script for running transmission experiments
    a) Login to the client node and verify cd to /usr/local and verify the FogAnalytics_TB-master directory is present, if not download it from the repo* using wget
    b) Verify that the hosts are connected and responding using ping, you can also verify network conditions with iperf
    c) cd FogAnalyticsTB-master/scripts # This is the main directory is where the main scripts reside for running experiments and evaluation
    d) You can edit the script file and change the file destination (e.g. h1, h2, etc.) on line #17 where the SCP command is being executed.
    d) $ sudo bash TxExp.sh
    e) The script will process the 10 images located in the /input/ExpImg_10 directory and will produce an output file /output/Tx_output.txt 
    f) The directory .csv files located in the /data directory is a composite of all data from each individual experiment in order to create a range of features values. 

=========================================================

=========================================================
II. Processing Delay Experiment
Workflow: reserve resources in GENI -> configure VM for deep learning application -> conduct experiments to measure processing time

II.a) Testbed Setup:

* - Note: be sure to use the latest distribution located at: https://github.com/junostar/FogAnalyticsTB/archive/master.tar.gz

II.a.1) Reserve ExoGENI resources:
    a) download the FogATB_proc_rspec.xml file from the repo*
    b) Reserve the VMS using any instageni Aggregation Manager (AM) and any tool (e.g., omni, Jack). There are several hardware options available when using ExoGENI nodes (e.g. XOSMall, XOMedium, etc.) so feel free to change hardware configurations as needed.

II.b) Measure Processing Time

I.b.1) Execute script for running transmission experiments
    a) Login to a node of choice and cd to /usr/local. Verify the FogAnalytics_TB-master directory is present, if not download it from the repo* using 'wget https://github.com/junostar/FogAnalyticsTB/archive/master.tar.gz'
    c) It is important to assume that the CPU load of a machine in a realistic environment would not be small, so we will use the Linux stress-ng package which should already be installed.
    d) Create stressors to increase CPU load (%) while processing images:
        $ stress-ng --cpu 0 -l P   # P represents the percentage of CPU load to add (e.g. 25 for 25%)
    b) Execute the Tensorflow object detection script (You might need to login to a new terminal if stress-ng is running:
        $ cd FogAnalytics_TB-master/scripts/
        $ python TpExp.py
    c) The script will use the same set of images as before located in /input/ExpImg_10 to measure the local processing time for the current machine's specs.
    e) The script will process the 10 images located in the /input/ExpImg_10 directory and will produce an output file /output/Tp_output.csv 
    f) The TxExp.csv file located in the /data directory is a composite of all data from each individual experiment in order to create a range of features values for the training dataset. 


=========================================================

=========================================================
III. Predictive Analytics on GENI data using Machine Learning
Workflow: import dataset -> configure machine learning models -> evaluate prediction results 
       
III.a) Model Evaluation Setup:

III.a.1) You can use the composite datasets located in the /data directory or you can compile your own. If you use custom datasets, be sure to modify the AnalyticsPlatform.py script to reflect the appropriate feature columns. In the TxExp.csv file, the target feature we are trying to predict is 'Tx_true', 'Link_Utilization', and 'T_p' for the TpExp.csv file. 

III.b) Evaluate prediction performance of Machine Learning models:
    a) Login to any of the nodes and cd to /usr/local/FogAnalyticsTB-master/scripts and evaluate the ML models against each of the datasets:
        $ python AnalyticsPlatform transmission
        $ python AnalyticsPlatform processing
        $ python AnalyticsPlatform pipeline
    b) The performance results are displayed in the terminal as the 'Mean Average Error (MAE)' and 'Root Mean-Squared Error (RMSE)', respectively.
    c) You can combine the respective errors from both datasets for each model in order to have the total combined error for end-to-end delay predictions.
    d) The time spent training the model and the making predictions is also reported and can be used to gather statistics about deployment latencies. 

III.c) Repeat Sections I-II and vary the feature parameters


=========================================================