![NUS LOGO](./assets/NUSLOGO.png)
# GROUP 2 REPOSITORY FOR CDE2310 - FUNDAMENTALS OF SYSTEMS DESIGN 

## INTRODUCTION  
### MISSION OVERVIEW  
For this project, we are tasked to design and build an Autonomous Mobile Robot that can handle and execute complex warehouse logistics. The robot must be able to self-navigate, identify visual landmarks, and deliver its payloads precisely, and sequentially if required, at each station. As a challenge, one of the stations will have a moving target.

### Mission Objectives
Our mission objectives can be broken down into two stages:
**Stage 1:** Primary Objectives
1. Identify the Station through Visual Markers
1. Align and dock within allowable docking distance
1. Track the motion profile of the target, if required
1. Unload a payload batch (3 Ping Pong Balls) into the target receptacle, in fixed timing sequence if required
**Stage 2:** Bonus Objectives
1. Identify the Lift Lobby & Final station through Visual Markers
1. Initiate and Use API Calls to control a lift
1. Safely travel to the second level
1. Unload the final payload batch (3 Ping Pong Balls) at the final station

> **NOTE:** The codebase deployed during the final Mission does not include capabilities to execute the bonus mission. 

### Constraints
The mission has the following constraints:
1. Timing
* Mission set-up, deployment and teardown must be done within 25 minutes
* The Robot System must be designed, built and fully functional by Week 12
1. Navigation
* The Robot must rely on its sensors’ data to map out its surroundings and for navigation.
* Navigation methods which uses line-following is not allowed
* Environment
1. The gaps between maze wall panels may cause issues the LiDAR reading

### REQUIREMENTS  

### CON-OPS  

### HIGH LEVEL DESIGN  

### INTERFACE CONTROL

## SUBSYSTEM DOCUMENTATIONS
### ELECTRICAL 

### MECHANICAL 

## SOFTWARE CODEBASE   
[Remote Laptop](./Software/Remote_Laptop/)   
[RPI](./Software/RPI/)  

## TESTING DOCUMENTATION

## USER MANUAL  
[User Manual](./General%20Docs/Group2_User_Manual%20-%20Google%20Docs.pdf)

## FINAL RUN VIDEOS LINK   
[Final Run](https://drive.google.com/drive/folders/1luweJNYKmffXvNXEVjMBalpTRKWKpJ6U?usp=sharing)