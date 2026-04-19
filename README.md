![NUS LOGO](./assets/NUSLOGO.png)
# GROUP 2 REPOSITORY FOR CDE2310 - FUNDAMENTALS OF SYSTEMS DESIGN 

## THE TEAM BEHIND   
![Group 2](./assets/photo_2026-04-14_21-22-09.jpg)  
Left to Right: LOO ZHONG EN SAMUEL, UDAIYAN BHAN, ARAVIND THEJUS, REAGAN MOSES WIDJAJA   

## ROBOT PHOTOS  
![FRONT](./assets/Front.jpg)  
Front View  

![SIDE](./assets/Side.jpg)  
Side View  

![BACK](./assets/Back.jpg)  
Back View

## INTRODUCTION  
### MISSION OVERVIEW  
For this project, we are tasked to design and build an Autonomous Mobile Robot that can handle and execute complex warehouse logistics. The robot must be able to self-navigate, identify visual landmarks, and deliver its payloads precisely, and sequentially if required, at each station. As a challenge, one of the stations will have a moving target.

### Mission Objectives
Our mission objectives can be broken down into two stages:   
**Stage 1:** Primary Objectives
+ Identify the Station through Visual Markers
+ Align and dock within allowable docking distance
+ Track the motion profile of the target, if required
+ Unload a payload batch (3 Ping Pong Balls) into the target receptacle, in fixed timing sequence if required    

**Stage 2:** Bonus Objectives
+ Identify the Lift Lobby & Final station through Visual Markers
+ Initiate and Use API Calls to control a lift
+ Safely travel to the second level
+ Unload the final payload batch (3 Ping Pong Balls) at the final station

> **NOTE:** The codebase deployed during the final Mission does not include capabilities to execute the bonus mission. 

### Constraints
The mission has the following constraints:
1. Timing
* Mission set-up, deployment and teardown must be done within 25 minutes
* The Robot System must be designed, built and fully functional by Week 12
2. Navigation
* The Robot must rely on its sensors’ data to map out its surroundings and for navigation.
* Navigation methods which uses line-following is not allowed
3. Environment
* The gaps between maze wall panels may cause issues the LiDAR reading

### REQUIREMENTS  
We have defined our requirements as such:  

### CON-OPS  

## SYSTEM OVERVIEW
The system is a **TurtleBot3-class autonomous mobile platform** with a **Raspberry Pi 4**, **360° LiDAR**, **USB camera**, and a **custom ping-pong payload** (servo-fed chute, flywheel launcher, ultrasonic tripwire). Compute is split: a **remote laptop** runs ROS 2 navigation and mission logic (`auto_nav`, `r2CDE2310_FINAL.py`); an **onboard Raspberry Pi** runs sensor drivers, vision nodes as launched for the mission, and **GPIO** for payload actuation.

The mission loop: consume an **occupancy grid** for planning, **explore** until a valid **ArUco station marker** appears in the docking window, **dock** via polar-arc visual control, publish a **payload command** string, block on **completion** from the Pi, then repeat for the remaining station. **Line-following is not used**; motion is driven by LiDAR, map data, TF pose, and ArUco poses. The deployed evaluation configuration implements **Stage 1** (two stations); bonus lift/API behaviour is not present in the final mission stack (see Mission Objectives note above).

### HIGH LEVEL DESIGN
**Compute partitioning**

| Layer | Function |
|--------|----------|
| **Remote laptop** | `autopilot_node`: subscribes `map`, `scan`; TF (`map` ↔ `base_footprint` / `base_link`) for pose; frontier exploration; `A*` on pooled/inflated grid; **regulated pure pursuit**; **polar-arc ArUco docking**; publishes `cmd_vel`; debug topics `planned_path`, `goal_marker`, `lookahead_marker`; publishes `/gpio_commands`, subscribes `rpi_response`. |
| **Raspberry Pi** | Bring-up as deployed; **USB camera** (`usb_cam`, e.g. `cameralaunch.py`); **ArUco** pipeline (e.g. `usbcam1` → `/usbcam1_markers`); LiDAR over USB; **`rpilivecode`**: subscribes `gpio_commands`, commands **L298N**, **SG90**, **HC-SR04** (Station B), publishes `rpi_response`. |
| **OpenCR / base** | Differential drive from `cmd_vel`; TurtleBot3 motor interface. |

**Navigation stack**

- **Mapping**: `nav_msgs/OccupancyGrid` on `map` feeds grid-based planning (source: deployed SLAM / mapping stack).
- **Exploration**: Frontier search over free/unknown cells until station markers are acquired.
- **Global path**: `A*` search, 8-connected neighbourhood, downsampled grid, optional wall inflation; post-processing per [Software/README.md](./Software/README.md).
- **Local tracking**: **Regulated Pure Pursuit** on `cmd_vel`; LiDAR angular sectors feed obstacle and recovery logic in the control timer.

**Docking and payload**

- **Docking**: Polar-arc law from camera-frame ArUco (`rho`, `alpha`); FOV and marker-loss rules per [Software/README.md](./Software/README.md).
- **Payload handoff**: `autopilot_node` publishes **`A`** or **`B`** on `/gpio_commands`; `rpilivecode` executes Station A (timed drops) or Station B (ultrasonic tripwire); completion strings **`A COMPLETE`**, **`B COMPLETE`** on `rpi_response`.

**Power and structure**: **11.1 V LiPo** → **OpenCR** distribution; Pi **GPIO** for launcher; **USB** for camera and LiDAR. Mechanical: annular storage, feeder, flywheel - see [Mechanical](./Mechanical/mechanical.md), [Electrical](./Electrical/electrical.md).

### INTERFACE CONTROL

## SUBSYSTEM DOCUMENTATIONS
### ELECTRICAL 
[Electrical Documentation](./Electrical/electrical.md)

### MECHANICAL
[Mechanical Documentation](./Mechanical/mechanical.md)

## SOFTWARE CODEBASE   
[Remote Laptop](./Software/Remote_Laptop/)   
[RPI](./Software/RPI/)  

## TESTING DOCUMENTATION

## USER MANUAL  
[User Manual](./General%20Docs/Group2_User_Manual%20-%20Google%20Docs.pdf)

## FINAL RUN VIDEOS LINK   
[Final Run](https://drive.google.com/drive/folders/1luweJNYKmffXvNXEVjMBalpTRKWKpJ6U?usp=sharing)