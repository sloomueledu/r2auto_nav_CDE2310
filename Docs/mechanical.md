#Mechanical Subsystem
Problem Description & Design Considerations
Storage

The robot must:

Store nine ping pong balls securely
Prevent ball escape during motion (including ramps and uneven terrain)
Ensure the LiDAR sensor remains unobstructed

To address this, balls are stored in an inclined housing arranged around the TurtleBot. This allows gravity to guide balls toward the feeding mechanism while avoiding vertical stacking that would block the LiDAR.

Launcher

The system must:

Launch ping pong balls horizontally
Maintain consistent launch velocity and direction
Fire at controlled intervals (e.g., 2s, 4s, 2s)

Unlike vertical launch systems, horizontal launching requires:

More precise alignment
Stable ball feeding timing
Consistent flywheel contact
Centre of Gravity (CG)
A high CG increases the risk of tipping, especially on ramps
Uneven weight distribution may destabilize the robot

Our design mitigates this by:

Distributing balls around the robot instead of stacking vertically
Using an additional structural layer while keeping heavy components low where possible
System Overview

The mechanical system consists of:

An inclined circular storage housing for 9 balls
A gravity-assisted feeder mechanism
A rotating arm-based gating system for controlled ball release
A single flywheel launcher mounted beside the shooting tube
A raised LiDAR mount
A multi-layer structural frame to support all components
Working Principle
Balls are stored in the inclined housing around the robot
Gravity guides balls toward the feeder entrance
A rotating arm mechanism allows balls to enter the shooting tube one at a time
The ball enters the tube and contacts the single flywheel
The flywheel accelerates the ball and launches it horizontally
Key Subsystems
1. Storage System
Balls are arranged in an inclined housing surrounding the TurtleBot
Gravity ensures passive movement toward the feeder
Prevents LiDAR obstruction

Advantages:

Even weight distribution improves stability
No vertical blockage of sensors
Efficient use of space around the robot
2. Ball Feeder Mechanism

The feeder mechanism controls how balls enter the shooting tube.

Design
Uses a rotating arm driven by a servo motor
The arm has protrusions positioned at 90° intervals
Working Mechanism
One protrusion blocks the next ball in line
Another protrusion allows exactly one ball to drop into the tube
As the arm rotates:
The previously blocked ball is released
The next ball is simultaneously stopped
Key Benefits
Ensures single-ball feeding
Prevents double feeding and jamming
Provides consistent timing control
3. Launching Mechanism (Single Flywheel)
A single flywheel motor ([motor name]) is mounted on the side of the shooting tube
The ball is pressed between the flywheel and the tube wall
Working Principle
The flywheel spins at high speed
Friction between the wheel and ball accelerates the ball forward
The ball exits the tube horizontally
Design Considerations
Simpler than double flywheel systems
Requires careful tuning to ensure:
Sufficient launch speed
Minimal deviation in trajectory
4. Structural Design
The robot includes an additional layer to support:
Storage housing
Feeder mechanism
Launcher assembly
The LiDAR is elevated to maintain a clear field of view
Materials & Fasteners
Structure primarily uses 3D-printed components
Assembly secured using standard M4 bolts and nuts
