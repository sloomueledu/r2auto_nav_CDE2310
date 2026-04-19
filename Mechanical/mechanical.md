# Mechanical Subsystem

## Problem Description & Design Considerations

### Storage
The robot must:
- Store **nine ping pong balls securely**
- Prevent ball escape during motion (including ramps and uneven terrain)
- Ensure the **LiDAR sensor remains unobstructed**

To address this, balls are stored in an **inclined housing arranged around the TurtleBot**. This allows gravity to guide balls toward the feeding mechanism while avoiding vertical stacking that would block the LiDAR.

---

### Launcher
The system must:
- Launch ping pong balls **horizontally**
- Maintain consistent launch velocity and direction
- Fire at controlled intervals (e.g., 2s, 4s, 2s)

Unlike vertical launch systems, horizontal launching requires:
- More precise **alignment**
- Stable **ball feeding timing**
- Consistent **flywheel contact**

---

### Centre of Gravity (CG)
- A high CG increases the risk of tipping, especially on ramps
- Uneven weight distribution may destabilize the robot

Our design mitigates this by:
- Distributing balls **around the robot instead of stacking vertically**
- Using an **additional structural layer** while keeping heavy components low where possible

---

## System Overview

The mechanical system consists of:
- An **inclined circular storage housing** for 9 balls
- A **gravity-assisted feeder mechanism**
- A **rotating arm-based gating system** for controlled ball release
- A **single flywheel launcher** mounted beside the shooting tube
- A **raised LiDAR mount**
- A **multi-layer structural frame** to support all components

<img width="613" height="552" alt="image" src="https://github.com/user-attachments/assets/22c071a0-5618-4004-9890-f41c87f69485" />
<br>
<img width="862" height="845" alt="image" src="https://github.com/user-attachments/assets/400238fd-27d2-4fff-b5d5-64da1a8a7249" />
<br>
<img width="743" height="504" alt="image" src="https://github.com/user-attachments/assets/4992d659-9515-44a5-a8b3-127f88a6708b" />
<br>

### Working Principle
1. Balls are stored in the inclined housing around the robot  
2. Gravity guides balls toward the feeder entrance  
3. A rotating arm mechanism allows balls to enter the shooting tube one at a time  
4. The ball enters the tube and contacts the single flywheel  
5. The flywheel accelerates the ball and launches it horizontally  

**[insert isometric full CAD showing entire robot]**

---

## Key Subsystems

### 1. Storage System
- Balls are arranged in an **inclined housing surrounding the TurtleBot**
- Gravity ensures passive movement toward the feeder
- Prevents LiDAR obstruction

**Advantages:**
- Even weight distribution improves stability  
- No vertical blockage of sensors  
- Efficient use of space around the robot  

**[insert top or angled view showing circular ball storage around robot]**

---

### 2. Ball Feeder Mechanism

The feeder mechanism controls how balls enter the shooting tube.

#### Design
- Uses a **rotating arm driven by a servo motor**
- The arm has **protrusions positioned at 90° intervals**

#### Working Mechanism
- One protrusion blocks the next ball in line  
- Another protrusion allows exactly one ball to drop into the tube  
- As the arm rotates:
  - The previously blocked ball is released  
  - The next ball is simultaneously stopped  

**[insert close-up CAD of rotating arm with protrusions]**

- Ensures **single-ball feeding**
- Prevents **double feeding and jamming**
- Provides **consistent timing control**

**[insert second image showing different arm position (release vs block)]**

---

### 3. Launching Mechanism (Single Flywheel)

- A **single flywheel motor ([motor name])** is mounted on the side of the shooting tube  
- The ball is pressed between the flywheel and the tube wall  

#### Working Principle
- The flywheel spins at high speed  
- Friction between the wheel and ball accelerates the ball forward  
- The ball exits the tube horizontally  

**[insert side view of shooting tube + flywheel + motor]**

#### Design Considerations
- Simpler than double flywheel systems  
- Requires careful tuning to ensure:
  - Sufficient launch speed  
  - Minimal deviation in trajectory  

**[insert zoomed-in view showing ball–flywheel contact region]**

---

### 4. Structural Design

- The robot includes an **additional layer** to support:
  - Storage housing  
  - Feeder mechanism  
  - Launcher assembly  

**[insert side view showing multiple layers]**

- The **LiDAR is elevated** to maintain a clear field of view  

**[insert image clearly showing raised LiDAR relative to structure]**

#### Materials & Fasteners
- Structure primarily uses **3D-printed components**
- Assembly secured using **standard M4 bolts and nuts**

---

## Design Considerations & Alternatives

### Storage Mechanisms

| Type | Pros | Cons | Decision |
|------|------|------|----------|
| Vertical Stack | Easy to implement | Blocks LiDAR | ❌ |
| Reservoir | Easy to implement | Stability issues | ❌ |
| Inclined Circular Housing | Balanced, no obstruction | More complex | ✅ |

---

### Feeding Mechanisms

| Type | Pros | Cons | Decision |
|------|------|------|----------|
| Rotating Arm (Current) | Precise control, prevents double feed | Requires tuning | ✅ |
| Trap Door | Simple | Unreliable on ramps | ❌ |
| Carousel | Controlled | Complex & bulky | ❌ |

---

### Launching Mechanisms

| Type | Pros | Cons | Decision |
|------|------|------|----------|
| Spring / Catapult | Simple | Inconsistent | ❌ |
| Double Flywheel | Stable | More complex, inconsistent direction | ❌ |
| Single Flywheel | Simple, compact | Needs tuning | ✅ |

---

## Testing & Validation

- The feeder mechanism was tested to ensure **consistent one-ball release**
- The rotating arm successfully prevented:
  - Ball stacking
  - Double feeding

- Flywheel testing showed:
  - Adequate launch distance
  - Sensitivity to alignment and speed

---

## Iterative Design Changes

### Feeding Reliability Issues
- Balls occasionally jammed at entry  
**Fix:** Adjusted spacing and protrusion geometry on the rotating arm  

### LiDAR Obstruction
- Initial design partially blocked sensor  
**Fix:** Raised LiDAR to a higher mounting position  

### Structural Stability
- Added layer introduced slight flexing  
**Fix:** Reinforced supports and improved mounting using M4 fasteners  

### Launch Consistency
- Initial shots were inconsistent  
**Fix:** Improved alignment of flywheel and shooting tube  

---

## Final Design Summary

<img width="855" height="837" alt="image" src="https://github.com/user-attachments/assets/57d1a183-fd47-4c57-80b2-5942f1e6d4ea" />
<br>

The final mechanical system:
- Stores 9 balls in an **inclined circular housing**
- Uses a **rotating arm feeder with 90° protrusions** for controlled ball release
- Launches balls **horizontally using a single flywheel system**
- Maintains **sensor visibility** through a raised LiDAR
- Ensures stability through **distributed mass and reinforced structure**

**[insert exploded CAD showing components separated]**
